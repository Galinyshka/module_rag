"""
RAG-пайплайн: оркестратор всех модулей.

Полная цепочка:
  Router
    → QueryExpander          (перефразировки + HyDE + декомпозиция)
    → RetrievalModule        (стратегия по типу + parent retrieval + кэш эмбедингов)
    → Reranker               (cross-encoder переранжирование)
    → FactExtractor          (прямое извлечение для single.simple — если находит)
    → GenerationModule       (если FactExtractor не сработал)
    → VerificationModule
    → RAGResponse

Fallback для single-запросов:
  Если верификатор отклонил и запрос single — загружаем полный документ,
  регенерируем по расширенному контексту.
"""
from __future__ import annotations

import logging

from expander      import QueryExpander
from fact_extractor import FactExtractor
from generation    import GenerationModule, build_context
from models        import QueryType, RAGResponse, VerificationResult
from reranker      import Reranker
from retrieval    import RetrievalModule
from router        import Router
from verification  import VerificationModule

log = logging.getLogger(__name__)

MAX_RETRIES  = 1
NO_DATA_MSG  = "К сожалению, в базе знаний недостаточно данных для ответа на этот вопрос."
CLARIFY_MSG    = "Пожалуйста, уточните запрос."
NOT_FOUND_MSG  = "Указанная дисциплина не найдена в базе учебных программ."
IRRELEVANT_MSG = "Этот вопрос не относится к учебным программам (РПД). Я могу помочь только с вопросами по дисциплинам."
SINGLE_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}


class RAGPipeline:
    """
    Полный RAG-пайплайн для корпуса рабочих программ дисциплин.

    Пример:
        pipeline = RAGPipeline()
        response = pipeline.ask("Сколько часов лекций в дисциплине по Python?")
        print(response.answer)
        print(response.fact_extracted)   # True если ответ взят напрямую из блока
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "discipline_chunks",
    ) -> None:
        log.info("Инициализация RAG-пайплайна ...")
        self._router        = Router()
        self._expander      = QueryExpander()
        self._retrieval     = RetrievalModule(qdrant_url=qdrant_url, collection=collection)
        self._reranker      = Reranker()
        self._fact_extractor = FactExtractor()
        self._generation    = GenerationModule()
        self._verification  = VerificationModule()
        log.info("Пайплайн готов.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str) -> RAGResponse:
        log.info("=== Запрос: %s", query)

        # 1. Маршрутизация
        route = self._router.route(query)

        if route.query_type == QueryType.CLARIFY:
            return RAGResponse(
                answer                   = route.message,  # текст вопроса от LLM
                query_type               = route.query_type,
                is_verified              = True,
                chunks_used              = [],
                fact_extracted           = False,
                verification_note        = "clarification required",
                clarification_candidates = route.disciplines,  # кандидаты для выбора
                disciplines = route.disciplines
            )

        if route.query_type == QueryType.NOT_FOUND:
            return RAGResponse(
                answer            = NOT_FOUND_MSG,
                query_type        = route.query_type,
                is_verified       = True,
                chunks_used       = [],
                fact_extracted    = False,
                verification_note = "discipline not in RPD_NAMES",
                disciplines = route.disciplines
            )

        if route.query_type == QueryType.IRRELEVANT:
            return RAGResponse(
                answer            = IRRELEVANT_MSG,
                query_type        = route.query_type,
                is_verified       = True,
                chunks_used       = [],
                fact_extracted    = False,
                verification_note = "irrelevant query",
                disciplines = route.disciplines
            )

        # 2. Разрешение дисциплин
        resolved = self._retrieval.resolve_disciplines(route.disciplines)

        # 3. Расширение запроса (перефразировки + HyDE + декомпозиция)
        expanded = self._expander.expand(query, route, resolved)

        # 4. Поиск → реранкинг → извлечение/генерация → верификация
        answer, chunks, verified, fact_extracted = self._run(query, expanded)

        # 5. Fallback для single: полный документ при провале верификации
        if not verified.is_valid and route.query_type in SINGLE_TYPES and not fact_extracted:
            answer, chunks, verified = self._single_fulltext_fallback(
                query, expanded, verified
            )
            fact_extracted = False


        return RAGResponse(
            answer            = answer,
            query_type        = route.query_type,
            is_verified       = verified.is_valid,
            chunks_used       = chunks,
            fact_extracted    = fact_extracted,
            verification_note = verified.note,
            disciplines = route.disciplines
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, query: str, expanded):
        """Поиск → реранкинг → FactExtractor или генерация → верификация."""
        answer         = NO_DATA_MSG
        chunks         = []
        verified       = VerificationResult(is_valid=False, note="не запускалось")
        fact_extracted = False

        for attempt in range(MAX_RETRIES + 1):
            # Поиск
            chunks = self._retrieval.retrieve(expanded)
            if not chunks:
                log.warning("Поиск без результатов (попытка %d).", attempt + 1)
                verified = VerificationResult(is_valid=False, note="нет чанков")
                break

            # Реранкинг
            chunks = self._reranker.rerank(query, chunks)

            # FactExtractor — пробуем прямое извлечение для single.simple
            fact = self._fact_extractor.try_extract(query, chunks, expanded.query_type)
            if fact:
                answer         = fact
                fact_extracted = True
                verified       = VerificationResult(is_valid=True, note="direct extraction")
                break

            # Обычная генерация
            answer   = self._generation.generate(query, chunks, expanded)
            verified = self._verification.verify(query, answer, chunks)

            if verified.is_valid:
                break

            if verified.retry and attempt < MAX_RETRIES:
                log.info("Retry поиска (попытка %d/%d) ...", attempt + 1, MAX_RETRIES)
                expanded.paraphrases.append(query + " подробно, развёрнуто")
                continue

            if not verified.retry:
                answer = NO_DATA_MSG
            break

        return answer, chunks, verified, fact_extracted

    def _single_fulltext_fallback(self, query: str, expanded, prev_verified):
        """Загружает полный документ и регенерирует ответ."""
        if not expanded.disciplines:
            log.warning("Fallback невозможен: дисциплина не определена.")
            return NO_DATA_MSG, [], prev_verified

        discipline = expanded.disciplines[0]
        log.info("Single fallback: полный документ '%s'", discipline)

        full_chunks = self._retrieval.get_full_document(discipline)
        if not full_chunks:
            return NO_DATA_MSG, [], prev_verified

        answer   = self._generation.generate_from_context(
            query, build_context(full_chunks), expanded.query_type.value
        )
        verified = self._verification.verify(query, answer, full_chunks)
        log.info("Single fallback: valid=%s", verified.is_valid)
        return answer, full_chunks, verified
