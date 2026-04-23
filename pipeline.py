"""
RAG-пайплайн: оркестратор всех модулей.

Полная цепочка:
  Router -> QueryExpander -> RetrievalModule -> GenerationModule
         -> VerificationModule -> TimeFilter -> RAGResponse

Fallback для single-запросов:
  Если верификатор отклонил ответ (retry=False) и запрос single — загружаем
  полный документ дисциплины и регенерируем ответ по расширенному контексту.
"""
from __future__ import annotations

import logging

from .expander     import QueryExpander
from .generation   import GenerationModule, build_context
from .models       import QueryType, RAGResponse, VerificationResult
from .retrieval    import RetrievalModule
from .router       import Router
from .time_filter  import TimeFilter
from .verification import VerificationModule

log = logging.getLogger(__name__)

MAX_RETRIES  = 1
NO_DATA_MSG  = "К сожалению, в базе знаний недостаточно данных для ответа на этот вопрос."
SINGLE_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}


class RAGPipeline:
    """
    Полный RAG-пайплайн для корпуса рабочих программ дисциплин.

    Пример:
        pipeline = RAGPipeline()
        response = pipeline.ask("Сколько часов лекций в дисциплине по Python?")
        print(response.answer)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "discipline_chunks",
    ) -> None:
        log.info("Инициализация RAG-пайплайна ...")
        self._router       = Router()
        self._expander     = QueryExpander()
        self._retrieval    = RetrievalModule(qdrant_url=qdrant_url, collection=collection)
        self._generation   = GenerationModule()
        self._verification = VerificationModule()
        self._time_filter  = TimeFilter()
        log.info("Пайплайн готов.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str) -> RAGResponse:
        log.info("=== Запрос: %s", query[:120])

        # 1. Маршрутизация
        route = self._router.route(query)

        # 2. Разрешение дисциплин
        resolved = self._retrieval.resolve_disciplines(route.disciplines)

        # 3. Расширение запроса
        expanded = self._expander.expand(query, route, resolved)

        # 4. Поиск -> генерация -> верификация (с retry)
        answer, chunks, verified = self._search_generate_verify(query, expanded)

        # 5. Fallback для single: полный документ при провале верификации
        if not verified.is_valid and route.query_type in SINGLE_TYPES:
            answer, chunks, verified = self._single_fulltext_fallback(
                query, expanded, verified
            )

        # 6. Time Filter — постпроцессинг в конце цепочки
        answer = self._time_filter.apply(query, answer)

        return RAGResponse(
            answer            = answer,
            query_type        = route.query_type,
            is_verified       = verified.is_valid,
            chunks_used       = chunks,
            verification_note = verified.note,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_generate_verify(self, query: str, expanded):
        """Поиск -> генерация -> верификация с одним retry."""
        answer   = NO_DATA_MSG
        chunks   = []
        verified = VerificationResult(is_valid=False, note="не запускалось")

        for attempt in range(MAX_RETRIES + 1):
            chunks = self._retrieval.retrieve(expanded)

            if not chunks:
                log.warning("Поиск без результатов (попытка %d).", attempt + 1)
                verified = VerificationResult(is_valid=False,
                                              note="нет релевантных чанков")
                break

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

        return answer, chunks, verified

    def _single_fulltext_fallback(self, query: str, expanded, prev_verified):
        """
        Fallback: загружаем полный текст документа дисциплины и регенерируем ответ.
        Используется когда векторный поиск не нашёл достаточного контекста.
        """
        if not expanded.disciplines:
            log.warning("Fallback невозможен: дисциплина не определена.")
            return NO_DATA_MSG, [], prev_verified

        discipline = expanded.disciplines[0]
        log.info("Single fallback: полный документ '%s'", discipline)

        full_chunks = self._retrieval.get_full_document(discipline)
        if not full_chunks:
            return NO_DATA_MSG, [], prev_verified

        # Строим контекст из полного документа
        full_context = build_context(full_chunks)
        answer       = self._generation.generate_from_context(
            query, full_context, expanded.query_type.value
        )
        verified     = self._verification.verify(query, answer, full_chunks)

        log.info("Single fallback: valid=%s", verified.is_valid)
        return answer, full_chunks, verified
