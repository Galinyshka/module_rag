from __future__ import annotations
import logging
import time
from expander      import QueryExpander
from fact_extractor import FactExtractor
from generation    import GenerationModule, build_context
from models        import ExpandedQuery, QueryType, RAGResponse, RetrievedChunk, VerificationResult
from reranker      import Reranker
from retrieval    import RetrievalModule
from router        import Router
from verification  import VerificationModule
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

MAX_RETRIES  = 1
NO_DATA_MSG  = "К сожалению, в базе знаний недостаточно данных для ответа на этот вопрос."
CLARIFY_MSG    = "Пожалуйста, уточните запрос."
NOT_FOUND_MSG  = "Указанная дисциплина не найдена в базе учебных программ."
IRRELEVANT_MSG = "Этот вопрос не относится к учебным программам (РПД). Я могу помочь только с вопросами по дисциплинам."
SINGLE_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}


class RAGPipeline:

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

    def ask(self, query: str) -> RAGResponse:
        log.info("=== Запрос: %s", query)


        route = self._router.route(query)

        if route.query_type == QueryType.CLARIFY:
            return RAGResponse(
                answer                   = route.message, 
                query_type               = route.query_type,
                is_verified              = True,
                chunks_used              = [],
                fact_extracted           = False,
                verification_note        = "clarification required",
                clarification_candidates = route.disciplines,  
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

  
        log.info("Router определил дисциплины: %s", route.disciplines)

        expanded = self._expander.expand(query, route, route.disciplines)

        # 4. Поиск → реранкинг → извлечение/генерация → верификация
        answer, chunks, verified, fact_extracted = self._run(query, expanded)

        return RAGResponse(
            answer            = answer,
            query_type        = route.query_type,
            is_verified       = verified.is_valid,
            chunks_used       = chunks,
            fact_extracted    = fact_extracted,
            verification_note = verified.note,
            disciplines = route.disciplines
        )


    def _run(self, query: str, expanded: ExpandedQuery,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool]:
        
        if expanded.query_type == QueryType.MULTI_RELATION:
            return self._run_multi_relation(query, expanded)
        if expanded.query_type == QueryType.MULTI_COMPARE:
            return self._run_multi_compare(query, expanded)
        return self._run_single(query, expanded)

    # ── полный цикл для single / multi.global ──────────────────────────────
    def _run_single(self, query: str, expanded: ExpandedQuery,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool]:
        
        """Retry-цикл + fulltext fallback. Используется и напрямую, и из _run_multi_relation."""
        answer         = NO_DATA_MSG
        chunks         = []
        verified       = VerificationResult(is_valid=False, note="не запускалось")
        fact_extracted = False

        for attempt in range(MAX_RETRIES + 1):
            chunks = self._retrieval.retrieve(expanded, reranker=self._reranker)
            if not chunks:
                log.warning("Поиск без результатов (попытка %d).", attempt + 1)
                verified = VerificationResult(is_valid=False, note="нет чанков")
                break

            if expanded.query_type == QueryType.MULTI_GLOBAL:
                chunks = self._reranker.rerank_per_discipline(query, chunks, expanded.disciplines)
            else:
                chunks = self._reranker.rerank(query, chunks)

            if expanded.query_type in {QueryType.SINGLE_GLOBAL, QueryType.MULTI_GLOBAL}:
                chunks = self._retrieval._enrich_with_parents(chunks)

            answer, context   = self._generation.generate(query, chunks, expanded)
            verified = self._verification.verify(query, answer, context, expanded.query_type)

            if verified.is_valid:
                break
                
            if verified.retry and attempt < MAX_RETRIES:
                log.info("Retry (попытка %d/%d) ...", attempt + 1, MAX_RETRIES)
                expanded.paraphrases.append(query + " подробно, развёрнуто")
                continue

            if not verified.retry:
                answer = NO_DATA_MSG
            break

        # fulltext fallback — переехал из ask()
        if not verified.is_valid and expanded.query_type in SINGLE_TYPES and not fact_extracted:
            answer, chunks, verified = self._single_fulltext_fallback(query, expanded, verified)

        return answer, chunks, verified, fact_extracted

    # ── multi.relation: sub_expanded → _run_single → синтез ───────────────
    def _run_multi_relation(self, query: str, expanded: ExpandedQuery,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool]:
        
        sub_answers: list[tuple[str, str]] = []
        all_chunks:  list[RetrievedChunk]  = []
        results = {}

        def process_sub(sub_eq: ExpandedQuery):
            t0 = time.perf_counter()
            log.info("MULTI_RELATION sub START: '%s'", sub_eq.original)
            answer, chunks, verified, _ = self._run_single(sub_eq.original, sub_eq)
            log.info(
                "MULTI_RELATION sub DONE: '%s' — %.1f с",
                sub_eq.original, time.perf_counter() - t0
            )
            return sub_eq.original, answer, chunks

        with ThreadPoolExecutor(max_workers=len(expanded.sub_expanded)) as executor:
            futures = {
                executor.submit(process_sub, sub_eq): sub_eq
                for sub_eq in expanded.sub_expanded
            }
            for future in as_completed(futures):
                original, answer, chunks = future.result()
                results[original] = (answer, chunks)

        # восстанавливаем порядок как в sub_expanded
        for sub_eq in expanded.sub_expanded:
            answer, chunks = results[sub_eq.original]
            all_chunks.extend(chunks)
            if answer != NO_DATA_MSG:
                sub_answers.append((sub_eq.original, answer))
                log.info("sub '%s': %d симв.", sub_eq.original, len(answer))
            else:
                log.warning("Нет данных для подзапроса: '%s'", sub_eq.original)

        if not sub_answers:
            return NO_DATA_MSG, [], VerificationResult(is_valid=False, note="нет ответов по подзапросам"), False

        final_answer, context = self._generation.generate_synthesis(query, sub_answers)
        verified = self._verification.verify(query, final_answer, context, expanded.query_type)
        return final_answer, all_chunks, verified, False
    
    def _run_multi_compare(
        self,
        query: str,
        expanded: ExpandedQuery,
    ) -> tuple[str, list[RetrievedChunk], VerificationResult, bool]:
        """Загружает полные документы всех дисциплин и генерирует сравнение."""

        all_chunks: list[RetrievedChunk] = []

        for discipline in expanded.disciplines:
            doc_chunks = self._retrieval.get_full_document(discipline)
            if doc_chunks:
                all_chunks.extend(doc_chunks)
                log.info("MULTI_COMPARE: загружен '%s' (%d чанков)", discipline, len(doc_chunks))
            else:
                log.warning("MULTI_COMPARE: нет документа для '%s'", discipline)

        if not all_chunks:
            return (
                NO_DATA_MSG,
                [],
                VerificationResult(is_valid=False, note="документы не найдены"),
                False,
            )

        answer, context   = self._generation.generate_compare(query, all_chunks)
        verified = self._verification.verify(query, answer, context, expanded.query_type)

        log.info(
            "MULTI_COMPARE итог: %d дисциплин, verified=%s",
            len(expanded.disciplines), verified.is_valid,
        )
        return answer, all_chunks, verified, False

    def _single_fulltext_fallback(self, query: str, expanded: ExpandedQuery, prev_verified: VerificationResult):
        """Загружает полный документ и регенерирует ответ."""
        if not expanded.disciplines:
            log.warning("Fallback невозможен: дисциплина не определена.")
            return NO_DATA_MSG, [], prev_verified

        discipline = expanded.disciplines[0]
        log.info("Single fallback: полный документ '%s'", discipline)

        full_chunks = self._retrieval.get_full_document(discipline)
        if not full_chunks:
            return NO_DATA_MSG, [], prev_verified

        answer, context   = self._generation.generate_from_context(
            query, build_context(full_chunks), expanded.query_type.value
        )
        verified = self._verification.verify(query, answer, context, expanded.query_type)
        log.info("Single fallback: valid=%s", verified.is_valid)
        return answer, full_chunks, verified
