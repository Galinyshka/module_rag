from __future__ import annotations
import logging
import time
from expander      import QueryExpander
from generation    import GenerationModule, build_context
from models        import ExpandedQuery, QueryType, RAGResponse, RetrievedChunk, VerificationResult
from reranker      import Reranker
from retrieval    import RetrievalModule
from router        import Router, RouteResult
from verification  import VerificationModule
from catalog import DisciplineCatalog
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

MAX_RETRIES  = 1
NO_DATA_MSG  = "К сожалению, в базе знаний недостаточно данных для ответа на этот вопрос."
CLARIFY_MSG    = "Пожалуйста, уточните запрос."
NOT_FOUND_MSG  = "Указанная дисциплина не найдена в базе учебных программ."
IRRELEVANT_MSG = "Этот вопрос не относится к учебным программам (РПД). Я могу помочь только с вопросами по дисциплинам."
SINGLE_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}

MULTI_GLOBAL_TYPES = {
    QueryType.MULTI_GLOBAL_CATALOG,
    QueryType.MULTI_GLOBAL_SEMANTIC,
}

class RAGPipeline:

    def __init__(self, qdrant_url: str = "http://localhost:6333", collection: str = "discipline_chunks",
                 ) -> None:
        
        log.info("=== Pipeline === Инициализация RAG-пайплайна ...")
        self._router        = Router()
        self._expander      = QueryExpander()
        self._retrieval     = RetrievalModule(qdrant_url=qdrant_url, collection=collection)
        self._reranker      = Reranker()
        self._generation    = GenerationModule()
        self._verification  = VerificationModule()
        self._catalog = DisciplineCatalog("test_data")
        log.info("=== Pipeline === Пайплайн инициализирован.")

    def ask(self, query: str) -> RAGResponse:
        log.info("=== Pipeline === Запрос: %s", query)

        route = self._router.route(query)

        if route.query_type == QueryType.CLARIFY:
            return RAGResponse(
                answer                   = route.message, 
                query_type               = route.query_type,
                is_verified              = True,
                chunks_used              = [],
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
                verification_note = "discipline not in RPD_NAMES",
                disciplines = route.disciplines
            )

        if route.query_type == QueryType.IRRELEVANT:
            return RAGResponse(
                answer            = IRRELEVANT_MSG,
                query_type        = route.query_type,
                is_verified       = True,
                chunks_used       = [],
                verification_note = "irrelevant query",
                disciplines = route.disciplines
            )

        log.info("=== Pipeline === Router определил дисциплины: %s", route.disciplines)

        answer, chunks, verified  = self._run(query, route)

        return RAGResponse(
            answer            = answer,
            query_type        = route.query_type,
            is_verified       = verified.is_valid,
            chunks_used       = chunks,
            verification_note = verified.note,
            disciplines = route.disciplines
        )

    def _run(self, query: str, route: RouteResult,
             ) -> tuple[str, list[RetrievedChunk], VerificationResult]:
        
        # MULTI_GLOBAL не нуждается в expander
        if route.query_type in MULTI_GLOBAL_TYPES:
            return self._run_multi_global(query, route)

        expanded = self._expander.expand(query, route, route.disciplines, expanded_flag=False)

        if route.query_type == QueryType.MULTI_RELATION:
            return self._run_multi_relation(query, route, expanded)
        if route.query_type == QueryType.MULTI_COMPARE:
            return self._run_multi_compare(query, expanded)
        return self._run_single(query, route, expanded)

    def _run_single(self, query: str, route: RouteResult, expanded: ExpandedQuery
                    ) -> tuple[str, list[RetrievedChunk], VerificationResult]:
        
        # Шаг 1: original only (expanded_flag=False: нет перефразов)
        chunks, verified, answer = self._generate_and_verify(query, expanded, step="1/3")
        if verified.is_valid:
            return answer, chunks, verified

        # Шаг 2: с парафразами (expanded_flag=True)
        expanded_v2 = self._expander.expand(query, route, route.disciplines, expanded_flag=True)
        chunks, verified, answer = self._generate_and_verify(query, expanded_v2, step="2/3")
        if verified.is_valid:
            return answer, chunks, verified

        # Шаг 3: fulltext fallback
        if expanded_v2.query_type in SINGLE_TYPES:
            answer, chunks, verified = self._single_fulltext_fallback(query, expanded_v2, verified)
            verified.note = f"[3/3] {verified.note}"
        else:
            answer, chunks = NO_DATA_MSG, []

        return answer, chunks, verified


    def _run_multi_relation(self, query: str, route: RouteResult, expanded: ExpandedQuery,
                            ) -> tuple[str, list[RetrievedChunk], VerificationResult]:

        # Шаг 1: если sub_expanded пуст (expanded_flag=False) — это нормально, расширяем
        if not expanded.sub_expanded:
            log.info("=== Pipeline === (%s): первый проход, делаем декомпозицию.", route.query_type)
            expanded = self._expander.expand(query, route, route.disciplines, expanded_flag=True)

        # После расширения всё ещё пусто — реальная ошибка
        if not expanded.sub_expanded:
            log.warning("=== Pipeline === (%s): sub_expanded пуст даже после расширения.", route.query_type)
            return NO_DATA_MSG, [], VerificationResult(is_valid=False, note="sub_expanded пуст")

        sub_answers: list[tuple[str, str]] = []
        all_chunks:  list[RetrievedChunk]  = []
        results = {}

        def process_sub(sub_eq: ExpandedQuery):
            t0 = time.perf_counter()
            log.info("=== Pipeline === (%s) sub START: '%s'", route.query_type, sub_eq.original)
            sub_route = RouteResult(query_type=sub_eq.query_type, disciplines=sub_eq.disciplines)
            answer, chunks, verified = self._run_single(sub_eq.original, sub_route, sub_eq)
            log.info(
                "=== Pipeline === (%s) sub DONE: '%s' — %.1f с",
                route.query_type, sub_eq.original, time.perf_counter() - t0,
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

        for sub_eq in expanded.sub_expanded:
            answer, chunks = results[sub_eq.original]
            all_chunks.extend(chunks)
            if answer != NO_DATA_MSG:
                sub_answers.append((sub_eq.original, answer))
                log.info("=== Pipeline === sub '%s': %d симв.", sub_eq.original, len(answer))
            else:
                log.warning("=== Pipeline === Нет данных для подзапроса: '%s'", sub_eq.original)

        if not sub_answers:
            return NO_DATA_MSG, [], VerificationResult(is_valid=False, note="нет ответов по подзапросам")

        final_answer, context = self._generation.generate_synthesis(query, sub_answers)
        verified = self._verification.verify(query, final_answer, context, expanded.query_type)
        return final_answer, all_chunks, verified

    def _run_multi_compare(self, query: str, expanded: ExpandedQuery,
                           ) -> tuple[str, list[RetrievedChunk], VerificationResult]:
        """Загружает полные документы всех дисциплин и генерирует сравнение."""
        all_chunks: list[RetrievedChunk] = []

        for discipline in expanded.disciplines:
            doc_chunks = self._retrieval.get_full_document(discipline)
            if doc_chunks:
                all_chunks.extend(doc_chunks)
                log.info("=== Pipeline === MULTI_COMPARE: загружен '%s' (%d чанков)", discipline, len(doc_chunks))
            else:
                log.warning("=== Pipeline === MULTI_COMPARE: нет документа для '%s'", discipline)

        if not all_chunks:
            return NO_DATA_MSG, [], VerificationResult(is_valid=False, note="документы не найдены")

        answer, context = self._generation.generate(query, all_chunks, expanded.query_type.value)
        verified = self._verification.verify(query, answer, context, expanded.query_type)
        log.info(
            "=== Pipeline === MULTI_COMPARE итог: %d дисциплин, verified=%s",
            len(expanded.disciplines), verified.is_valid,
        )
        return answer, all_chunks, verified


    def _run_multi_global(self, query: str, route: RouteResult,
                          ) -> tuple[str, list[RetrievedChunk], VerificationResult]:
        """ Обработчик multi.global. Catalog - через контекст из каталога, Semantic - через семантический поиск по всему корпусу."""
        
        if route.query_type == QueryType.MULTI_GLOBAL_CATALOG:
            context = self._catalog.as_llm_context(mode="full")
            answer, ctx = self._generation.generate_from_context(
                query, context, route.query_type.value
            )
            verified = self._verification.verify(query, answer, ctx, route.query_type)
            return answer, [], verified

        if route.query_type == QueryType.MULTI_GLOBAL_SEMANTIC:
            return self._run_multi_global_semantic(query, route)


    def _run_multi_global_semantic(self, query: str, route: RouteResult,
                                   ) -> tuple[str, list[RetrievedChunk], VerificationResult]:
        """
        Семантический поиск по всему корпусу.
        Ищет без фильтра по дисциплине, группирует результаты,
        берёт top-3 чанка на дисциплину.
        """
        # ExpandedQuery без дисциплин → RetrievalModule ищет по всему корпусу
        expanded = ExpandedQuery(
            original   = query,
            query_type = route.query_type,
            disciplines = [],       # нет фильтра по дисциплине
            paraphrases = [],
            hyde_text   = None,
            sub_queries = [],
        )

        chunks = self._retrieval.retrieve(expanded, reranker=None)
        if not chunks:
            return NO_DATA_MSG, [], VerificationResult(is_valid=False, note="нет чанков")

        grouped = self._group_chunks_by_discipline(chunks, top_k=3)
        log.info("=== Pipeline === MULTI_GLOBAL_SEMANTIC: %d дисциплин", len(grouped))
        selected = [c for disc_chunks in grouped.values() for c in disc_chunks]

        answer, ctx = self._generation.generate(query, selected, route.query_type.value)
        verified = self._verification.verify(query, answer, ctx, route.query_type)
        return answer, selected, verified

    def _group_chunks_by_discipline(self, chunks: list[RetrievedChunk], top_k: int = 3,
                                    ) -> dict[str, list[RetrievedChunk]]:
        """
        Группирует чанки по дисциплине, оставляя top_k лучших на дисциплину.
        Порядок чанков сохраняется (они уже отсортированы по score).
        """
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            discipline = chunk.discipline   # поле в RetrievedChunk
            if discipline not in grouped:
                grouped[discipline] = []
            if len(grouped[discipline]) < top_k:
                grouped[discipline].append(chunk)
        return grouped

    def _single_fulltext_fallback(self, query: str, expanded: ExpandedQuery, prev_verified: VerificationResult):
        """Загружает полный документ и регенерирует ответ."""
        if not expanded.disciplines:
            log.warning("=== Pipeline === Fallback невозможен: дисциплина не определена.")
            return NO_DATA_MSG, [], prev_verified

        discipline = expanded.disciplines[0]
        log.info("=== Pipeline === Single fallback: полный документ '%s'", discipline)

        full_chunks = self._retrieval.get_full_document(discipline)
        if not full_chunks:
            return NO_DATA_MSG, [], prev_verified

        answer, context   = self._generation.generate_from_context(
            query, build_context(full_chunks), expanded.query_type.value
        )
        verified = self._verification.verify(query, answer, context, expanded.query_type)
        log.info("=== Pipeline === Single fallback: valid=%s", verified.is_valid)
        return answer, full_chunks, verified

    def _generate_and_verify(self, query: str, expanded: ExpandedQuery, *, step: str = "",
                             ) -> tuple[list[RetrievedChunk], VerificationResult, str]:
        
        chunks = self._retrieve_chunks(query, expanded)

        if not chunks:
            log.warning("=== Pipeline === Чанки не найдены.")
            note = "нет чанков"
            return [], VerificationResult(is_valid=False, note=f"[{step}] {note}" if step else note), NO_DATA_MSG

        answer, context = self._generation.generate(query, chunks, expanded.query_type.value)
        verified = self._verification.verify(query, answer, context, expanded.query_type)

        verified.note = f"[{step}] {verified.note}" if step else verified.note
        log.info("=== Pipeline === Генерация и верификация: valid=%s, %d чанков, note=%s",
                 verified.is_valid, len(chunks), verified.note)
        return chunks, verified, answer

    def _retrieve_chunks(self, query: str, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        chunks = self._retrieval.retrieve(expanded, reranker=self._reranker)
        chunks = self._reranker.rerank(query, chunks)
        if expanded.query_type == QueryType.SINGLE_GLOBAL:
            chunks = self._retrieval._enrich_with_parents(chunks)
        return chunks

'''    def _retrieve_chunks(self, query: str, expanded: ExpandedQuery,
                         ) -> list[RetrievedChunk]:
        
        chunks = self._retrieval.retrieve(expanded, reranker=self._reranker)
        if expanded.query_type == QueryType.MULTI_GLOBAL:
            chunks = self._reranker.rerank_per_discipline(query, chunks, expanded.disciplines)
        else:
            chunks = self._reranker.rerank(query, chunks)
        if expanded.query_type in {QueryType.SINGLE_GLOBAL, QueryType.MULTI_GLOBAL}:
            chunks = self._retrieval._enrich_with_parents(chunks)
        return chunks'''