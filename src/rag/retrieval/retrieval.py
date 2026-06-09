from __future__ import annotations
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Prefetch, FusionQuery, Fusion,
)
from rag.domain.models import ExpandedQuery, QueryType, RetrievedChunk

from rag.config.config import (
    EMBED_MODEL, QDRANT_URL, QDRANT_COLLECTION,
    VEC_QUESTIONS, VEC_SUMMARY,
    TOP_K_SINGLE,
    TOP_K_GLOBAL,
    RRF_PREFETCH_K,
    ALL_BLOCKS,
)


log = logging.getLogger(__name__)

class RetrievalModule:
    def __init__(
        self,
        qdrant_url:  str = QDRANT_URL,
        collection:  str = QDRANT_COLLECTION,
        embed_model: str = EMBED_MODEL,

    ) -> None:
        log.info("Загрузка модели эмбедингов: %s ...", embed_model)
        self._embedder   = SentenceTransformer(embed_model)
        self._embed_cache: dict[str, list[float]] = {}
        self._qdrant     = QdrantClient(url=qdrant_url)
        self._collection = collection

    
    def retrieve(self, expanded: ExpandedQuery, reranker=None) -> list[RetrievedChunk]:
        """Каждый тип запроса требует своей стратегии поиска."""
        query_type = expanded.query_type
    
        if query_type == QueryType.MULTI_GLOBAL_SEMANTIC:
            return self._retrieve_multi_global_semantic(expanded)
        #if query_type == QueryType.MULTI_RELATION:
        #   return self._retrieve_multi_relation(expanded, reranker)
        # SINGLE_SIMPLE, SINGLE_GLOBAL и подзапросы MULTI_RELATION (как SINGLE_GLOBAL)
        return self._retrieve_single(expanded)

    def get_full_document(self, discipline: str) -> list[RetrievedChunk]:
        """Собирает всю дисциплину для подачи в модель генерации."""

        result = self._qdrant.scroll(
            collection_name = self._collection,
            scroll_filter   = self._discipline_filter([discipline]),
            limit           = 500,
            with_payload    = True,
            with_vectors    = False,
        )
        points = result[0]
        chunks = [self._point_to_chunk(p, 1.0) for p in points]
        chunks.sort(key=lambda c: ALL_BLOCKS.index(c.block_type) if c.block_type in ALL_BLOCKS else 99)
        log.info("=== Retrieval === Full document '%s': %d блоков", discipline, len(chunks))
        return chunks

    def _retrieve_single(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        '''Для single запросов сразу делаем RRF: один запрос → RRF → реранк по всему корпусу.'''
        chunks = self._multi_query_rrf(
            queries=self._queries(expanded),
            qdrant_filter=self._discipline_filter(expanded.disciplines),
            top_k=TOP_K_SINGLE,
        )
        return chunks

    def _retrieve_multi_global_semantic(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        """
        Поиск по всему корпусу без фильтра по дисциплине.
    
        Особенности:
        - qdrant_filter=None → ищем по всей коллекции
        - top_k=TOP_K_GLOBAL → достаточно, чтобы покрыть все ~50 дисциплин
        - без _enrich_with_parents → не раздуваем результат
        - группировка по дисциплинам происходит в pipeline (_group_chunks_by_discipline)
        """
        chunks = self._multi_query_rrf(
            queries      = self._queries(expanded),
            qdrant_filter = None,           # нет фильтра → весь корпус
            top_k        = TOP_K_GLOBAL,
        )
        chunks = [chunk for chunk in chunks if chunk.score > 0.1] # интересная константа
        log.info(
            "=== Retrieval === MULTI_GLOBAL_SEMANTIC: %d чанков из %d дисциплин",
            len(chunks),
            len({c.discipline for c in chunks}),
        )
        return chunks
        log.info('=== Retrieval === chunks check: %s', chunks)
       
    def _enrich_with_parents(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """ Подтягивает родительские блоки и всех их детей"""
        existing_ids = {c.block_id for c in chunks}
        
        # Шаг 1: Собираем уникальные parent_id
        parent_ids = []
        for c in chunks:
            pid = c.metadata.get("parent_id")
            if pid:
                parent_ids.append(pid)
        
        parent_ids = list(dict.fromkeys(parent_ids))
        if not parent_ids:
            return chunks

        # Шаг 2: Подтягиваем сами родительские блоки
        parent_points = self._qdrant.retrieve(
            collection_name=self._collection,
            ids=parent_ids,
            with_payload=True,
            with_vectors=False,
        )

        parents = []
        valid_parent_ids = []

        for p in parent_points:
            chunk = self._point_to_chunk(p, score=0.0)

            # дополнительно можно отсечь пустые тексты
            if not chunk.text or not chunk.text.strip():
                continue

            parents.append(chunk)
            valid_parent_ids.append(chunk.block_id)

        existing_ids.update(p.block_id for p in parents)
        parent_ids = valid_parent_ids
        
        # Шаг 3: Подтягиваем всех ДЕТЕЙ каждого родителя
        children = []
        for parent_id in parent_ids:
            # Ищем все точки, где metadata.parent_id == parent_id
            result = self._qdrant.scroll(
                collection_name = self._collection,
                scroll_filter   = Filter(must=[
                    FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
                ]),
                limit           = 100,
                with_payload    = True,
                with_vectors    = False,
            )
            
            for point in result[0]:
                #if str(point.id) not in existing_ids:
                children.append(self._point_to_chunk(point, score=0.0))
                    #existing_ids.add(str(point.id))
        
        log.info("=== Retrieval === Parent retrieval: +%d родителей, +%d детей", len(parents), len(children))
        log.info("=== Retrieval === Chunks before build_context: %s",
         [(c.discipline, c.block_name[:20]) for c in chunks])
        
        for p in parents:
            log.debug("=== Retrieval === PARENT BLOCK:\n%s\n", p.text)
        for c in children:
            log.debug("=== Retrieval === CHILD BLOCK:\n%s\n", c.text[:200])

        log.debug("=== Retrieval === PARENT IDS: %s", parent_ids)
        log.debug("=== Retrieval === PARENTS: %s", [p.block_id for p in parents])
        log.debug("=== Retrieval === CHILDREN: %s", [c.block_id for c in children])
        
        result = [*chunks, *parents, *children]
        log.info("=== Retrieval === Total after parent retrieval: %d чанков", len(result))

        #Убираем дубликаты по block_id, сохраняя порядок (parents и children могут пересекаться)
        seen = set()
        unique = []
        for c in result:
            key = c.block_id
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        log.info("=== Retrieval === Total after deduplication: %d чанков", len(unique))
        return unique
    
    # ------------------------------------------------------------------
    # RRF поиск
    # ------------------------------------------------------------------

    def _queries(self, expanded: ExpandedQuery) -> list[str]:
        '''Объединяет все варианты запросов в один список для RRF.'''
        
        queries = [expanded.original, *expanded.paraphrases]
        if expanded.hyde_text:
            queries.append(expanded.hyde_text)
        queries.extend(expanded.sub_queries)
        log.info("=== Retrieval === Number of queries for retrieval: %d", len(queries))
        return queries

    def _multi_query_rrf(self, queries: list[str], qdrant_filter, top_k: int) -> list[RetrievedChunk]:
        """
        RRF по нескольким запросам.

        Для каждого запроса Qdrant делает prefetch по questions_vec и summary_vec,
        затем объединяет через Fusion.RRF — score блока определяется его позицией
        в обоих рейтингах, а не абсолютным cosine-score.

        Результаты по всем запросам объединяются на Python-уровне:
        берём максимальный RRF-score блока среди всех запросов.
        """
        unique = list(dict.fromkeys(queries))
        self._warm_cache(unique)

        seen:   dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}

        for query in unique:
            log.info("=== Retrieval === RRF search: query=%s", query)
            # берем эмбединг запроса из кеша
            query_vec = self._embed_cached(query)
            for i, chunk in enumerate(self._rrf_search(query_vec, qdrant_filter, top_k)):
                if chunk.block_id not in seen:
                    seen[chunk.block_id]   = chunk
                    scores[chunk.block_id] = chunk.score
                else:
                    scores[chunk.block_id] = max(scores[chunk.block_id], chunk.score)

                log.info("=== Retrieval === Chunk № %d: score=%.3f, discipline=%s, block=%s",
                         i+1, chunk.score, chunk.discipline, chunk.block_name)
        
        chunks = self._sort_by_score(seen, scores)
        log.info("=== Retrieval === RRF search: найдено %d уникальных чанков", len(chunks))

        chunks = chunks[:top_k * len(unique)]
        log.info("=== Retrieval === RRF search: после обрезки до top_k*кол-во_запросов (%d) чанков", len(chunks))
        
        if chunks:
            sc = [scores[chunk.block_id] for chunk in chunks]
            log.info("=== Retrieval === RRF scores (%d chunks): min=%.3f max=%.3f avg=%.3f",
                    len(chunks), min(sc), max(sc), sum(sc)/len(sc))
        
        return chunks


    def _rrf_search(self, query_vec: list[float], qdrant_filter, top_k: int) -> list[RetrievedChunk]:
        """
        Один RRF-запрос в Qdrant:
          prefetch questions_vec → prefetch summary_vec → Fusion.RRF → top_k.

        questions_vec ловит запросы вида "какие вопросы на контрольной".
        summary_vec   ловит запросы вида "расскажи про самостоятельную работу".
        RRF объединяет оба рейтинга без необходимости подбирать веса.
        """
        try:
            result = self._qdrant.query_points(
                collection_name = self._collection,
                prefetch        = [
                    Prefetch(
                        query        = query_vec,
                        using        = VEC_QUESTIONS,
                        filter       = qdrant_filter,
                        limit        = RRF_PREFETCH_K,
                    ),
                    Prefetch(
                        query        = query_vec,
                        using        = VEC_SUMMARY,
                        filter       = qdrant_filter,
                        limit        = RRF_PREFETCH_K,
                    ),
                ],
                query           = FusionQuery(fusion=Fusion.RRF),
                limit           = top_k,
                with_payload    = True,
            )
            chunks = [self._point_to_chunk(h, h.score) for h in result.points]
            return chunks
        
        except Exception as exc:
            log.exception("=== Retrieval === RRF недоступен (%s)", exc)


    # ------------------------------------------------------------------
    # Кэш эмбедингов
    # ------------------------------------------------------------------

    def _warm_cache(self, texts: list[str]) -> None:
        missing = [t for t in texts if t not in self._embed_cache]
        if not missing:
            return
        vecs = self._embedder.encode(missing, normalize_embeddings=True,
                                     show_progress_bar=False)
        for text, vec in zip(missing, vecs):
            self._embed_cache[text] = vec.tolist()

    def _embed_cached(self, text: str) -> list[float]:
        if text not in self._embed_cache:
            self._warm_cache([text])
        return self._embed_cache[text]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sort_by_score(self, chunks: dict[str, RetrievedChunk], scores: dict[str, float]) -> list[RetrievedChunk]:
        ''''Сортирует чанки по score, который может быть в двух местах: либо в поле chunk.score (для RRF), либо в отдельном словаре (для реранкера).'''
        ranked = sorted(chunks.values(), key=lambda c: scores[c.block_id], reverse=True)
        for c in ranked:
            c.score = scores[c.block_id]
        # возвращаем отсортированные по убыванию чанки 
        return ranked

    def _point_to_chunk(self, point, score: float) -> RetrievedChunk:
        p = point.payload or {}
        return RetrievedChunk(
            block_id   = str(point.id),
            block_type = p.get("block_type", ""),
            block_name = p.get("block_name", ""),
            text       = p.get("text", ""),
            summary    = p.get("summary", ""),
            discipline = p.get("discipline", ""),
            score      = score,
            metadata   = {k: v for k, v in p.items()
                          if k not in ("text", "summary", "block_name", "block_type")},
        )

    def _discipline_filter(self, disciplines: list[str]) -> Filter | None:
        '''Создаёт фильтр в бд для поиска по дисциплинам'''
        if not disciplines:
            return None
        if len(disciplines) == 1:
            return Filter(must=[FieldCondition(
                key="discipline", match=MatchValue(value=disciplines[0])
            )])
        return Filter(must=[FieldCondition(
            key="discipline", match=MatchAny(any=disciplines)
        )])
