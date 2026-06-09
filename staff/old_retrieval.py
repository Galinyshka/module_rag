"""
Гибридный модуль поиска (Retrieval).

Улучшения:
  - Кэш эмбедингов: каждый уникальный текст эмбедируется один раз,
    вектор переиспользуется для VEC_TEXT и VEC_SUMMARY поиска.
  - HyDE: если ExpandedQuery содержит hyde_text — он используется
    как дополнительный поисковый запрос.
  - Parent retrieval: после поиска автоматически подтягиваются
    родительские блоки для найденных дочерних.

Стратегии:
  single.*       — фильтр по дисциплине + семантический поиск
  multi.relation — независимый поиск по каждой дисциплине
  multi.global   — двухэтапный: обзорные блоки → дисциплины → точный поиск
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from src.rag.config.config import (
    EMBED_MODEL, QDRANT_URL, QDRANT_COLLECTION,
    VEC_TEXT, VEC_SUMMARY,
)
from src.rag.domain.models import ExpandedQuery, QueryType, RetrievedChunk

log = logging.getLogger(__name__)

TOP_K_SINGLE          = 6
TOP_K_STAGE1          = 30
TOP_K_PER_DISC        = 8
MATCH_THRESHOLD       = 0.55
MAX_DISCIPLINES_MULTI = 10
OVERVIEW_BLOCKS       = ["course_info", "topics", "competencies"]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class RetrievalModule:
    def __init__(
        self,
        qdrant_url:  str = QDRANT_URL,
        collection:  str = QDRANT_COLLECTION,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        log.info("Загрузка модели эмбедингов: %s ...", embed_model)
        self._embedder   = SentenceTransformer(embed_model)
        self._embed_cache: dict[str, list[float]] = {}   # кэш эмбедингов
        self._qdrant     = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._discipline_index: list[str] = self._load_discipline_names()
        log.info("Индекс дисциплин: %d записей", len(self._discipline_index))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        qt = expanded.query_type
        if qt == QueryType.MULTI_GLOBAL:
            chunks = self._retrieve_multi_global(expanded)
        elif qt == QueryType.MULTI_RELATION:
            chunks = self._retrieve_multi_relation(expanded)
        else:
            chunks = self._retrieve_single(expanded)

        # Parent retrieval: подтягиваем родительские блоки
        return self._enrich_with_parents(chunks)

    def get_full_document(self, discipline: str) -> list[RetrievedChunk]:
        """Все блоки дисциплины — для fallback после провала верификации."""
        ORDER = ["course_info", "topics", "competencies", "topic", "competency",
                 "self_study_resources", "assessment_fund", "literature",
                 "online_resources", "other_sections", "other_section"]
        result = self._qdrant.scroll(
            collection_name = self._collection,
            scroll_filter   = self._discipline_filter([discipline]),
            limit           = 500,
            with_payload    = True,
            with_vectors    = False,
        )
        points = result[0]
        chunks = [self._point_to_chunk(p, 1.0) for p in points]
        chunks.sort(key=lambda c: ORDER.index(c.block_type)
                    if c.block_type in ORDER else 99)
        log.info("Full document '%s': %d блоков", discipline, len(chunks))
        return chunks

    def resolve_disciplines(self, raw_names: list[str]) -> list[str]:
        resolved = []
        for name in raw_names:
            best_score, best_match = 0.0, ""
            for idx_name in self._discipline_index:
                s = _sim(name, idx_name)
                if s > best_score:
                    best_score, best_match = s, idx_name
            if best_score >= MATCH_THRESHOLD and best_match:
                resolved.append(best_match)
                log.info("'%s' -> '%s' (%.2f)", name, best_match, best_score)
            else:
                log.warning("'%s' не распознана (max=%.2f)", name, best_score)
        return list(dict.fromkeys(resolved))

    # ------------------------------------------------------------------
    # Стратегии
    # ------------------------------------------------------------------

    def _retrieve_single(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        return self._multi_query_search(
            self._queries(expanded),
            self._discipline_filter(expanded.disciplines),
            TOP_K_SINGLE,
        )

    def _retrieve_multi_relation(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        """Независимый поиск по каждой дисциплине — гарантированное покрытие."""
        disciplines = expanded.disciplines or self._discipline_index
        queries     = self._queries(expanded)
        merged: dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}

        for disc in disciplines:
            for c in self._multi_query_search(
                queries, self._discipline_filter([disc]), TOP_K_PER_DISC
            ):
                if c.block_id not in merged:
                    merged[c.block_id] = c
                    scores[c.block_id] = c.score
                else:
                    scores[c.block_id] = max(scores[c.block_id], c.score)

        return self._sort_by_score(merged, scores)

    def _retrieve_multi_global(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        """
        Stage 1 — обзорные блоки по всему корпусу → выявляем дисциплины.
        Stage 2 — точный поиск по выявленным дисциплинам.
        """
        queries = self._queries(expanded)

        # Stage 1
        stage1 = self._multi_query_search(
            queries,
            Filter(must=[FieldCondition(
                key="block_type", match=MatchAny(any=OVERVIEW_BLOCKS)
            )]),
            TOP_K_STAGE1,
        )
        relevant = list(dict.fromkeys(
            c.discipline for c in stage1 if c.discipline
        ))[:MAX_DISCIPLINES_MULTI]
        log.info("multi.global Stage 1: %d дисциплин", len(relevant))

        if not relevant:
            return stage1

        # Stage 2
        stage2 = self._multi_query_search(
            queries, self._discipline_filter(relevant), TOP_K_PER_DISC
        )

        merged: dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}
        for c in [*stage1, *stage2]:
            if c.block_id not in merged:
                merged[c.block_id] = c
                scores[c.block_id] = c.score
            else:
                scores[c.block_id] = max(scores[c.block_id], c.score)

        log.info("multi.global Stage 2: итого %d чанков", len(merged))
        return self._sort_by_score(merged, scores)

    # ------------------------------------------------------------------
    # Parent retrieval
    # ------------------------------------------------------------------

    def _enrich_with_parents(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Для каждого дочернего блока (у которого есть parent_id)
        подтягивает родительский блок, если его ещё нет в результатах.
        """
        existing_ids = {c.block_id for c in chunks}
        parent_ids   = [
            c.metadata["parent_id"]
            for c in chunks
            if c.metadata.get("parent_id") and c.metadata["parent_id"] not in existing_ids
        ]
        parent_ids = list(dict.fromkeys(parent_ids))  # дедупликация
        if not parent_ids:
            return chunks

        parent_points = self._qdrant.retrieve(
            collection_name = self._collection,
            ids             = parent_ids,
            with_payload    = True,
            with_vectors    = False,
        )
        parents = [self._point_to_chunk(p, score=0.0) for p in parent_points]
        log.info("Parent retrieval: +%d родительских блоков", len(parents))
        return [*chunks, *parents]

    # ------------------------------------------------------------------
    # Поиск + кэш эмбедингов
    # ------------------------------------------------------------------

    def _queries(self, expanded: ExpandedQuery) -> list[str]:
        """Собирает все тексты для поиска: оригинал + перефразировки + HyDE + подзапросы."""
        queries = [expanded.original, *expanded.paraphrases]
        if expanded.hyde_text:
            # HyDE идёт первым после оригинала — наиболее близок к языку документов
            queries.insert(1, expanded.hyde_text)
        queries.extend(expanded.sub_queries)
        return queries

    def _multi_query_search(
        self,
        queries:      list[str],
        qdrant_filter,
        top_k:        int,
    ) -> list[RetrievedChunk]:
        """
        Поиск по нескольким запросам × 2 вектора.
        Эмбединги кэшируются: один уникальный текст → один вызов модели.
        """
        seen:   dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}

        # Предварительно эмбедируем все уникальные запросы одним батчем
        unique = list(dict.fromkeys(queries))
        self._warm_cache(unique)

        for q in queries:
            vec = self._embed_cached(q)
            for vec_name in (VEC_TEXT, VEC_SUMMARY):
                for c in self._vector_search(vec, vec_name, qdrant_filter, top_k):
                    if c.block_id not in seen:
                        seen[c.block_id]   = c
                        scores[c.block_id] = c.score
                    else:
                        scores[c.block_id] = max(scores[c.block_id], c.score)

        return self._sort_by_score(seen, scores)[:top_k * 3]

    def _vector_search(
        self,
        vec:          list[float],
        vector_name:  str,
        qdrant_filter,
        top_k:        int,
    ) -> list[RetrievedChunk]:
        result = self._qdrant.query_points(
            collection_name = self._collection,
            query           = vec,
            using           = vector_name,
            query_filter    = qdrant_filter,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._point_to_chunk(h, h.score) for h in result.points]

    # ------------------------------------------------------------------
    # Кэш эмбедингов
    # ------------------------------------------------------------------

    def _warm_cache(self, texts: list[str]) -> None:
        """Эмбедирует все тексты которых нет в кэше — одним батч-вызовом."""
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

    def _sort_by_score(
        self,
        chunks: dict[str, RetrievedChunk],
        scores: dict[str, float],
    ) -> list[RetrievedChunk]:
        ranked = sorted(chunks.values(), key=lambda c: scores[c.block_id], reverse=True)
        for c in ranked:
            c.score = scores[c.block_id]
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
        if not disciplines:
            return None
        if len(disciplines) == 1:
            return Filter(must=[FieldCondition(
                key="discipline", match=MatchValue(value=disciplines[0])
            )])
        return Filter(must=[FieldCondition(
            key="discipline", match=MatchAny(any=disciplines)
        )])

    def _load_discipline_names(self) -> list[str]:
        names: set[str] = set()
        offset = None
        while True:
            result = self._qdrant.scroll(
                collection_name = self._collection,
                limit           = 250,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            batch, next_offset = result[0], result[1]
            for point in batch:
                d = (point.payload or {}).get("discipline", "")
                if d:
                    names.add(d)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(names)
