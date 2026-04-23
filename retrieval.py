"""
Гибридный модуль поиска (Retrieval).

Стратегии по типу запроса:
  single.*       — фильтр по дисциплине + семантический поиск
  multi.relation — поиск по каждой дисциплине отдельно → гарантированное покрытие
  multi.global   — двухэтапный:
                     1) широкое сканирование обзорных блоков → выявление дисциплин
                     2) точечный поиск по выявленным дисциплинам

Дополнительно: get_full_document(discipline) для fallback-сценария.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from .models import ExpandedQuery, QueryType, RetrievedChunk

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION      = "discipline_chunks"
QDRANT_URL      = "http://localhost:6333"

TOP_K_SINGLE    = 6
TOP_K_STAGE1    = 30
TOP_K_PER_DISC  = 8
MATCH_THRESHOLD = 0.55
MAX_DISCIPLINES_MULTI = 10

OVERVIEW_BLOCKS = ["course_info", "topics", "competencies"]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class RetrievalModule:
    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = COLLECTION,
        embed_model: str = EMBEDDING_MODEL,
    ) -> None:
        log.info("Загрузка модели эмбедингов: %s ...", embed_model)
        self._embedder   = SentenceTransformer(embed_model)
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
            return self._retrieve_multi_global(expanded)
        if qt == QueryType.MULTI_RELATION:
            return self._retrieve_multi_relation(expanded)
        return self._retrieve_single(expanded)

    def get_full_document(self, discipline: str) -> list[RetrievedChunk]:
        """
        Возвращает ВСЕ блоки дисциплины, отсортированные по типу.
        Используется как fallback после провала верификации для single-запросов.
        """
        ORDER = ["course_info", "topics", "competencies",
                 "topic", "competency", "self_study_resources",
                 "assessment_fund", "literature", "online_resources",
                 "other_sections", "other_section"]
        disc_filter = Filter(must=[FieldCondition(
            key="discipline", match=MatchValue(value=discipline)
        )])
        points, _ = self._qdrant.scroll(
            collection_name = self._collection,
            scroll_filter   = disc_filter,
            limit           = 500,
            with_payload    = True,
            with_vectors    = False,
        )
        chunks = [self._point_to_chunk(p, score=1.0) for p in points]
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
                log.info("Дисциплина '%s' -> '%s' (%.2f)", name, best_match, best_score)
            else:
                log.warning("Дисциплина '%s' не распознана (max=%.2f)", name, best_score)
        return list(dict.fromkeys(resolved))

    # ------------------------------------------------------------------
    # Стратегии
    # ------------------------------------------------------------------

    def _retrieve_single(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        disc_filter = self._discipline_filter(expanded.disciplines)
        queries     = [expanded.original, *expanded.paraphrases]
        return self._multi_query_search(queries, disc_filter, TOP_K_SINGLE)

    def _retrieve_multi_relation(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        """
        Для каждой дисциплины — независимый поиск.
        Гарантирует, что каждая дисциплина представлена в итоговом контексте.
        """
        disciplines = expanded.disciplines or self._discipline_index
        queries     = [expanded.original, *expanded.paraphrases, *expanded.sub_queries]
        all_chunks: dict[str, RetrievedChunk] = {}
        scores:     dict[str, float]          = {}

        for disc in disciplines:
            disc_filter = self._discipline_filter([disc])
            for c in self._multi_query_search(queries, disc_filter, TOP_K_PER_DISC):
                if c.block_id not in all_chunks:
                    all_chunks[c.block_id] = c
                    scores[c.block_id]     = c.score
                else:
                    scores[c.block_id] = max(scores[c.block_id], c.score)

        ranked = sorted(all_chunks.values(),
                        key=lambda c: scores[c.block_id], reverse=True)
        for c in ranked:
            c.score = scores[c.block_id]
        log.info("multi.relation: %d чанков из %d дисциплин",
                 len(ranked), len(disciplines))
        return ranked

    def _retrieve_multi_global(self, expanded: ExpandedQuery) -> list[RetrievedChunk]:
        """
        Stage 1 - сканирование обзорных блоков по всему корпусу.
        Stage 2 - точечный поиск по выявленным релевантным дисциплинам.
        """
        queries = [expanded.original, *expanded.paraphrases]

        # Stage 1
        overview_filter = Filter(must=[FieldCondition(
            key="block_type", match=MatchAny(any=OVERVIEW_BLOCKS)
        )])
        stage1_chunks = self._multi_query_search(queries, overview_filter, TOP_K_STAGE1)

        relevant = list(dict.fromkeys(
            c.discipline for c in stage1_chunks if c.discipline
        ))[:MAX_DISCIPLINES_MULTI]
        log.info("multi.global Stage 1: %d релевантных дисциплин", len(relevant))

        if not relevant:
            return stage1_chunks

        # Stage 2
        disc_filter   = self._discipline_filter(relevant)
        stage2_chunks = self._multi_query_search(queries, disc_filter, TOP_K_PER_DISC)

        merged: dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}
        for c in [*stage1_chunks, *stage2_chunks]:
            if c.block_id not in merged:
                merged[c.block_id] = c
                scores[c.block_id] = c.score
            else:
                scores[c.block_id] = max(scores[c.block_id], c.score)

        ranked = sorted(merged.values(),
                        key=lambda c: scores[c.block_id], reverse=True)
        for c in ranked:
            c.score = scores[c.block_id]
        log.info("multi.global Stage 2: итого %d чанков", len(ranked))
        return ranked

    # ------------------------------------------------------------------
    # Внутренние
    # ------------------------------------------------------------------

    def _multi_query_search(self, queries, qdrant_filter, top_k):
        seen:   dict[str, RetrievedChunk] = {}
        scores: dict[str, float]          = {}
        for q in queries:
            for vec_name in ("text", "summary"):
                for c in self._vector_search(q, vec_name, qdrant_filter, top_k):
                    if c.block_id not in seen:
                        seen[c.block_id]   = c
                        scores[c.block_id] = c.score
                    else:
                        scores[c.block_id] = max(scores[c.block_id], c.score)
        ranked = sorted(seen.values(), key=lambda c: scores[c.block_id], reverse=True)
        for c in ranked:
            c.score = scores[c.block_id]
        return ranked[:top_k * 3]

    def _vector_search(self, query, vector_name, qdrant_filter, top_k):
        vec  = self._embed(query)
        hits = self._qdrant.search(
            collection_name = self._collection,
            query_vector    = (vector_name, vec),
            query_filter    = qdrant_filter,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._point_to_chunk(h, h.score) for h in hits]

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

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(
            [text], normalize_embeddings=True
        )[0].tolist()

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
            result, next_offset = self._qdrant.scroll(
                collection_name = self._collection,
                scroll_filter   = None,
                limit           = 250,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for point in result:
                d = (point.payload or {}).get("discipline", "")
                if d:
                    names.add(d)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(names)
