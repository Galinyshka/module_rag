"""
Модуль реранкинга (Reranker).

Использует cross-encoder для точной оценки релевантности пары (запрос, текст).
Принципиально точнее косинусного сходства, т.к. оценивает запрос и документ
совместно, а не по отдельным эмбедингам.

Модель: mmarco-mMiniLMv2-L12-H384-v1 — мультиязычная, поддерживает русский.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANKER_TOP_K
from models import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self) -> None:
        log.info("Загрузка cross-encoder: %s ...", RERANKER_MODEL)
        self._model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Переранжирует чанки по релевантности запросу.
        Для оценки использует text блока — полный смысловой контекст.
        Возвращает топ RERANKER_TOP_K чанков.
        """
        if not chunks:
            return chunks

        pairs  = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key   = lambda x: x[1],
            reverse = True,
        )

        result = []
        for chunk, score in ranked[:RERANKER_TOP_K]:
            chunk.score = float(score)   # заменяем косинусный score на reranker score
            result.append(chunk)

        log.info("Reranker: %d -> %d чанков (top score=%.3f)",
                 len(chunks), len(result), result[0].score if result else 0)
        return result
