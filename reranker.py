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

from config import RERANKER_MODEL, RERANKER_TOP_K, RERANKER_TOP_K_BALANCE
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

    def rerank_with_balance(self, query: str, chunks: list[RetrievedChunk],
                       disciplines: list[str] = None) -> list[RetrievedChunk]:
        """
        Для multi запросов: гарантирует, что в результате есть данные 
        из каждой дисциплины.
        Балансирует ДО применения TOP_K, чтобы каждая дисциплина была представлена.
        """
        if not chunks:
            return chunks
            
        if not disciplines or len(disciplines) == 1:
            log.debug("rerank_with_balance: fallback на обычный rerank (1 дисциплина)")
            return self.rerank(query, chunks)
        
        # СНАЧАЛА: считаем скоры для ВСЕ чанков (не урезаем!)
        pairs  = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        
        # Сортируем по скорам
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk
            chunk_copy.score = float(score)
            scored_chunks.append(chunk_copy)
        
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        
        # Считаем распределение ДО балансировки
        before = {}
        for c in scored_chunks[:RERANKER_TOP_K_BALANCE * 3]:  # Смотрим в расширенном топе
            before.setdefault(c.discipline, 0)
            before[c.discipline] += 1
        log.debug("Rerank before balance (top %d): %s", RERANKER_TOP_K_BALANCE * 3, before)
        
        # ПОТОМ: балансируем по дисциплинам
        # Гарантируем минимум 2 чанка на каждую дисциплину
        by_disc = {}
        for c in scored_chunks:
            by_disc.setdefault(c.discipline, []).append(c)
        
        # Берем минимум 4 из каждой дисциплины, остаток лучших
        min_per_disc = 4
        balanced = []
        
        # Шаг 1: минимум из каждой
        for disc in disciplines:
            disc_chunks = by_disc.get(disc, [])[:min_per_disc]
            balanced.extend(disc_chunks)
            log.debug("  %s (min): +%d чанков", disc, len(disc_chunks))
        
        # Шаг 2: добиваем лучшими оставшимися
        used_ids = {c.block_id for c in balanced}
        remaining = [c for c in scored_chunks if c.block_id not in used_ids]
        balanced.extend(remaining[:RERANKER_TOP_K_BALANCE - len(balanced)])
        
        # НАКОНЕЦ: сортируем по скорам
        balanced.sort(key=lambda x: x.score, reverse=True)
        result = balanced[:RERANKER_TOP_K_BALANCE]
        
        # Считаем распределение ПОСЛЕ балансировки
        after = {}
        for c in result:
            after.setdefault(c.discipline, 0)
            after[c.discipline] += 1
        log.info("Reranker after balance: %s (всего %d, min per disc=%d)",
                 after, len(result), min_per_disc)
        
        return result
