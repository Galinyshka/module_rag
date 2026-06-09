from __future__ import annotations
import logging
from sentence_transformers import CrossEncoder
from rag.config.config import RERANKER_MODEL, RERANKER_TOP_K
from rag.domain.models import RetrievedChunk

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
        scores = self._model.predict(pairs, show_progress_bar=False) # тут полоска Batches
        log.debug("=== Reranker (single query) === raw scores: %s", " ".join(f"{s:.2f}" for s in scores))
        scores = [float(s)*0.9 + float(c.score) * 0.1 for s, c in zip(scores, chunks)]  # комбинируем с косинусным score для стабильности (можно регулировать веса)
        log.debug("=== Reranker (single query) === combined scores: %s", " ".join(f"{s:.2f}" for s in scores))

        ranked = sorted(
            zip(chunks, scores),
            key   = lambda x: x[1],
            reverse = True,
        )

        reranked_chunks = []
        for chunk, score in ranked[:RERANKER_TOP_K]:
            if score > 0.01:
                chunk.score = float(score)   # заменяем косинусный score на reranker score
                reranked_chunks.append(chunk)

        log.info("=== Reranker === %d -> %d чанков (top score=%.3f)",
                 len(chunks), len(reranked_chunks), reranked_chunks[0].score if reranked_chunks else 0)
        log.info("=== Reranker === Chunks after rerank:\n%s",
         "\n\n".join(
             f"  [{i+1}] {c.score:+.2f} | {c.discipline} | {c.block_name}\n  {c.text[:100]}"
             for i, c in enumerate(reranked_chunks)
         ))
        
        if reranked_chunks:
            sc = [c.score for c in reranked_chunks]
            log.info("=== Reranker === scores: min=%.3f max=%.3f avg=%.3f | distribution: %s",
                     min(sc), max(sc), sum(sc)/len(sc),
                     " ".join(f"{s:.2f}" for s in sc))


        return reranked_chunks
'''
    def rerank_per_discipline(self, query: str, chunks: list[RetrievedChunk],
                            disciplines: list[str]) -> list[RetrievedChunk]:
        """
        Для MULTI_RELATION: реранкинг отдельно по каждой дисциплине,
        затем объединение лучших чанков каждой.
        """
        if not chunks:
            return chunks

        by_disc = {}
        for c in chunks:
            by_disc.setdefault(c.discipline, []).append(c)

        result = []
        for disc in disciplines:
            disc_chunks = by_disc.get(disc, [])
            if not disc_chunks:
                log.warning("rerank_per_discipline: нет чанков для '%s'", disc)
                continue

            pairs = [(query, c.text) for c in disc_chunks]
            scores = self._model.predict(pairs)

            ranked = sorted(zip(disc_chunks, scores), key=lambda x: x[1], reverse=True)

            top = []
            for chunk, score in ranked[:RERANKER_TOP_K_BALANCE]:
                if score < 0:
                    break
                chunk.score = float(score)
                top.append(chunk)
                
            # fallback: если всё отрицательное — берём 2 лучших с пометкой
            if not top:
                log.warning("  %s: все скоры отрицательные, берём 2 лучших как fallback", disc)
                for chunk, score in ranked[:2]:
                    chunk.score = float(score)
                    top.append(chunk)

            log.info("Chunks after rerank:\n%s",
            "\n\n".join(
                f"  [{i+1}] {c.score:+.2f} | {c.discipline} | {c.block_name}\n  {c.text[:200]}"
                for i, c in enumerate(result)
            ))

            sc = [c.score for c in top]
            log.info("  %s: %d -> %d чанков | scores: %s",
                    disc, len(disc_chunks), len(top),
                    " ".join(f"{s:.2f}" for s in sc))

            result.extend(top)

        return result'''
    
'''    def rerank_single_query(self, query: str, chunks: list[RetrievedChunk],
                            top_k: int = 3) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        log.debug("Reranker single query: raw scores: %s", " ".join(f"{s:.2f}" for s in scores))
        scores = [float(s)*0.8 + float(c.score) * 0.2 for s, c in zip(scores, chunks)]  # комбинируем с косинусным score для стабильности (можно регулировать веса)
        log.debug("Reranker single query: combined scores: %s", " ".join(f"{s:.2f}" for s in scores))
        score_map = {c.block_id: float(s) for c, s in zip(chunks, scores)}

        # топ по cross-encoder
        ranked = sorted(chunks, key=lambda c: score_map[c.block_id], reverse=True)
        result = []
        for chunk in ranked[:top_k]:
            chunk.score = score_map[chunk.block_id]
            result.append(chunk)

        # гарантируем топ-1 по RRF если его нет в результатах
        rrf_top = chunks[0]
        if rrf_top.block_id not in {c.block_id for c in result}:
            rrf_top.score = score_map[rrf_top.block_id]
            result.append(rrf_top)
            log.debug("  RRF-top добавлен принудительно: %s (ce_score=%.2f)",
                    rrf_top.block_name[:50], rrf_top.score)

        return result'''
    

'''
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
'''