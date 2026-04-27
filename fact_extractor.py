"""
Модуль прямого извлечения фактов (FactExtractor).

Применяется для single.simple запросов вместо полной генерации.
Берёт один наиболее релевантный блок после реранкинга и извлекает
конкретное значение через дешёвый LLM_MODEL_FAST вызов.

Преимущества перед полной генерацией:
  - Один блок вместо полного контекста → меньше шансов на галлюцинацию.
  - Дешёвая модель → быстро и экономично.
  - Если факт не найден — прозрачно возвращает None и пайплайн
    переходит к обычной генерации.
"""
from __future__ import annotations

import logging

from openai import OpenAI

from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_FAST, LLM_MAX_TOKENS_FACT
from .models import QueryType, RetrievedChunk
from .prompts import FACT_EXTRACT_PROMPT

log = logging.getLogger(__name__)

# Только для этих типов запросов пробуем прямое извлечение
APPLICABLE_TYPES = {QueryType.SINGLE_SIMPLE}


class FactExtractor:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def try_extract(
        self,
        query:      str,
        chunks:     list[RetrievedChunk],
        query_type: QueryType,
    ) -> str | None:
        """
        Пытается извлечь факт напрямую из топ-чанка.
        Возвращает строку с фактом или None если не удалось.

        None означает: пайплайн должен идти в обычную генерацию.
        """
        if query_type not in APPLICABLE_TYPES:
            return None
        if not chunks:
            return None

        # Берём только самый релевантный блок
        top_chunk = chunks[0]

        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL_FAST,
                max_tokens = LLM_MAX_TOKENS_FACT,
                messages   = [{"role": "user", "content": FACT_EXTRACT_PROMPT.format(
                    query = query,
                    text  = top_chunk.text,
                )}],
            )
            result = resp.choices[0].message.content.strip()

            # Пустая строка = факт не найден в этом блоке
            if not result:
                log.info("FactExtractor: факт не найден в '%s'", top_chunk.block_name)
                return None

            log.info("FactExtractor: '%s' -> '%s'", top_chunk.block_name, result[:80])
            return result

        except Exception as exc:
            log.warning("FactExtractor error: %s", exc)
            return None
