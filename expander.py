"""
Модуль расширения запроса (Query Expander).

Стратегии:
  1. Перефразировки — для всех типов запросов.
  2. Декомпозиция — только для multi.relation.
  3. HyDE — генерация гипотетического фрагмента документа для улучшения
            эмбединга запроса (для single.simple и single.global).
"""
from __future__ import annotations

import json
import re
import logging

from openai import OpenAI

from config import (
    LLM_BASE_URL, LLM_API_KEY,
    LLM_MODEL_FAST, LLM_MAX_TOKENS_FAST, LLM_MAX_TOKENS_HYDE,
)
from models import ExpandedQuery, QueryType, RouteResult
from prompts import DECOMPOSE_PROMPT, PARAPHRASE_PROMPT, HYDE_PROMPT

log = logging.getLogger(__name__)

# HyDE применяется только для одиночных запросов — там наибольший выигрыш
HYDE_QUERY_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}



def _parse_json(text: str) -> dict:
    """Парсит JSON из ответа модели, убирая markdown-блоки если они есть."""
    text = text.strip()
    # Убираем ```json ... ``` или ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())

class QueryExpander:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def expand(
        self,
        query:                str,
        route:                RouteResult,
        resolved_disciplines: list[str],
    ) -> ExpandedQuery:
        paraphrases = self._paraphrase(query)
        sub_queries = (
            self._decompose(query)
            if route.query_type == QueryType.MULTI_RELATION
            else []
        )
        hyde_text = (
            self._hyde(query)
            if route.query_type in HYDE_QUERY_TYPES
            else ""
        )

        log.info(
            "Expander: %d перефразировок, %d подзапросов, HyDE=%s",
            len(paraphrases), len(sub_queries), bool(hyde_text),
        )
        return ExpandedQuery(
            original    = query,
            paraphrases = paraphrases,
            sub_queries = sub_queries,
            disciplines = resolved_disciplines,
            query_type  = route.query_type,
            hyde_text   = hyde_text,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_json(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS_FAST) -> dict:
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_FAST,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        return _parse_json(resp.choices[0].message.content)

    def _call_text(self, prompt: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_FAST,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def _paraphrase(self, query: str) -> list[str]:
        try:
            return self._call_json(
                PARAPHRASE_PROMPT.format(query=query)
            ).get("paraphrases") or []
        except Exception as exc:
            log.warning("Paraphrase failed: %s", exc)
            return []

    def _decompose(self, query: str) -> list[str]:
        try:
            return self._call_json(
                DECOMPOSE_PROMPT.format(query=query)
            ).get("sub_queries") or []
        except Exception as exc:
            log.warning("Decompose failed: %s", exc)
            return []

    def _hyde(self, query: str) -> str:
        """Генерирует гипотетический фрагмент РПД для улучшения эмбединга запроса."""
        try:
            return self._call_text(
                HYDE_PROMPT.format(query=query),
                max_tokens=LLM_MAX_TOKENS_HYDE,
            )
        except Exception as exc:
            log.warning("HyDE failed: %s", exc)
            return ""
