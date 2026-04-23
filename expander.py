"""
Модуль расширения запроса (Query Expander).

Стратегии:
  1. Перефразировки — для всех типов запросов.
  2. Декомпозиция — только для multi.relation (разбивка на подзапросы).
"""
from __future__ import annotations

import json
import logging

import anthropic

from .models import ExpandedQuery, QueryType, RouteResult
from .prompts import DECOMPOSE_PROMPT, PARAPHRASE_PROMPT

log = logging.getLogger(__name__)

LLM_MODEL      = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 400


class QueryExpander:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def expand(self, query: str, route: RouteResult, resolved_disciplines: list[str]) -> ExpandedQuery:
        paraphrases = self._paraphrase(query)

        sub_queries: list[str] = []
        if route.query_type == QueryType.MULTI_RELATION:
            sub_queries = self._decompose(query)

        log.info(
            "Expander: %d перефразировок, %d подзапросов",
            len(paraphrases), len(sub_queries),
        )
        return ExpandedQuery(
            original     = query,
            paraphrases  = paraphrases,
            sub_queries  = sub_queries,
            disciplines  = resolved_disciplines,
            query_type   = route.query_type,
            is_time_query = route.is_time_query,
        )

    def _paraphrase(self, query: str) -> list[str]:
        try:
            msg = self._client.messages.create(
                model      = LLM_MODEL,
                max_tokens = LLM_MAX_TOKENS,
                messages   = [{"role": "user",
                               "content": PARAPHRASE_PROMPT.format(query=query)}],
            )
            data = json.loads(msg.content[0].text.strip())
            return data.get("paraphrases") or []
        except Exception as exc:
            log.warning("Paraphrase failed: %s", exc)
            return []

    def _decompose(self, query: str) -> list[str]:
        try:
            msg = self._client.messages.create(
                model      = LLM_MODEL,
                max_tokens = LLM_MAX_TOKENS,
                messages   = [{"role": "user",
                               "content": DECOMPOSE_PROMPT.format(query=query)}],
            )
            data = json.loads(msg.content[0].text.strip())
            return data.get("sub_queries") or []
        except Exception as exc:
            log.warning("Decompose failed: %s", exc)
            return []
