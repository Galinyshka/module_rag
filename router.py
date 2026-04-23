from __future__ import annotations
import json, logging
from openai import OpenAI, RateLimitError
from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_FAST, LLM_MAX_TOKENS_FAST
from .models import QueryType, RouteResult
from .prompts import ROUTER_PROMPT

log = logging.getLogger(__name__)


class Router:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def route(self, query: str) -> RouteResult:
        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL_FAST,
                max_tokens = LLM_MAX_TOKENS_FAST,
                messages   = [{"role": "user",
                               "content": ROUTER_PROMPT.format(query=query)}],
            )
            data = json.loads(resp.choices[0].message.content.strip())
        except Exception as exc:
            log.warning("Router fallback: %s", exc)
            return RouteResult(QueryType.SINGLE_SIMPLE, [], f"fallback: {exc}")

        try:
            qt = QueryType(data["query_type"])
        except ValueError:
            qt = QueryType.SINGLE_SIMPLE

        result = RouteResult(
            query_type  = qt,
            disciplines = data.get("disciplines") or [],
            reasoning   = data.get("reasoning", ""),
        )
        log.info("Router: type=%s  disciplines=%s", result.query_type, result.disciplines)
        return result
