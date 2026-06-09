from __future__ import annotations
import json
import re, logging
from openai import OpenAI
from rag.config.config import (
    LLM_BASE_URL, LLM_API_KEY,
    LLM_MODEL_VERIFY, LLM_MAX_TOKENS_VERIFY,
)
from rag.domain.models import QueryType, VerificationResult
from rag.config.prompts import VERIFY_PROMPTS

log = logging.getLogger(__name__)

def _parse_json(text: str) -> dict:
    """Парсит JSON из ответа модели, убирая markdown-блоки если они есть."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from LLM")
    # Убираем ```json ... ``` или ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON after removing markdown")
    return json.loads(text)

class VerificationModule:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def verify(self, query: str, answer: str, context: str, query_type: QueryType) -> VerificationResult:
    
        if len(context) > 80_000:
            log.warning("=== Verification === context обрезан %d → 80000 симв.", len(context))
            context = context[:80_000]
            
        if query_type == QueryType.SINGLE_SIMPLE:
            template = VERIFY_PROMPTS.get("single.simple")
        elif query_type == QueryType.SINGLE_GLOBAL:
            template = VERIFY_PROMPTS.get("single.global")  
        elif query_type == QueryType.MULTI_RELATION:
            template = VERIFY_PROMPTS.get("multi.relation")
        elif query_type == QueryType.MULTI_COMPARE:
            template = VERIFY_PROMPTS.get("multi.compare")
        elif query_type in {QueryType.MULTI_GLOBAL_CATALOG, 
                            QueryType.MULTI_GLOBAL_SEMANTIC}:
            template = VERIFY_PROMPTS.get("multi.global")
        else:
            template = VERIFY_PROMPTS.get("single.global")  # safe fallback

        prompt = template.format(query=query, context=context, answer=answer)
        log.info("=== Verification === query_type: %s, query: %s", query_type, query)
        log.info("=== Verification === prompt length: %d", len(prompt))
        log.debug("=== Verification === prompt: %s", prompt)  
        log.debug("=== Verification === answer: %s", answer[:500])
        try:
            log.debug('=== Verification === ver_model: %s', LLM_MODEL_VERIFY)
            resp = self._client.chat.completions.create(
                model=LLM_MODEL_VERIFY,
                max_tokens=LLM_MAX_TOKENS_VERIFY,
                messages=[{"role": "user", "content": prompt}],
            )
            data = _parse_json(resp.choices[0].message.content)
            return VerificationResult(
                is_valid=bool(data.get("is_valid", True)),
                note=data.get("note", ""),
            )
        except Exception as exc:
            log.warning("=== Verification === parse error: %s", exc)
            return VerificationResult(is_valid=True, note=f"parse error: {exc}")
