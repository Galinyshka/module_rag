from __future__ import annotations
import json
import re, logging
from openai import OpenAI
from config import (
    LLM_BASE_URL, LLM_API_KEY,
    LLM_MODEL_VERIFY, LLM_MAX_TOKENS_VERIFY,
)
from models import QueryType, RetrievedChunk, VerificationResult
from prompts import VERIFY_COMPARE_PROMPT, VERIFY_PROMPT

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
            
        template = (
            VERIFY_COMPARE_PROMPT
            if query_type == QueryType.MULTI_COMPARE or query_type == QueryType.MULTI_RELATION
            else VERIFY_PROMPT
        )
        prompt = template.format(query=query, context=context, answer=answer)
        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL_VERIFY,
                max_tokens = LLM_MAX_TOKENS_VERIFY,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw_content = resp.choices[0].message.content
            log.debug("=== Verification === raw LLM response: %r", raw_content)
            
            data   = _parse_json(raw_content)
            result = VerificationResult(
                is_valid = bool(data.get("is_valid", True)),
                retry    = bool(data.get("retry", False)),
                note     = data.get("note", ""),
            )
        except Exception as exc:
            log.warning("=== Verification === parse error: %s", exc)
            result = VerificationResult(is_valid=True, note=f"parse error: {exc}")

        return result
