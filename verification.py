from __future__ import annotations
import json
import re, logging
from openai import OpenAI
from config import (
    LLM_BASE_URL, LLM_API_KEY,
    LLM_MODEL_VERIFY, LLM_MAX_TOKENS_VERIFY,
)
from models import RetrievedChunk, VerificationResult
from prompts import VERIFY_PROMPT

log = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 6_000



def _parse_json(text: str) -> dict:
    """Парсит JSON из ответа модели, убирая markdown-блоки если они есть."""
    text = text.strip()
    # Убираем ```json ... ``` или ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())

class VerificationModule:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def verify(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> VerificationResult:
        context = "\n\n".join(
            f"[{c.block_name}]\n{c.text}" for c in chunks
        )[:MAX_CONTEXT_CHARS]

        prompt = VERIFY_PROMPT.format(query=query, context=context, answer=answer)
        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL_VERIFY,
                max_tokens = LLM_MAX_TOKENS_VERIFY,
                messages   = [{"role": "user", "content": prompt}],
            )
            data   = _parse_json(resp.choices[0].message.content)
            result = VerificationResult(
                is_valid = bool(data.get("is_valid", True)),
                retry    = bool(data.get("retry", False)),
                note     = data.get("note", ""),
            )
        except Exception as exc:
            log.warning("Verification parse error: %s", exc)
            result = VerificationResult(is_valid=True, note=f"parse error: {exc}")

        log.info("Verification [%s]: valid=%s  retry=%s  note=%s",
                 LLM_MODEL_VERIFY, result.is_valid, result.retry, result.note)
        return result
