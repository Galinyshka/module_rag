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
MAX_CONTEXT_CHARS = 40000



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
        # группируем по дисциплинам с равным лимитом
        by_disc: dict[str, list[RetrievedChunk]] = {}
        for c in chunks:
            by_disc.setdefault(c.discipline, []).append(c)

        n = len(by_disc) or 1
        per_disc_limit = 8_000 // n

        parts = []
        for disc, disc_chunks in by_disc.items():
            section = f"=== {disc} ===\n"
            section += "\n\n".join(f"[{c.block_name}]\n{c.text}" for c in disc_chunks)
            parts.append(section[:per_disc_limit])

        context = "\n\n".join(parts)
   

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
