from __future__ import annotations
import logging
from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_MAIN, LLM_MAX_TOKENS_MAIN
from models import ExpandedQuery, RetrievedChunk
from prompts import GENERATE_PROMPTS

log = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 50_000


def build_context(chunks: list[RetrievedChunk]) -> str:
    by_discipline: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        by_discipline.setdefault(c.discipline, []).append(c)

    n = len(by_discipline)
    per_disc_limit = MAX_CONTEXT_CHARS // n if n else MAX_CONTEXT_CHARS

    parts = []
    for discipline, disc_chunks in by_discipline.items():
        log.info("build_context '%s': %d чанков, блоки: %s",
                 discipline,
                 len(disc_chunks),
                 [c.block_name[:30] for c in disc_chunks])
        section_parts = [f"=== Дисциплина: {discipline} ==="]
        section_chars = 0
        truncated = False

        for c in disc_chunks:
            block = f"[{c.block_name}]\n{c.text}"
            if section_chars + len(block) > per_disc_limit:
                truncated = True
                break
            section_parts.append(block)
            section_chars += len(block)

        if truncated:
            section_parts.append("[...данные обрезаны из-за лимита контекста...]")
            log.warning("build_context: '%s' обрезана до %d симв.", discipline, section_chars)

        parts.append("\n\n".join(section_parts))

    return "\n\n".join(parts)

class GenerationModule:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def generate(self, query: str, chunks: list[RetrievedChunk], expanded: ExpandedQuery) -> str:
        template = GENERATE_PROMPTS.get(expanded.query_type.value,
                                        GENERATE_PROMPTS["single.simple"])
        context  = build_context(chunks)
        log.info("Generation: тип=%s, контекст=%d симв.", expanded.query_type, len(context))
        return self._call(template.format(query=query, context=context))

    def generate_from_context(self, query: str, context: str, query_type_value: str) -> str:
        template = GENERATE_PROMPTS.get(query_type_value, GENERATE_PROMPTS["single.simple"])
        log.info("Generation (fallback): контекст=%d симв.", len(context))
        return self._call(template.format(query=query, context=context[:MAX_CONTEXT_CHARS]))

    def _call(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = LLM_MAX_TOKENS_MAIN,
            messages   = [{"role": "user", "content": prompt}],
        )
        answer = resp.choices[0].message.content.strip()
        log.info("Generation: ответ %d симв.", len(answer))
        return answer
