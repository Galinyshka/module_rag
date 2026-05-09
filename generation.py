from __future__ import annotations
import logging
from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_MAIN, LLM_MAX_TOKENS_MAIN
from models import ExpandedQuery, RetrievedChunk
from prompts import COMPARE_PROMPT, GENERATE_PROMPTS, SYNTHESIS_PROMPT

log = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 100000


def build_context(chunks: list[RetrievedChunk]) -> str:
    by_discipline: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        by_discipline.setdefault(c.discipline, []).append(c)

    n = len(by_discipline)
    per_disc_limit = MAX_CONTEXT_CHARS // n if n else MAX_CONTEXT_CHARS

    parts = []
    for discipline, disc_chunks in by_discipline.items():
        log.info("=== Generation === build_context '%s': %d чанков, блоки: %s",
                 discipline,
                 len(disc_chunks),
                 [c.block_name[:50] for c in disc_chunks])
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
            log.warning("=== Generation === build_context: '%s' обрезана до %d симв.", discipline, section_chars)

        parts.append("\n\n".join(section_parts))

    return "\n\n".join(parts)

class GenerationModule:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def generate(self, query: str, chunks: list[RetrievedChunk], expanded: ExpandedQuery) -> str:
        template = GENERATE_PROMPTS.get(expanded.query_type.value,
                                        GENERATE_PROMPTS["single.simple"])
        context = build_context(chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            log.warning("=== Generation === context обрезан с %d до %d симв.", len(context), MAX_CONTEXT_CHARS)
            context = context[:MAX_CONTEXT_CHARS]
        log.debug("=== Generation === Context: %s", context[:500].replace("\n", "\\n") + ("..." if len(context) > 500 else ""))
        log.info("=== Generation === тип=%s, контекст=%d симв.", expanded.query_type, len(context))
        return self._call(template.format(query=query, context=context)), context

    def generate_from_context(self, query: str, context: str, query_type_value: str) -> str:
        template = GENERATE_PROMPTS.get(query_type_value, GENERATE_PROMPTS["single.simple"])
        log.info("=== Generation === (fallback): контекст=%d симв.", len(context))
        return self._call(template.format(query=query, context=context)), context

    def generate_synthesis(self, query: str,
        sub_answers: list[tuple[str, str]],  # (sub_query, answer)
    ) -> str:
        
        parts = [
            f"Аспект: {sub_q}\nОтвет: {ans}"
            for sub_q, ans in sub_answers
        ]
        context  = "\n\n---\n\n".join(parts)
        combined = "\n\n---\n\n".join(parts)
        prompt = SYNTHESIS_PROMPT.format(query=query, sub_answers=combined)
        log.info("=== Generation === Synthesis: %d частичных ответов, промпт %d симв.", len(sub_answers), len(prompt))
        return self._call(prompt), context

    def generate_compare(self, query: str, chunks: list[RetrievedChunk],) -> str:
        context = build_context(chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
        prompt = COMPARE_PROMPT.format(query=query, context=context)
        log.info("=== Generation === Compare: %d дисциплин, контекст=%d симв.", 
                 len({c.discipline for c in chunks}), len(context))
        return self._call(prompt), context
    
    def _call(self, prompt: str) -> str:
        log.debug("=== Generation === Final prompt: %s", prompt[:1000].replace("\n", "\\n") + ("..." if len(prompt) > 500 else ""))
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = LLM_MAX_TOKENS_MAIN,
            messages   = [{"role": "user", "content": prompt}],
        )
        if not resp.choices:
            log.error("=== Generation === пустой ответ от LLM! Ответ: %s", resp)
            return "[Ошибка: пустой ответ от LLM]"
        answer = resp.choices[0].message.content.strip()
        log.info("=== Generation === ответ %d симв.", len(answer))
        return answer
