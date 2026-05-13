from __future__ import annotations
import logging
from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_MAIN, LLM_MAX_TOKENS_MAIN
from models import ExpandedQuery, RetrievedChunk
from prompts import COMPARE_PROMPT, GENERATE_PROMPTS, SYNTHESIS_PROMPT

log = logging.getLogger(__name__)
MAX_CONTEXT_CHARS = 100000


def build_context(chunks: list[RetrievedChunk]) -> str:
    '''Строит контекст для генерации, группируя чанки по дисциплинам и обрезая при необходимости.'''

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

    def generate(self, query: str, chunks: list[RetrievedChunk], query_type_value: str,
                 ) -> tuple[str, str]:
        '''Генерация с использовнием контекста из чанков'''

        context = build_context(chunks)

        template = GENERATE_PROMPTS.get(
            query_type_value,
            GENERATE_PROMPTS["single.global"]  # safe fallback
        )

        prompt = template.format(query=query,context=context)

        return self._generate_with_prompt(
            prompt=prompt,
            context=context,
            label=f"generate:{query_type_value}",
        )

    def generate_from_context(self, query: str, context: str, query_type_value: str,
                              ) -> tuple[str, str]:
        '''Генерация сразу из готового контекста'''

        template = GENERATE_PROMPTS.get(
            query_type_value,
            GENERATE_PROMPTS["single.global"]  # safe fallback
        )

        prompt = template.format(query=query,context=context)

        return self._generate_with_prompt(
            prompt=prompt,
            context=context,
            label=f"context:{query_type_value}",
        )

    def generate_synthesis(self, query: str, sub_answers: list[tuple[str, str]],
                           ) -> tuple[str, str]:
        '''Генерация синтеза из нескольких подответов (multi.relation)'''

        parts = [
            f"Аспект: {sub_q}\nОтвет: {ans}"
            for sub_q, ans in sub_answers
        ]

        context = "\n\n---\n\n".join(parts)

        prompt = SYNTHESIS_PROMPT.format(query=query, sub_answers=context)

        return self._generate_with_prompt(
            prompt=prompt,
            context=context,
            label="synthesis",
        )

    def _generate_with_prompt(self, *, prompt: str, context: str, label: str,
                              ) -> tuple[str, str]:
        '''Вызывает LLM с данным промптом и контекстом, логирует длины и обрезает контекст при необходимости'''

        original_len = len(context)

        if original_len > MAX_CONTEXT_CHARS:
            log.warning(
                "=== Generation === (%s) context truncated: %d -> %d",
                label,
                original_len,
                MAX_CONTEXT_CHARS,
            )

            context = context[:MAX_CONTEXT_CHARS]

        log.info(
            "=== Generation === (%s) context=%d chars, prompt=%d chars",
            label,
            len(context),
            len(prompt),
        )

        answer = self._call(prompt)

        return answer, context
    
    def _call(self, prompt: str) -> str:
        log.debug("=== Generation === Final prompt: %s", prompt[:1000].replace("\n", "\\n") + ("..." if len(prompt) > 500 else ""))
        log.debug("=== Generation === gen_model: %s", LLM_MODEL_MAIN)
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
        log.info("=== Generation === ответ (первые 1000 симв.): %s", answer[:1000].replace("\n", "\\n") + ("..." if len(answer) > 500 else ""))
        return answer
