from __future__ import annotations
import logging, re
from openai import OpenAI
from .config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_FAST, LLM_MAX_TOKENS_TIME

log = logging.getLogger(__name__)

_TIME_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bчас(ов|а|ы)?\b",
    r"\bзачётн\w+\s+единиц\w*",
    r"\bзе\b",
    r"\bсеместр\w*\b",
    r"\bлекци\w+\s+час\w*",
    r"\bпрактик\w+\s+час\w*",
    r"\bаудиторн\w+\s+час\w*",
    r"\bсамостоятельн\w+\s+час\w*",
    r"\bтрудо[её]мк\w*\b",
    r"\bаттестаци\w*\b",
    r"\bэкзамен\w*\b",
    r"\bзачёт\w*|\bзачет\w*",
    r"\bформ\w+\s+(контрол\w+|аттестаци\w+|промежуточн\w+)",
    r"сколько\s+(часов|зе|единиц|лекций|семинаров)",
]]

EXTRACT_PROMPT = """\
Из текста ниже извлеки ТОЛЬКО конкретный ответ на вопрос: числовое значение, \
название формы контроля или иной краткий факт.
Без вступлений и пояснений — только факт (1–2 строки максимум).

Вопрос: {query}

Текст ответа:
{answer}

Факт:"""


def is_time_query(query: str) -> bool:
    return any(p.search(query) for p in _TIME_PATTERNS)


class TimeFilter:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def apply(self, query: str, answer: str) -> str:
        if not is_time_query(query):
            return answer
        log.info("TimeFilter: числовой запрос, извлекаем факт...")
        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL_FAST,
                max_tokens = LLM_MAX_TOKENS_TIME,
                messages   = [{"role": "user",
                               "content": EXTRACT_PROMPT.format(
                                   query=query, answer=answer
                               )}],
            )
            fact = resp.choices[0].message.content.strip()
            log.info("TimeFilter: '%s...' -> '%s'", answer[:50], fact)
            return fact
        except Exception as exc:
            log.warning("TimeFilter error: %s", exc)
            return answer
