from __future__ import annotations
import json
import re
import logging
from openai import OpenAI

from config import (
    LLM_BASE_URL, LLM_API_KEY,
    LLM_MODEL_MAIN, LLM_MAX_TOKENS_HYDE, LLM_MAX_TOKENS_MAIN,
    PARAPHRASES_COUNT,
)

from models import ExpandedQuery, QueryType, RouteResult
from prompts import DECOMPOSE_PROMPT, PARAPHRASE_PROMPT, HYDE_PROMPT

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# HyDE применяется только для одиночных запросов — там наибольший выигрыш
HYDE_QUERY_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}


def _parse_json(text: str) -> dict:
    text = text.strip()
    log.debug("Raw LLM content before parsing: %r", text[:1000])  

    # Более агрессивная очистка
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    
    # Ищем первый JSON между фигурных скобок (на случай мусора до/после)
    matches = re.findall(r'(\{.*?\})', text, re.DOTALL)
    if matches:
        text = matches[0]
    
    try:
        data = json.loads(text)
        if isinstance(data, str):   
            log.warning("LLM returned a string instead of dict: %r", data)
            # Попытка второго шанса
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        return data
    except Exception as e:
        log.error("JSON parsing failed. Content: %r", text[:1500])
        raise

class QueryExpander:
    """Расширяет запрос с помощью LLM: перефразировки, декомпозиция, HyDE."""
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def expand(
        self,
        query: str,
        route: RouteResult,
        resolved_disciplines: list[str],
    ) -> ExpandedQuery:
        
        if route.query_type == QueryType.MULTI_RELATION:
            # для запросов типа multi.relation сначала делаем декомпозицию, а потом перефразируем каждую часть
            sub_queries = self._decompose(query)
            
            # для каждого подзапроса делаем перефразировку в количестве 3х вариантов
            sub_queries_expanded = []
            for sub_q in sub_queries:
                paraphrases = self._paraphrase(sub_q)
                # результат сохраняем в виде списка словарей с оригиналом и перефразировками
                sub_queries_expanded.append({
                    "original": sub_q,
                    "paraphrases": paraphrases
                })
            
            paraphrases = []
            hyde_text = ""

            # логируем multi.relation c перефразированными подзапросами
            log.info(
                "Expander (MULTI_RELATION):\n"
                "  sub_queries (%d): %r\n"
                "  expanded sub-queries: %r",
                len(sub_queries),
                sub_queries,
                sub_queries_expanded
            )
            
        else:
            paraphrases = self._paraphrase(query)
            sub_queries = []
            sub_queries_expanded = []
            
            # HyDE для одиночных запросов, где он может дать наибольший прирост
            hyde_text = (
                self._hyde(query)
                if route.query_type in HYDE_QUERY_TYPES
                else ""
            )

            # логируем остальные типы
            log.info(
                "Expander:\n"
                "  paraphrases (%d): %r\n"
                "  sub_queries: (%d): %r\n"
                "  hyde_text (%d): %r\n",
                len(paraphrases),
                paraphrases,
                len(sub_queries),
                sub_queries,
                bool(hyde_text),
                hyde_text
                )

        return ExpandedQuery(
            original=query,
            paraphrases=paraphrases,
            sub_queries=sub_queries,
            disciplines=resolved_disciplines,
            query_type=route.query_type,
            sub_queries_expanded=sub_queries_expanded,  
        )
    


    def _call_json(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS_MAIN) -> dict:
        '''Вызов LLM с данным промптом, ожидая JSON в ответе. Возвращает распарсенный словарь.'''
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        log.debug("LLM response (raw): %r", resp)
        return _parse_json(resp.choices[0].message.content)

    def _call_text(self, prompt: str, max_tokens: int) -> str:
        '''Вызов LLM с данным промптом, возвращая текстовый ответ.'''
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        log.debug("LLM response (raw): %r", resp)
        return resp.choices[0].message.content.strip()



    def _paraphrase(self, query: str) -> list[str]:
        '''Функция для вызова LLM для перефразировки запроса. Возвращает список перефразировок.'''
        try:

            prompt = PARAPHRASE_PROMPT.format(query=query, num_paraphrases=PARAPHRASES_COUNT)
            result = self._call_json(prompt)
            log.debug("Parsed result type: %s, content: %r", type(result), result)

            if isinstance(result, str):
                log.error("LLM returned raw string instead of JSON dict!")
                return []

            paraphrases = result.get("paraphrases") or result.get("Paraphrases")
            if not isinstance(paraphrases, list):
                log.warning("No valid 'paraphrases' list in result: %r", result)
                return []

            # Фильтруем пустые
            paraphrases = [p.strip() for p in paraphrases if p and isinstance(p, str)]
            log.debug("Successfully got %d paraphrases", len(paraphrases))
            return paraphrases

        except Exception as exc:
            log.exception("Paraphrase failed")  
            return []

    def _decompose(self, query: str) -> list[str]:
        '''Функция для вызова LLM для декомпозиции multi.relation запроса. Возвращает список подзапросов.'''
        try:
            return self._call_json(
                DECOMPOSE_PROMPT.format(query=query)
            ).get("sub_queries") or []
        except Exception as exc:
            log.warning("Decompose failed: %s", exc)
            return []

    def _hyde(self, query: str) -> str:
        """Функция для вызова LLM для генерации HyDE-текста. Возвращает сгенерированный текст."""
        try:
            return self._call_text(
                HYDE_PROMPT.format(query=query),
                max_tokens=LLM_MAX_TOKENS_HYDE,
            )
        except Exception as exc:
            log.warning("HyDE failed: %s", exc)
            return ""
