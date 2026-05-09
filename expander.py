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
from prompts import DECOMPOSE_EXPAND_PROMPT, PARAPHRASE_PROMPT, HYDE_PROMPT

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# HyDE применяется только для одиночных запросов — там наибольший выигрыш
HYDE_QUERY_TYPES = {QueryType.SINGLE_SIMPLE, QueryType.SINGLE_GLOBAL}


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Находим позицию первой { и последней } — это внешний объект
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        return data
    except Exception as exc:
        log.error("=== Expander === JSON parsing failed. Content: %r. Error: %s", text[:2000], exc)
        raise

class QueryExpander:
    """Расширяет запрос с помощью LLM: перефразировки, декомпозиция, HyDE."""
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def expand(self, query: str, route: RouteResult, resolved_disciplines: list[str], expanded_flag = True) -> ExpandedQuery:
        
        sub_queries = []
        sub_expanded = []
        paraphrases = []
        hyde_text = ""  

        if route.query_type == QueryType.MULTI_COMPARE or not expanded_flag:
            if route.query_type == QueryType.MULTI_COMPARE:
                log.info("=== Expander === (%s): - возвращаем без расширения", route.query_type)
            else:
                log.info("=== Expander === (%s): expanded_flag=False (первый проход) - возвращаем без расширения", route.query_type)
            return ExpandedQuery(
                original     = query,
                paraphrases  = [],
                sub_queries  = [],
                disciplines  = resolved_disciplines,
                query_type   = route.query_type,
                hyde_text    = "",
                sub_expanded = [],
            )
        
        if route.query_type == QueryType.MULTI_RELATION:
            # для запросов типа multi.relation сначала делаем декомпозицию и перефразируем каждую часть
            raw_sub_queries = self._decompose_and_expand(query, resolved_disciplines, PARAPHRASES_COUNT)
            
            # для каждого подзапроса создаём ExpandedQuery, используя discipline из декомпозиции и определяем тип как SINGLE_GLOBAL
            sub_expanded = [
                ExpandedQuery(
                    original    = sq["original"],
                    paraphrases = [],
                    sub_queries = [],
                    disciplines = [sq["discipline"]],
                    query_type  = QueryType.SINGLE_GLOBAL,
                    hyde_text   = "",
                    sub_expanded= [],
                )
                for sq in raw_sub_queries
                if sq.get("original") and sq.get("discipline")
            ]

            # логируем multi.relation c перефразированными подзапросами
            log.info(
                "=== Expander === (%s): %d sub_expanded: %r",
                route.query_type,
                len(sub_expanded),
                [(e.original, e.disciplines) for e in sub_expanded],
            )

            return ExpandedQuery(
                original      = query,
                paraphrases   = [],
                sub_queries   = [e.original for e in sub_expanded],  # для логов/отладки
                disciplines   = resolved_disciplines,
                query_type    = route.query_type,
                hyde_text     = "",
                sub_expanded  = sub_expanded,
            )
            
        else:
            paraphrases = self._paraphrase(query)
            sub_queries = []
            sub_expanded = []
            
            # HyDE для одиночных запросов, где он может дать наибольший прирост
            hyde_text = (
                self._hyde(query)
                if route.query_type in HYDE_QUERY_TYPES
                else ""
            )

            # логируем остальные типы
            log.info(
                "=== Expander === (%s):\n"
                "  paraphrases (%d): %r\n"
                "  sub_queries: (%d): %r\n"
                "  hyde_text (%d): %r\n",
                route.query_type,
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
            sub_expanded=sub_expanded,
            hyde_text=hyde_text,  
        )
    


    def _call_json(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS_MAIN) -> dict:
        '''Вызов LLM с данным промптом, ожидая JSON в ответе. Возвращает распарсенный словарь.'''
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = 5000,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw_content = resp.choices[0].message.content
        log.debug("=== Expander (call_json) === LLM raw content: %r", raw_content[:1000])
        return _parse_json(raw_content)     

    def _call_text(self, prompt: str, max_tokens: int) -> str:
        '''Вызов LLM с данным промптом, возвращая текстовый ответ.'''
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_MAIN,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        log.debug("=== Expander (call_text) === LLM response (raw): %r", resp)
        return resp.choices[0].message.content.strip()



    def _paraphrase(self, query: str) -> list[str]:
        '''Функция для вызова LLM для перефразировки запроса. Возвращает список перефразировок.'''
        try:

            prompt = PARAPHRASE_PROMPT.format(query=query, num_paraphrases=PARAPHRASES_COUNT)
            result = self._call_json(prompt)
            log.debug("=== Expander (paraphrase) === Parsed result type: %s, content: %r", type(result), result)

            if isinstance(result, str):
                log.error("=== Expander (paraphrase) === LLM returned raw string instead of JSON dict!")
                return []

            paraphrases = result.get("paraphrases") or result.get("Paraphrases")
            if not isinstance(paraphrases, list):
                log.warning("=== Expander (paraphrase) === No valid 'paraphrases' list in result: %r", result)
                return []

            # Фильтруем пустые
            paraphrases = [p.strip() for p in paraphrases if p and isinstance(p, str)]
            log.debug("=== Expander (paraphrase) === Successfully got %d paraphrases", len(paraphrases))
            return paraphrases

        except Exception as exc:
            log.exception("=== Expander (paraphrase) === Paraphrase failed")
            return []

    def _decompose_and_expand(self, query: str, disciplines: list[str], num_paraphrases: int) -> list[dict]:
        try:
            prompt = DECOMPOSE_EXPAND_PROMPT.format(
                query=query,
                disciplines=", ".join(f'«{d}»' for d in disciplines),
                num_paraphrases=num_paraphrases,
            )
            # логируем сырой ответ ДО _call_json
            resp = self._client.chat.completions.create(
                model=LLM_MODEL_MAIN,
                max_tokens=5000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content
            log.debug("=== Expander (decompose_and_expand) === decompose_and_expand raw response: %r", raw)  
            
            result = _parse_json(raw)
            log.debug("=== Expander (decompose_and_expand) === decompose_and_expand parsed: %r", result)   
            
            return result.get("sub_queries") or []
        except Exception as exc:
            log.exception("=== Expander (decompose_and_expand) === decompose_and_expand failed") 
            return []

    def _hyde(self, query: str) -> str:
        """Функция для вызова LLM для генерации HyDE-текста. Возвращает сгенерированный текст."""
        try:
            return self._call_text(
                HYDE_PROMPT.format(query=query),
                max_tokens=LLM_MAX_TOKENS_HYDE,
            )
        except Exception as exc:
            log.warning("=== Expander (hyde) === HyDE failed: %s", exc)
            return ""
