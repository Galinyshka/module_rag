from __future__ import annotations
import json
import re
import logging
from openai import OpenAI
from rapidfuzz import fuzz, process
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_FAST, LLM_MAX_TOKENS_FAST, RPD_NAMES, FUZZY_THRESHOLD, FUZZY_TOP_K
from models import QueryType, RouteResult
from prompts import (
    PROMPT_CLASSIFY_MULTI,
    PROMPT_EXTRACT_QUERY_DISCIPLINE,
    PROMPT_EXTRACT_DISCIPLINES,
    PROMPT_CLASSIFY_SINGLE,
    PROMPT_CLASSIFY_ZERO,
    PROMPT_CLASSIFY_GLOBAL_SUBTYPE,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_GLOBAL_SUBTYPE_MAP = {
    "catalog":               QueryType.MULTI_GLOBAL_CATALOG,
    "semantic":       QueryType.MULTI_GLOBAL_SEMANTIC,
}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Парсит JSON из ответа модели, убирая markdown-блоки если они есть."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _llm_call(client: OpenAI, prompt: str) -> dict:
    """Универсальная обёртка для вызова LLM и парсинга JSON-ответа."""
    resp = client.chat.completions.create(
        model=LLM_MODEL_FAST,
        max_tokens=LLM_MAX_TOKENS_FAST,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json(resp.choices[0].message.content)


def _extract_query_names(client: OpenAI, query: str) -> list[str]:
    """ LLM вытаскивает сырые названия из запроса. """
    prompt = PROMPT_EXTRACT_QUERY_DISCIPLINE.format(query=query)
    try:
        data = _llm_call(client, prompt)
        names = data.get("names") or []
        return [n.lower().strip() for n in names if n.strip()]
    except Exception as exc:
        log.warning("extract_query_names failed: %s", exc)
        return []


def _fuzzy_candidates(extracted_names: list[str]) -> tuple[list[str], list[str]]:
    """Fuzzy поиск кандидатов по настоящим названиям дисциплин.
    
    Returns:
        tuple[list[str], list[str]]: (найденные_кандидаты, не_найденные_имена)
    """
    if not extracted_names:
        return [], []
    
    found_candidates = {}  # словарь для хранения лучших score
    not_found_names = []

    for name in extracted_names:
        results = process.extract(
            name,
            RPD_NAMES,
            scorer=fuzz.token_set_ratio,
            limit=FUZZY_TOP_K,
        )
        matches = [(match, score) for match, score, _ in results if score >= FUZZY_THRESHOLD]
        if matches:
            for match, score in matches:
                if match not in found_candidates or score > found_candidates[match]:
                    found_candidates[match] = score
        else:
            not_found_names.append(name)

    # Возвращаем отсортированный список кандидатов
    candidates = sorted(found_candidates.keys(), key=lambda x: found_candidates[x], reverse=True)
    
    return candidates, not_found_names


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    def __init__(self) -> None:
        self._client  = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self._rpd_names = set(RPD_NAMES)

    # ------------------------------------------------------------------
    # Шаг 1: fuzzy → LLM → дисциплины / уточнение / пусто
    # ------------------------------------------------------------------
    def _extract_disciplines(self, query: str) -> RouteResult | None:
        """
        Шаг 1: извлекаем дисциплины из запроса. 
        - нашли дисциплины (found)
        - нужно уточнение (clarify)
        Возвращает None если дисциплины не найдены — тогда route() идёт дальше.
        """
        # Шаг 1.1: LLM извлекает сырые имена из запроса
        extracted_names = _extract_query_names(self._client, query) # возвращет список названий строго из запроса, может быть пустым
        log.info("=== Router === Names from query: %s", extracted_names)

        if not extracted_names:
            # если LLM не извлёк ничего, возвращаем None, чтобы route() попытался классифицировать запрос как zero
            log.info("=== Router === No extracted names, returning None")
            return None

        # Шаг 1.2: fuzzy 
        candidates, not_found_names = _fuzzy_candidates(extracted_names)
        log.info("=== Router === Fuzzy candidates: %s", candidates)
        log.info("=== Router === Not found names: %s", not_found_names)

        if not_found_names:
            message = "Дисциплины " + ", ".join(not_found_names) + " не найдены."
            return RouteResult(
                query_type=QueryType.CLARIFY,
                disciplines=candidates,
                message=message,
            )
        
        if not candidates:
            log.info("=== Router === No fuzzy candidates found, returning None")
            return None

        candidates_text = "\n".join(f"- {c}" for c in candidates)
        prompt = PROMPT_EXTRACT_DISCIPLINES.format(
            candidates=candidates_text,
            query=query,
        )

        try:
            data = _llm_call(self._client, prompt)
        except Exception as exc:
            log.warning("extract_disciplines LLM failed: %s", exc)
            return None

        status = data.get("status")
        log.info("=== Router === LLM extraction status: %s", status) # варианты: "found", "clarify", "not_found"

        # Шаг 1.3: анализируем результат LLMа и возвращаем RouteResult
        if status == "found":
            # если LLM понял что термин из названия, но intent глобальный — не считаем found
            if data.get("intent") == "about_topic":
                log.info("=== Router === intent=about_topic despite fuzzy match, treating as zero")
                return None  # → уйдёт в classify_zero → MULTI_GLOBAL

            raw = data.get("disciplines") or []
            disciplines = [d for d in raw if d in self._rpd_names]
            if disciplines:
                return RouteResult(
                    query_type=QueryType.CLARIFY,
                    disciplines=disciplines,
                    message="found",
                )
    
        elif status == "clarify":
            # если LLM сомневается, просим уточнить. Вернуть RouteResult с query_type=CLARIFY и списком кандидатов для выбора.
            candidates = [d for d in (data.get("candidates") or []) if d in self._rpd_names]
            message   = "Уточните, пожалуйста: вы имеете в виду " + " или ".join(f"«{d}»" for d in candidates) + "?"
            if candidates:
                return RouteResult(
                    query_type=QueryType.CLARIFY,
                    disciplines=candidates,
                    message=message,
                )
        # status == "not_found" или что-то неожиданное
        return None


    def _classify_single(self, query: str) -> QueryType:
        """ Шаг 2а: классифицировать запрос с одной дисциплиной. """
        try:
            data = _llm_call(self._client, PROMPT_CLASSIFY_SINGLE.format(query=query))
            return QueryType(data["query_type"])
        except Exception as exc:
            log.warning("classify_single failed: %s", exc)
            return QueryType.SINGLE_SIMPLE


    def _classify_zero(self, query: str) -> QueryType:
        """ Шаг 2б: классифицировать запрос без дисциплин. """
        try:
            data = _llm_call(self._client, PROMPT_CLASSIFY_ZERO.format(query=query, disciplines="\n".join(f"- {d}" for d in RPD_NAMES)))
            return QueryType(data["query_type"])
        except Exception as exc:
            log.warning("classify_zero failed: %s", exc)
            return QueryType.IRRELEVANT

    def _classify_multi(self, query: str) -> QueryType:
        try:
            data = _llm_call(self._client, PROMPT_CLASSIFY_MULTI.format(query=query))
            return QueryType(data["query_type"])
        except Exception as exc:
            log.warning("classify_multi failed: %s", exc)
            return QueryType.MULTI_RELATION  # safe fallback

    def _classify_global_subtype(self, query: str) -> tuple[str, str]:
        """
        Определяет подтип MULTI_GLOBAL-запроса и извлекает ключевой термин.
 
        Returns:
            subtype: "catalog" | "semantic"
            entity:  извлечённый термин (код компетенции, название темы,
                     ключевое слово) или "" для catalog
        """
        try:
            data = _llm_call(self._client, PROMPT_CLASSIFY_GLOBAL_SUBTYPE.format(query=query))
            subtype = data.get("subtype", "semantic")
            entity = data.get("entity", "")
            query_type = _GLOBAL_SUBTYPE_MAP.get(subtype, QueryType.MULTI_GLOBAL_SEMANTIC)
            return query_type, entity
        except Exception as exc:
            log.warning("classify_global_subtype failed: %s", exc)
            return QueryType.MULTI_GLOBAL_SEMANTIC, ""

    def route(self, query: str) -> RouteResult:
        """ Главная функция маршрутизации."""
        try:
            extraction = self._extract_disciplines(query) # RouteResult с query_type=CLARIFY или None

            if extraction and extraction.query_type == QueryType.CLARIFY:
                if extraction.message == "found":
                    # LLM нашёл уверенно, фильтруем по количеству дисциплин 
                    disciplines = extraction.disciplines
                    n = len(disciplines)
                    if n > 1:
                        # если дисциплин несколько определяем multi.relation или multi.compare
                        query_type = self._classify_multi(query)
                    else:
                        # если дисциплина одна — нужно понять, single.simple или single.global
                        query_type = self._classify_single(query) # может вернуть query_type: single.simple или single.global
                    result = RouteResult(query_type=query_type, disciplines=disciplines)
                else:
                    # если message != "found", значит это clarify, возвращаем как есть
                    result = extraction
            else:
                # случай None из extract_disciplines — значит LLM не нашёл дисциплин, нужно классифицировать запрос как zero
                query_type = self._classify_zero(query) # может вернуть query_type: multi.global, not_found или irrelevant
                if query_type == QueryType.MULTI_GLOBAL:
                    # определяем подтип и ключевой термин
                    global_query_type, global_entity = self._classify_global_subtype(query)
                    log.info("=== Router === MULTI_GLOBAL subtype=%s entity=%r", global_query_type, global_entity)
                    result = RouteResult(
                        query_type=global_query_type,   # уточнённый подтип для multi.global
                        disciplines=[],
                        global_entity=global_entity,
                    )
                else:
                    result = RouteResult(query_type=query_type, disciplines=[])                

        except Exception as exc:
            log.warning("=== Router === Router fallback: %s", exc)
            return RouteResult(QueryType.SINGLE_SIMPLE, [], f"fallback: {exc}")

        log.info("=== Router Final: type=%s  disciplines=%s", result.query_type, result.disciplines)
        return result
# Дальнейшая обработка RouteResult происходит в pipeline.py, который принимает результат маршрутизации и решает, что делать дальше (например, если query_type=CLARIFY, то возвращает RAGResponse с кандидатами для уточнения).