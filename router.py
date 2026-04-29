from __future__ import annotations
import json
import re
import logging
from openai import OpenAI
from rapidfuzz import fuzz, process
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_FAST, LLM_MAX_TOKENS_FAST, RPD_NAMES
from models import QueryType, RouteResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Настройки fuzzy-поиска
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD = 60   # минимальный score для попадания в кандидаты
_FUZZY_TOP_K     = 5    # максимум кандидатов, передаваемых в LLM

# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

_PROMPT_EXTRACT_DISCIPLINES = """
Ты — ассистент, который определяет, какие учебные дисциплины имеет в виду пользователь.

Тебе дан запрос пользователя и список дисциплин-кандидатов (отобранных заранее по схожести).

Твои возможные решения:

1. Если можешь уверенно определить дисциплину(ы) — верни:
{{"status": "found", "disciplines": ["точное название из кандидатов", ...]}}

2. Если кандидаты слишком похожи и непонятно, какую дисциплину имеет в виду пользователь — верни:
{{"status": "clarify", "candidates": ["кандидат 1", "кандидат 2"], "message": "Уточните, пожалуйста: вы имеете в виду «кандидат 1» или «кандидат 2»?"}}

3. Если ни один кандидат не подходит — верни:
{{"status": "not_found"}}

Правила:
- Используй ТОЛЬКО названия из списка кандидатов, без изменений.
- В "found" можно вернуть несколько дисциплин, если пользователь явно упомянул несколько.
- В "clarify" перечисляй только реально неоднозначные кандидаты (не все подряд).
- Если один из кандидатов явно точнее совпадает с запросом — выбирай его (status: found), не проси уточнения.

Кандидаты:
{candidates}

Запрос пользователя:
{query}

Ответь строго в формате JSON, без пояснений и markdown.
""".strip()

_PROMPT_CLASSIFY_SINGLE = """
Ты — ассистент, который классифицирует вопросы об учебных дисциплинах (РПД).

Определи тип запроса:

single.simple — вопрос об одном конкретном факте или поле РПД:
  - сколько часов / зачётных единиц
  - какая форма контроля (экзамен, зачёт)
  - в каком семестре
  - кто ведёт
  - какие компетенции формирует
  - есть ли лабораторные / практики

single.global — запрос на большой раздел, список или обзор дисциплины целиком:
  - примерные вопросы к экзамену / контрольной / зачёту
  - перечень тем / список тем раздела
  - список литературы / источников
  - критерии оценки / балльная система
  - задания для самостоятельной работы
  - чему учит курс / расскажи подробно / опиши программу
  - как устроен курс / структура дисциплины
  - содержание семестра

Правило: если ответ на вопрос — это большой текст или список (а не одно-два слова / число) → single.global.

Примеры:
  "Сколько часов лекций?" → single.simple
  "Какая форма итогового контроля?" → single.simple
  "В каком семестре изучается?" → single.simple
  "Какие компетенции формирует дисциплина?" → single.simple
  "Примерные вопросы к контрольной работе" → single.global
  "Вопросы для подготовки к экзамену" → single.global
  "Какие темы изучаются в 1 семестре?" → single.global
  "Расскажи о содержании курса" → single.global
  "Какая литература рекомендована?" → single.global
  "Как оценивается работа студентов?" → single.global
  "Какие задания для самостоятельной работы?" → single.global

Запрос пользователя:
{query}

Ответь строго в формате JSON, без пояснений и markdown:
{{"query_type": "single.simple"}} или {{"query_type": "single.global"}}
""".strip()

_PROMPT_CLASSIFY_ZERO = """
Ты — ассистент, который классифицирует учебные запросы.

Пользователь задал запрос, в котором не удалось распознать ни одной
дисциплины из списка учебных программ (РПД).
Определи тип запроса:

- "multi.global"  — запрос касается всего корпуса дисциплин:
  какие дисциплины есть, где изучается тема X, в каких курсах упоминается Y.

- "not_found"     — пользователь спрашивает про конкретную дисциплину,
  которой нет в системе (неизвестный курс, опечатка, вымышленное название).

- "irrelevant"    — запрос вообще не относится к учебным программам.

Запрос пользователя:
{query}

Ответь строго в формате JSON, без пояснений и markdown:
{{"query_type": "multi.global"}} или {{"query_type": "not_found"}} или {{"query_type": "irrelevant"}}
""".strip()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _llm_call(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=LLM_MODEL_FAST,
        max_tokens=LLM_MAX_TOKENS_FAST,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json(resp.choices[0].message.content)


_PROMPT_EXTRACT_QUERY_DISCIPLINE = """
Из запроса пользователя извлеки фрагмент, который похож на название учебной дисциплины.

Правила:
- Верни только предполагаемое название — без лишних слов ("по дисциплине", "курс", "предмет", номера семестров, вопросов и т.д.)
- Если в запросе несколько возможных названий — верни все.
- Если ничего похожего на название дисциплины нет — верни пустой список.
- Не исправляй и не дополняй — верни ровно то, что написал пользователь.

Примеры:
  "Сколько часов лекций по линейной алгебре?"
  → {{"names": ["линейная алгебра"]}}

  "Примерные вопросы к контрольной по дисциплине Алгоритмы и структура данных в Python для 1 семестра"
  → {{"names": ["Алгоритмы и структура данных в Python"]}}

  "Сравни машинное обучение и глубокое обучение"
  → {{"names": ["машинное обучение", "глубокое обучение"]}}

  "Какая погода сегодня?"
  → {{"names": []}}

Запрос пользователя:
{query}

Ответь строго в формате JSON, без пояснений и markdown.
""".strip()


def _extract_query_names(client: OpenAI, query: str) -> list[str]:
    """Шаг 0: LLM вытаскивает сырые названия из запроса."""
    prompt = _PROMPT_EXTRACT_QUERY_DISCIPLINE.format(query=query)
    try:
        data = _llm_call(client, prompt)
        names = data.get("names") or []
        return [n.lower().strip() for n in names if n.strip()]
    except Exception as exc:
        log.warning("extract_query_names failed: %s", exc)
        return []


def _fuzzy_candidates(extracted_names: list[str]) -> list[str]:
    """
    Fuzzy по коротким извлечённым именам — score теперь стабильный.
    Порог можно поднять до 70+ потому что сравниваем короткое с коротким.
    """
    if not extracted_names:
        return []

    found: dict[str, float] = {}
    for name in extracted_names:
        results = process.extract(
            name,
            RPD_NAMES,
            scorer=fuzz.token_set_ratio,
            limit=_FUZZY_TOP_K,
        )
        for match, score, _ in results:
            if score >= _FUZZY_THRESHOLD:
                if match not in found or score > found[match]:
                    found[match] = score

    return sorted(found, key=found.get, reverse=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    def __init__(self) -> None:
        self._client  = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self._rpd_set = set(RPD_NAMES)

    # ------------------------------------------------------------------
    # Шаг 1: fuzzy → LLM → дисциплины / уточнение / пусто
    # ------------------------------------------------------------------
    def _extract_disciplines(self, query: str) -> RouteResult | None:
        """
        Возвращает RouteResult только если уже можно завершить маршрутизацию:
        - нашли дисциплины (found)
        - нужно уточнение (clarify)
        Возвращает None если дисциплины не найдены — тогда route() идёт дальше.
        """
        # Шаг 0: LLM извлекает сырые имена из запроса
        extracted_names = _extract_query_names(self._client, query)
        log.debug("Extracted names from query: %s", extracted_names)

        # Шаг 1: fuzzy по коротким именам
        candidates = _fuzzy_candidates(extracted_names)
        log.debug("Fuzzy candidates: %s", candidates)

        if not candidates:
            return None

        candidates_text = "\n".join(f"- {c}" for c in candidates)
        prompt = _PROMPT_EXTRACT_DISCIPLINES.format(
            candidates=candidates_text,
            query=query,
        )

        try:
            data = _llm_call(self._client, prompt)
        except Exception as exc:
            log.warning("extract_disciplines LLM failed: %s", exc)
            return None

        status = data.get("status")

        if status == "found":
            # Фильтруем строго по RPD_NAMES на случай галлюцинаций
            raw = data.get("disciplines") or []
            disciplines = [d for d in raw if d in self._rpd_set]
            if disciplines:
                return RouteResult(
                    query_type=QueryType.CLARIFY,   # временный, route() заменит
                    disciplines=disciplines,
                    reasoning="found",
                )

        elif status == "clarify":
            ambiguous = [d for d in (data.get("candidates") or []) if d in self._rpd_set]
            message   = data.get("message", "Уточните, какую дисциплину вы имеете в виду.")
            if ambiguous:
                return RouteResult(
                    query_type=QueryType.CLARIFY,
                    disciplines=ambiguous,
                    reasoning=message,
                )

        # status == "not_found" или что-то неожиданное
        return None

    # ------------------------------------------------------------------
    # Шаг 2а: классифицировать запрос с одной дисциплиной
    # ------------------------------------------------------------------
    def _classify_single(self, query: str) -> QueryType:
        try:
            data = _llm_call(self._client, _PROMPT_CLASSIFY_SINGLE.format(query=query))
            return QueryType(data["query_type"])
        except Exception as exc:
            log.warning("classify_single failed: %s", exc)
            return QueryType.SINGLE_SIMPLE

    # ------------------------------------------------------------------
    # Шаг 2б: классифицировать запрос без дисциплин
    # ------------------------------------------------------------------
    def _classify_zero(self, query: str) -> QueryType:
        try:
            data = _llm_call(self._client, _PROMPT_CLASSIFY_ZERO.format(query=query))
            return QueryType(data["query_type"])
        except Exception as exc:
            log.warning("classify_zero failed: %s", exc)
            return QueryType.IRRELEVANT

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    def route(self, query: str) -> RouteResult:
        try:
            extraction = self._extract_disciplines(query)

            # Уточнение — возвращаем сразу, без дальнейшей классификации
            if extraction and extraction.query_type == QueryType.CLARIFY:
                if extraction.reasoning == "found":
                    # LLM нашёл уверенно — классифицируем по количеству
                    disciplines = extraction.disciplines
                    n = len(disciplines)
                    if n > 1:
                        query_type = QueryType.MULTI_RELATION
                    else:
                        query_type = self._classify_single(query)
                    result = RouteResult(query_type=query_type, disciplines=disciplines)
                else:
                    # LLM сомневается — просим уточнить
                    result = extraction

            else:
                # Дисциплины не найдены вообще
                query_type = self._classify_zero(query)
                result = RouteResult(query_type=query_type, disciplines=[])

        except Exception as exc:
            log.warning("Router fallback: %s", exc)
            return RouteResult(QueryType.SINGLE_SIMPLE, [], f"fallback: {exc}")

        log.info("Router: type=%s  disciplines=%s", result.query_type, result.disciplines)
        return result