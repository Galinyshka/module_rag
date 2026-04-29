"""
Модуль индексации обработанных JSON рабочих программ дисциплин.

Пайплайн для каждого блока:
  1. Сериализация блока в текст
  2. LLM-резюме   через OpenAI-совместимый API  → summary_vec
  3. LLM-вопросы  через OpenAI-совместимый API  → questions_vec
  4. Два эмбединга: questions_vec и summary_vec  (sentence-transformers, multilingual)
  5. Загрузка в Qdrant с named vectors + полным payload

Иерархия:
  topics              → overview-точка
    └─ topic_N        → дочерняя точка (parent_id = topics.id)
  competencies        → overview-точка
    └─ comp_K         → дочерняя точка
  self_study_resources → overview-точка
    └─ sub_N          → дочерняя точка (каждый подраздел отдельно)
  other_sections      → overview-точка (если есть подразделы)
    └─ sub_N          → дочерняя точка
  course_info / literature / assessment_fund / … → плоские точки

IDs детерминированы: uuid5(discipline + block_type + local_key)
— повторный запуск = upsert, без дублирования.

Зависимости:
    pip install openai sentence-transformers qdrant-client
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL_IDX,
    LLM_MAX_TOKENS_IDX,
    LLM_RETRY_DELAY,
    LLM_MAX_RETRIES,
    QDRANT_URL,
    QDRANT_COLLECTION,
    VEC_QUESTIONS,   # было VEC_TEXT
    VEC_SUMMARY,
    EMBED_MODEL,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
    QUESTIONS_COUNT,
)
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# избегаем дублирования хендлеров (важно при reload / notebook / uvicorn)
if not log.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # file
    file_handler = logging.FileHandler("indexing.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    log.addHandler(console_handler)
    log.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Block dataclass
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """Единица индексации — один смысловой блок или его дочерний элемент."""
    block_id:   str
    block_type: str
    block_name: str
    text:       str
    parent_id:  str | None      = None
    summary:    str             = ""
    questions:  str             = ""   # вопросы, на которые отвечает блок
    metadata:   dict[str, Any]  = field(default_factory=dict)

    def add_meta(self, **kwargs) -> "Block":
        self.metadata.update(kwargs)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(discipline: str, *parts: str) -> str:
    key = "|".join([discipline, *parts])
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _flatten(obj: Any, sep: str = "\n") -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        return sep.join(_flatten(item) for item in obj if item)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            v_str = _flatten(v)
            if v_str:
                parts.append(f"{k}: {v_str}")
        return sep.join(parts)
    return str(obj)


def _section_text(section: Any) -> str:
    if section is None:
        return ""
    if isinstance(section, str):
        return section
    parts = []
    if isinstance(section, dict):
        if section.get("text"):
            parts.append(section["text"])
        children = section.get("children") or {}
        if isinstance(children, dict):
            for title, body in children.items():
                parts.append(f"{title}:\n{_flatten(body)}")
        elif isinstance(children, list):
            parts.extend(_flatten(c) for c in children)
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 1. BlockBuilder
# ---------------------------------------------------------------------------

class BlockBuilder:

    def build(self, data: dict) -> list[Block]:
        discipline = data.get("discipline", "")
        year       = str(data.get("year") or "")
        base       = {"discipline": discipline, "year": year}
        blocks: list[Block] = []

        blocks.append(self._course_info(data, discipline, base))
        blocks.extend(self._topics(data, discipline, base))
        blocks.extend(self._competencies(data, discipline, base))
        blocks.extend(self._self_study(data, discipline, base))   # ← новый метод

        for key, btype, label in [
            ("assessment_fund", "assessment_fund",
             "Фонд оценочных средств / примерные вопросы к аттестации"),
            ("literature",      "literature",
             "Перечень учебной литературы"),
            ("online_resources", "online_resources",
             "Интернет-ресурсы для освоения дисциплины"),
        ]:
            b = self._simple_section(
                data.get(key), btype, label, discipline, {**base, "block": key}
            )
            if b:
                blocks.append(b)

        blocks.extend(self._other_sections(data.get("other_sections"), discipline, base))
        return blocks

    # --- course_info ---

    def _course_info(self, data: dict, discipline: str, base: dict) -> Block:
        ci = data.get("course_info") or {}
        lines = [
            f"Дисциплина: {data.get('discipline', '')}",
            f"Год: {data.get('year', '')}",
            f"Кафедра: {ci.get('department', '')}",
            f"Компетенции: {ci.get('компетенции', '')}",
            f"Зачётные единицы: {ci.get('зачетные_единицы_всего', '')}",
            f"Трудоёмкость (часов): {ci.get('трудоемкость_в_часах_всего', '')}",
            f"Аудиторная работа: {ci.get('аудиторная_всего', '')} ч."
            f" (лекции: {ci.get('аудиторная_всего_лекции', '')}, "
            f"семинары: {ci.get('аудиторная_всего_семинары', '')})",
            f"Самостоятельная работа: {ci.get('аудиторная_всего_самостоятельная_работа', '')} ч.",
        ]
        for sem_num, sem in (ci.get("семестры") or {}).items():
            lines.append(
                f"Семестр {sem_num}: лекции {sem.get('лекции')} ч., "
                f"семинары {sem.get('семинары')} ч., "
                f"сам. работа {sem.get('самостоятельная_работа_часов')} ч., "
                f"аттестация: {sem.get('вид_промежуточной_аттестации')}, "
                f"текущий контроль: {sem.get('вид_текущего_контроля')}"
            )
        if ci.get("info"):
            lines.append(f"\nАннотация: {ci['info']}")

        return Block(
            block_id   = _make_id(discipline, "course_info"),
            block_type = "course_info",
            block_name = f"Общие сведения о дисциплине «{discipline}»",
            text       = "\n".join(lines),
        ).add_meta(**base, block="course_info")

    # --- topics ---

    def _topics(self, data: dict, discipline: str, base: dict) -> list[Block]:
        topics_list = data.get("topics") or []
        if not topics_list:
            return []

        parent_id = _make_id(discipline, "topics")
        overview_lines = [
            f"Тема {t.get('topic_index')}: {t.get('название_темы', '')}  "
            f"({t.get('трудоемкость_в_часах_всего', '?')} ч.)"
            for t in topics_list
        ]
        parent = Block(
            block_id   = parent_id,
            block_type = "topics",
            block_name = f"Тематический план дисциплины «{discipline}» "
                         f"({len(topics_list)} тем)",
            text       = "\n".join(overview_lines),
        ).add_meta(**base, block="topics", topics_count=len(topics_list))

        children = []
        for topic in topics_list:
            idx   = topic.get("topic_index", "?")
            title = topic.get("название_темы", "")
            parts = [f"Тема {idx}. {title}"]

            if topic.get("содержание_темы"):
                parts.append(f"Содержание:\n{topic['содержание_темы']}")

            pz = topic.get("практические_занятия") or {}
            if pz.get("перечень_вопросов_для_обсуждения"):
                parts.append(
                    f"Практические занятия ({pz.get('форма_проведения') or ''}):\n"
                    f"{pz['перечень_вопросов_для_обсуждения']}"
                )

            sr = topic.get("самостоятельная_работа") or {}
            if sr.get("перечень_вопросов_для_освоения"):
                parts.append(
                    f"Самостоятельная работа ({sr.get('форма_работы') or ''}):\n"
                    f"{sr['перечень_вопросов_для_освоения']}"
                )

            hours = (
                f"Трудоёмкость: {topic.get('трудоемкость_в_часах_всего')} ч. "
                f"(лекции {(topic.get('аудиторная_работа') or {}).get('лекции')}, "
                f"семинары {(topic.get('аудиторная_работа') or {}).get('семинары')}, "
                f"сам. работа {topic.get('самостоятельная_работа_часов')})"
            )
            parts.append(hours)

            if topic.get("форма_текущего_контроля_успеваемости"):
                parts.append(
                    f"Текущий контроль: {topic['форма_текущего_контроля_успеваемости']}"
                )

            children.append(
                Block(
                    block_id   = _make_id(discipline, "topic", str(idx)),
                    block_type = "topic",
                    block_name = f"Тема {idx}: {title}",
                    text       = "\n\n".join(parts),
                    parent_id  = parent_id,
                ).add_meta(
                    **base,
                    block       = "topic",
                    topic_index = idx,
                    topic_title = title,
                    hours_total = topic.get("трудоемкость_в_часах_всего"),
                )
            )

        return [parent, *children]

    # --- competencies ---

    def _competencies(self, data: dict, discipline: str, base: dict) -> list[Block]:
        comp_list = data.get("competencies") or []
        if not comp_list:
            return []

        parent_id = _make_id(discipline, "competencies")
        overview_lines = [
            f"{c.get('код_компетенции')}: {c.get('наименование_компетенции', '')}"
            for c in comp_list
        ]
        parent = Block(
            block_id   = parent_id,
            block_type = "competencies",
            block_name = f"Компетенции дисциплины «{discipline}» "
                         f"({len(comp_list)} компетенции)",
            text       = "\n".join(overview_lines),
        ).add_meta(**base, block="competencies", competencies_count=len(comp_list))

        children = []
        for comp in comp_list:
            code = comp.get("код_компетенции", "")
            name = comp.get("наименование_компетенции", "")
            parts = [f"Компетенция {code}: {name}"]

            for ind in (comp.get("индикаторы_достижения") or []):
                ind_parts = []
                if ind.get("наименование_индикатора"):
                    ind_parts.append(f"Индикатор: {ind['наименование_индикатора']}")
                if ind.get("результаты_обучения"):
                    ind_parts.append(f"Результаты обучения: {ind['результаты_обучения']}")
                if ind.get("типовые_контрольные_задания"):
                    ind_parts.append(f"Контрольные задания: {ind['типовые_контрольные_задания']}")
                if ind_parts:
                    parts.append("\n".join(ind_parts))

            children.append(
                Block(
                    block_id   = _make_id(discipline, "competency", code),
                    block_type = "competency",
                    block_name = f"Компетенция {code}",
                    text       = "\n\n".join(parts),
                    parent_id  = parent_id,
                ).add_meta(
                    **base,
                    block            = "competency",
                    competency_code  = code,
                    indicators_count = len(comp.get("индикаторы_достижения") or []),
                )
            )

        return [parent, *children]

    # --- self_study_resources (НОВЫЙ МЕТОД — вместо _simple_section) ---

    def _self_study(self, data: dict, discipline: str, base: dict) -> list[Block]:
        """
        self_study_resources — словарь {заголовок_подраздела: текст}.
        Каждый подраздел индексируется отдельным блоком чтобы:
          - заголовок подраздела стал block_name (основа для questions_vec)
          - summary LLM правильно описал именно этот подраздел
          - поиск находил нужный раздел (вопросы к контрольной, к экзамену и т.д.)

        Старый _simple_section был неверен: _section_text искал ключи "text"/"children",
        которых нет — возвращал пустую строку и блок молча выбрасывался.
        """
        section = data.get("self_study_resources")
        if not section:
            return []

        # Если вдруг строка — один плоский блок
        if isinstance(section, str):
            if not section.strip():
                return []
            return [Block(
                block_id   = _make_id(discipline, "self_study_resources"),
                block_type = "self_study_resources",
                block_name = "Учебно-методическое обеспечение самостоятельной работы",
                text       = section,
            ).add_meta(**base, block="self_study_resources")]

        if not isinstance(section, dict):
            return []

        parent_id = _make_id(discipline, "self_study_resources")
        parent = Block(
            block_id   = parent_id,
            block_type = "self_study_resources",
            block_name = "Учебно-методическое обеспечение самостоятельной работы",
            text       = "\n".join(k for k in section),
        ).add_meta(**base, block="self_study_resources")

        children = []
        for idx, (title, content) in enumerate(section.items()):
            text = content if isinstance(content, str) else _flatten(content)
            if not text.strip():
                continue
            children.append(
                Block(
                    block_id   = _make_id(discipline, "self_study", str(idx)),
                    block_type = "self_study_section",
                    # Заголовок подраздела как block_name — LLM сгенерирует
                    # вопросы именно про этот раздел ("вопросы к контрольной" и т.д.)
                    block_name = title,
                    text       = f"{title}\n\n{text}",
                    parent_id  = parent_id,
                ).add_meta(**base, block="self_study_section", section_title=title)
            )

        return [parent, *children] if children else []

    # --- простые плоские блоки ---

    def _simple_section(
        self,
        section:    Any,
        btype:      str,
        label:      str,
        discipline: str,
        meta:       dict,
    ) -> Block | None:
        text = _section_text(section)
        if not text.strip():
            return None
        return Block(
            block_id   = _make_id(discipline, btype),
            block_type = btype,
            block_name = label,
            text       = text,
        ).add_meta(**meta)

    # --- other_sections ---
    ''' def _other_sections(self, other: Any, discipline: str, base: dict) -> list[Block]:
        if not other:
            return []

        if isinstance(other, str):
            b = self._simple_section(
                other, "other_sections",
                "Дополнительные разделы рабочей программы",
                discipline, {**base, "block": "other_sections"}
            )
            return [b] if b else []

        if not isinstance(other, dict):
            return []

        is_single = set(other.keys()) <= {"text", "children"}
        if is_single:
            b = self._simple_section(
                other, "other_sections",
                "Дополнительные разделы рабочей программы",
                discipline, {**base, "block": "other_sections"}
            )
            return [b] if b else []

        parent_id = _make_id(discipline, "other_sections")
        parent = Block(
            block_id   = parent_id,
            block_type = "other_sections",
            block_name = "Дополнительные разделы рабочей программы",
            text       = "\n".join(k[:60] for k in other),
        ).add_meta(**base, block="other_sections", subsections=list(other.keys()))

        children = []
        for idx, (title, content) in enumerate(other.items()):
            text = _section_text(content)
            if not text.strip():
                continue
            children.append(
                Block(
                    block_id   = _make_id(discipline, "other_section", str(idx)),
                    block_type = "other_section",
                    block_name = title,
                    text       = text,
                    parent_id  = parent_id,
                ).add_meta(**base, block="other_section", section_title=title)
            )

        if not children:
            return []
        return [parent, *children]'''

    def _other_sections(self, other: Any, discipline: str, base: dict) -> list[Block]:
        if not other:
            return []

        if isinstance(other, str):
            b = self._simple_section(
                other,
                "other_sections",
                "Дополнительные разделы рабочей программы",
                discipline,
                {**base, "block": "other_sections"},
            )
            return [b] if b else []

        if not isinstance(other, dict):
            return []

        is_single = set(other.keys()) <= {"text", "children"}
        if is_single:
            b = self._simple_section(
                other,
                "other_sections",
                "Дополнительные разделы рабочей программы",
                discipline,
                {**base, "block": "other_sections"},
            )
            return [b] if b else []

        parent_id = _make_id(discipline, "other_sections")

        # 🔷 Parent = только структура, без текста вообще
        parent = Block(
            block_id=parent_id,
            block_type="other_sections",
            block_name="Дополнительные разделы рабочей программы",
            text="",  # важно: не индексируем шум
        ).add_meta(
            **base,
            block="other_sections",
            subsections=list(other.keys()),
            children_count=len(other),
        )

        children = []
        for idx, (title, content) in enumerate(other.items()):
            text = _section_text(content)
            if not text or not text.strip():
                continue

            children.append(
                Block(
                    block_id=_make_id(discipline, "other_section", str(idx)),
                    block_type="other_section",
                    block_name=title,
                    text=text,
                    parent_id=parent_id,
                ).add_meta(
                    **base,
                    block="other_section",
                    section_title=title,
                )
            )

        if not children:
            return []

        return [parent, *children]


# ---------------------------------------------------------------------------
# 2. Summarizer
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
Ты — ассистент, создающий описания блоков рабочей программы дисциплины \
для векторного поиска.

Твоя задача — описать, КАКАЯ ИМЕННО ИНФОРМАЦИЯ содержится в блоке, \
а не пересказывать его содержание.
Описание должно явно называть типы данных, присутствующие в блоке, \
чтобы поисковые запросы о них находили этот блок.

Упоминай всё что есть: названия тем, числа часов (лекции/семинары/сам. работа), \
компетенции и их коды, форму контроля, перечни вопросов, литературу и т.п.
Пиши 2–4 предложения на русском языке, конкретно и без воды.

Примеры хороших описаний:
- «Блок о теме "Функции в Python". Содержит трудоёмкость: 16 ч. \
(лекции 2 ч., семинары 4 ч., самостоятельная работа 10 ч.). \
Включает перечень вопросов для семинара по объявлению и вызову функций, \
форма текущего контроля — самостоятельное решение задач.»
- «Блок содержит сводную информацию о дисциплине: кафедру, перечень компетенций \
(ПКН-4, SS-1), общую трудоёмкость 8 зачётных единиц (288 часов), \
разбивку по двум семестрам, форму аттестации — экзамен.»
- «Блок описывает раздел самостоятельной работы. Содержит примерные вопросы \
к контрольной работе для 1 и 2 семестра, задания на программирование на Python, \
критерии балльной оценки текущего контроля успеваемости.»

Тип блока: {block_type}
Название: {block_name}

Текст блока:
{text}
""".strip()


class Summarizer:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def summarize(self, block: Block) -> str:
        prompt = SUMMARY_PROMPT.format(
            block_type = block.block_type,
            block_name = block.block_name,
            text       = block.text,
        )
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model      = LLM_MODEL_IDX,
                    max_tokens = LLM_MAX_TOKENS_IDX,
                    messages   = [{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                if attempt < LLM_MAX_RETRIES:
                    log.warning("Rate limit, повтор через %d с. (попытка %d/%d)",
                                LLM_RETRY_DELAY, attempt, LLM_MAX_RETRIES)
                    time.sleep(LLM_RETRY_DELAY * attempt)
                else:
                    log.error("Rate limit исчерпан, резюме пропущено: %s", block.block_id)
                    return ""
            except Exception as exc:
                log.error("Ошибка Summarizer для %s: %s", block.block_id, exc)
                return ""
        return ""


# ---------------------------------------------------------------------------
# 3. QuestionsGenerator
# ---------------------------------------------------------------------------


QUESTIONS_PROMPT = """\
Ты — эксперт по созданию поисковых запросов для векторной базы учебных программ дисциплин.

Твоя задача: для данного блока рабочей программы дисциплины сгенерировать **ровно {n}** очень качественных и разнообразных вопросов, на которые этот блок содержит точный и полный ответ.

### Требования к вопросам (обязательно соблюдать):

1. **Максимальное разнообразие**:
   - Разные типы вопросов: "сколько / как много", "какие именно", "что входит / что включает", "как проводится / в какой форме", "есть ли / предусмотрено ли", "каковы критерии / требования".
   - Избегай шаблонных начал. Не начинай больше одного вопроса со слов "Сколько...", "Какие...", "Что включает...".

2. **Высокая специфичность**:
   - Обязательно используй реальные названия из блока (название темы, компетенции, раздела самостоятельной работы и т.д.).
   - Не используй обобщения типа "в этой теме", "в этом блоке", "здесь".
   - Вопрос должен быть понятен без контекста блока.

3. **Реалистичность**:
   - Вопросы должны звучать как реальные запросы студентов или преподавателей.
   - Примеры хорошего стиля: 
     - "Сколько часов самостоятельной работы предусмотрено по теме «Словари и множества»?"
     - "Какие вопросы выносятся на практические занятия по теме «Рекурсия»?"
     - "В какой форме проходит текущий контроль по дисциплине «Алгоритмы и структуры данных в Python»?"
     - "Какие типовые контрольные задания есть по индикатору ПКН-4?"

### Примеры хороших наборов вопросов:

Блок: Тема 3: Рекурсия
Хорошие вопросы:
Сколько часов отведено на изучение рекурсии в дисциплине «Алгоритмы и структуры данных»?
Какие задачи на рекурсию разбираются на практических занятиях?
В какой форме студенты сдают самостоятельную работу по теме «Рекурсия»?

Блок: Общие сведения о дисциплине
Хорошие вопросы:
Какая форма итоговой аттестации по дисциплине «Алгоритмы и структуры данных в Python»?
Сколько всего зачётных единиц и академических часов составляет дисциплина?
Какие компетенции формирует курс «Алгоритмы и структуры данных в Python»?

### Сейчас сгенерируй вопросы для следующего блока:

Тип блока: {block_type}
Название блока: {block_name}

Текст блока:
{text}

Сгенерируй **ровно {n}** вопросов. Каждый вопрос — с новой строки, без нумерации, без кавычек и дополнительных пояснений.
""".strip()

class QuestionsGenerator:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def generate(self, block: Block, n: int = QUESTIONS_COUNT) -> str:
        """
        Генерирует n вопросов к блоку.
        Возвращает их одной строкой через \\n — она и будет эмбедирована как questions_vec.
        Логика: запрос пользователя семантически близок к вопросам,
        а не к канцелярским названиям разделов РПД.
        """
        prompt = QUESTIONS_PROMPT.format(
            n          = n,
            block_type = block.block_type,
            block_name = block.block_name,
            text       = block.text,
        )
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model      = LLM_MODEL_IDX,
                    max_tokens = LLM_MAX_TOKENS_IDX,
                    messages   = [{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                if attempt < LLM_MAX_RETRIES:
                    log.warning("Rate limit, повтор через %d с. (попытка %d/%d)",
                                LLM_RETRY_DELAY, attempt, LLM_MAX_RETRIES)
                    time.sleep(LLM_RETRY_DELAY * attempt)
                else:
                    log.error("Rate limit исчерпан, вопросы пропущены: %s", block.block_id)
                    return block.block_name  # fallback
            except Exception as exc:
                log.error("Ошибка QuestionsGenerator для %s: %s", block.block_id, exc)
                return block.block_name
        return block.block_name


# ---------------------------------------------------------------------------
# 4. Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self) -> None:
        log.info("Загрузка модели эмбедингов: %s ...", EMBED_MODEL)
        self._model = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# 5. QdrantStore
# ---------------------------------------------------------------------------

class QdrantStore:
    def __init__(
        self,
        url:        str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
    ) -> None:
        self._collection = collection
        self._client = (
            QdrantClient(":memory:")
            if url == ":memory:"
            else QdrantClient(url=url)
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    VEC_QUESTIONS: VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                    VEC_SUMMARY:   VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                },
            )
            log.info("Коллекция «%s» создана (dim=%d).", self._collection, EMBED_DIM)
        else:
            log.info("Коллекция «%s» уже существует.", self._collection)

    def upsert(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self._client.upsert(collection_name=self._collection, points=points)
        log.info("Загружено %d точек в «%s».", len(points), self._collection)

    def count(self) -> int:
        return self._client.count(self._collection).count


# ---------------------------------------------------------------------------
# 6. Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """
    Полный пайплайн: JSON → блоки → резюме + вопросы → эмбединги → Qdrant.

    Два вектора на блок:
      questions_vec — эмбединг вопросов, на которые отвечает блок.
                      Запросы пользователя семантически близки к вопросам.
      summary_vec   — эмбединг описания содержимого блока.
                      Хорошо ловит описательные / тематические запросы.
    Поиск ведётся по обоим через RRF (reciprocal rank fusion) в retrieval.py.
    """

    def __init__(
        self,
        qdrant_url:   str  = QDRANT_URL,
        collection:   str  = QDRANT_COLLECTION,
        skip_summary: bool = False,
    ) -> None:
        self._builder    = BlockBuilder()
        self._summarizer = None if skip_summary else Summarizer()
        self._questions  = None if skip_summary else QuestionsGenerator()
        self._embedder   = Embedder()
        self._store      = QdrantStore(qdrant_url, collection)

    def index_file(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self.index(data)

    def index(self, data: dict) -> int:
        discipline = data.get("discipline", "unknown")
        log.info("Индексация: «%s»", discipline)

        blocks = self._builder.build(data)
        log.info("Построено %d блоков.", len(blocks))

        similarities: list[float] = []   # ← для статистики

        if self._summarizer:
            for i, block in enumerate(blocks, 1):
                log.info("[%d/%d] %s", i, len(blocks), block.block_name)

                block.summary   = self._summarizer.summarize(block)
                block.questions = self._questions.generate(block)

                log.info("  summary:   %s", block.summary[:200] + "..." if len(block.summary) > 200 else block.summary)
                log.info("  questions: %s", block.questions)

                # === Проверка схожести ===
                if block.summary.strip() and block.questions.strip():
                    sim = self._compute_questions_summary_similarity(block.questions, block.summary)
                    similarities.append(sim)
                    log.info("  similarity (q vs s): %.4f", sim)

                    if sim > 0.78:
                        log.warning("  ⚠️  ВЫСОКАЯ схожесть (%.4f) — questions_vec может быть слабым!", sim)
                    elif sim > 0.72:
                        log.info("  ⚠️  Средняя схожесть (%.4f)", sim)
                # =========================
        else:
            log.info("Режим --skip-summary: вопросы и summary не генерируются.")

        points = self._build_points(blocks)
        self._store.upsert(points)

        # === Итоговая статистика схожести ===
        if similarities:
            avg_sim = np.mean(similarities)
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            high_sim_count = sum(1 for s in similarities if s > 0.78)

            log.info("=" * 60)
            log.info("СТАТИСТИКА СХОЖЕСТИ questions_vec vs summary_vec для дисциплины «%s»", discipline)
            log.info("Средняя схожесть: %.4f", avg_sim)
            log.info("Максимальная:     %.4f", max_sim)
            log.info("Минимальная:      %.4f", min_sim)
            log.info("Блоков с высокой схожестью (>0.78): %d из %d (%.1f%%)",
                     high_sim_count, len(similarities), 100 * high_sim_count / len(similarities))
            log.info("=" * 60)

            if avg_sim > 0.75:
                log.warning("❗ Рекомендуется улучшить промпт генерации вопросов — слишком высокая схожесть!")
            elif avg_sim < 0.68:
                log.info("✅ Отличное разделение сигналов между questions_vec и summary_vec.")

        log.info(
            "Готово: %d точек → коллекция «%s» (итого в БД: %d).",
            len(points), self._store._collection, self._store.count()
        )
        return len(points)

    def _build_points(self, blocks: list[Block]) -> list[PointStruct]:
        # questions_vec: вопросы к блоку — близки к живым запросам пользователя
        # Fallback на block_name если вопросы не сгенерированы (--skip-summary)
        questions = [b.questions or b.block_name for b in blocks]

        # summary_vec: описание содержимого — близко к тематическим запросам
        summaries = [b.summary or b.block_name for b in blocks]

        question_vecs = self._batch_encode(questions)
        summary_vecs  = self._batch_encode(summaries)

        points = []
        for block, qv, sv in zip(blocks, question_vecs, summary_vecs):
            payload = {
                "block_type": block.block_type,
                "block_name": block.block_name,
                "text":       block.text,
                "summary":    block.summary,
                "questions":  block.questions,  # сохраняем для отладки
                "parent_id":  block.parent_id,
                **block.metadata,
            }
            points.append(PointStruct(
                id     = block.block_id,
                vector = {
                    VEC_QUESTIONS: qv,
                    VEC_SUMMARY:   sv,
                },
                payload = payload,
            ))
        return points

    def _batch_encode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i: i + EMBED_BATCH_SIZE]
            result.extend(self._embedder.encode(batch))
            log.debug("Эмбединги: %d/%d", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))
        return result

    def _compute_questions_summary_similarity(self, questions_text: str, summary_text: str) -> float:
        """Вычисляет косинусную схожесть между questions и summary одного блока."""
        if not questions_text.strip() or not summary_text.strip():
            return 0.0
        try:
            vec_q = self._embedder.encode_one(questions_text)
            vec_s = self._embedder.encode_one(summary_text)

            vec_q = np.array(vec_q).reshape(1, -1)
            vec_s = np.array(vec_s).reshape(1, -1)

            sim = cosine_similarity(vec_q, vec_s)[0][0]
            return float(sim)
        except Exception as e:
            log.error("Ошибка при вычислении similarity для блока: %s", e)
            return 0.0
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Индексация JSON-файла или папки с JSON-файлами"
    )
    parser.add_argument("input",          help="Путь к JSON-файлу или папке")
    parser.add_argument("--qdrant",       default=QDRANT_URL)
    parser.add_argument("--collection",   default=QDRANT_COLLECTION)
    parser.add_argument("--skip-summary", action="store_true",
                        help="Пропустить LLM-шаг (без резюме и вопросов)")
    parser.add_argument("--memory",       action="store_true",
                        help="In-memory Qdrant (тест)")
    args = parser.parse_args()

    indexer = Indexer(
        qdrant_url   = ":memory:" if args.memory else args.qdrant,
        collection   = args.collection,
        skip_summary = args.skip_summary,
    )

    input_path = Path(args.input)
    total = 0

    if input_path.is_file():
        total += indexer.index_file(str(input_path))
    elif input_path.is_dir():
        files = list(input_path.glob("*.json"))
        if not files:
            print("❌ В папке нет JSON-файлов")
            exit(1)
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {f}")
            try:
                n = indexer.index_file(str(f))
                total += n
                print(f"  → {n} блоков")
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
    else:
        print("❌ Путь не существует")
        exit(1)

    print(f"\n✅ Итого проиндексировано: {total} блоков")
