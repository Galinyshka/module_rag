"""
Модуль индексации обработанных JSON рабочих программ дисциплин.

Пайплайн для каждого блока:
  1. Сериализация блока в текст
  2. LLM-резюме через OpenAI-совместимый API
  3. Два эмбединга: text_vec и summary_vec  (sentence-transformers, multilingual)
  4. Загрузка в Qdrant с named vectors + полным payload

Иерархия:
  topics         → overview-точка
    └─ topic_N   → дочерняя точка (parent_id = topics.id)
  competencies   → overview-точка
    └─ comp_K    → дочерняя точка
  other_sections → overview-точка (если есть подразделы)
    └─ sub_N     → дочерняя точка
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
from qdrant_client.models import Distance, PointStruct, VectorParams, VectorsConfig

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL_IDX,
    LLM_MAX_TOKENS_IDX,
    LLM_RETRY_DELAY,
    LLM_MAX_RETRIES,
    QDRANT_URL,
    QDRANT_COLLECTION,
    VEC_TEXT,
    VEC_SUMMARY,
    EMBED_MODEL,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Block dataclass
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """Единица индексации — один смысловой блок или его дочерний элемент."""
    block_id:   str
    block_type: str                      # course_info / topic / competency / …
    block_name: str                      # человекочитаемое название
    text:       str                      # полный текст для эмбединга
    parent_id:  str | None   = None      # None — корневой блок
    summary:    str          = ""        # заполняет Summarizer
    metadata:   dict[str, Any] = field(default_factory=dict)

    def add_meta(self, **kwargs) -> "Block":
        self.metadata.update(kwargs)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(discipline: str, *parts: str) -> str:
    """Детерминированный UUID5 на основе дисциплины и любых ключей."""
    key = "|".join([discipline, *parts])
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _flatten(obj: Any, sep: str = "\n") -> str:
    """Рекурсивно собирает строки из dict / list / str."""
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
    """Собирает текст из блока вида {text: ..., children: {...}}."""
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
# 1. BlockBuilder — строит иерархию Block из JSON
# ---------------------------------------------------------------------------

class BlockBuilder:
    """
    Строит плоский список Block из processed JSON.
    Родительские блоки хранят overview; дочерние — полное содержание.
    """

    def build(self, data: dict) -> list[Block]:
        discipline = data.get("discipline", "")
        year       = str(data.get("year") or "")
        base       = {"discipline": discipline, "year": year}
        blocks: list[Block] = []

        blocks.append(self._course_info(data, discipline, base))
        blocks.extend(self._topics(data, discipline, base))
        blocks.extend(self._competencies(data, discipline, base))

        for key, btype, label in [
            ("self_study_resources", "self_study_resources",
             "Учебно-методическое обеспечение самостоятельной работы"),
            ("assessment_fund",      "assessment_fund",
             "Фонд оценочных средств / примерные вопросы к аттестации"),
            ("literature",           "literature",
             "Перечень учебной литературы"),
            ("online_resources",     "online_resources",
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

    # --- topics (родитель + дочерние) ---

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

    # --- competencies (родитель + дочерние) ---

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
                    block_name = f"Компетенция {code}: {name[:60]}",
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

    # --- other_sections (родитель + дочерние подразделы) ---

    def _other_sections(self, other: Any, discipline: str, base: dict) -> list[Block]:
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

        # Одиночный блок {text, children} или словарь подразделов
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
                    block_name = title[:120],
                    text       = text,
                    parent_id  = parent_id,
                ).add_meta(**base, block="other_section", section_title=title)
            )

        if not children:
            return []
        return [parent, *children]


# ---------------------------------------------------------------------------
# 2. Summarizer — LLM-резюме через OpenAI-совместимый API
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
- «Блок описывает компетенцию ПКН-4 "Способен проектировать программные системы". \
Содержит 3 индикатора достижения, планируемые знания и умения по языку Python, \
типовые контрольные задания на разработку и тестирование кода.»

Тип блока: {block_type}
Название: {block_name}

Текст блока:
{text}
"""


class Summarizer:
    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def summarize(self, block: Block) -> str:
        prompt = SUMMARY_PROMPT.format(
            block_type = block.block_type,
            block_name = block.block_name,
            text       = block.text[:6000],
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
                log.error("Ошибка LLM для %s: %s", block.block_id, exc)
                return ""
        return ""


# ---------------------------------------------------------------------------
# 3. Embedder — sentence-transformers, multilingual
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
# 4. QdrantStore — векторная БД
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
                    VEC_TEXT: VectorParams(
                        size=EMBED_DIM,
                        distance=Distance.COSINE,
                    ),
                    VEC_SUMMARY: VectorParams(
                        size=EMBED_DIM,
                        distance=Distance.COSINE,
                    ),
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
# 5. Indexer — оркестратор
# ---------------------------------------------------------------------------

class Indexer:
    """
    Полный пайплайн: JSON → блоки → резюме → эмбединги → Qdrant.

    Параметры:
        qdrant_url   — URL Qdrant или ":memory:" для тестов
        collection   — имя коллекции
        skip_summary — True = пропустить LLM-шаг (быстро, без резюме)
    """

    def __init__(
        self,
        qdrant_url:   str  = QDRANT_URL,
        collection:   str  = QDRANT_COLLECTION,
        skip_summary: bool = False,
    ) -> None:
        self._builder    = BlockBuilder()
        self._summarizer = None if skip_summary else Summarizer()
        self._embedder   = Embedder()
        self._store      = QdrantStore(qdrant_url, collection)

    def index_file(self, path: str) -> int:
        """Индексирует один JSON-файл. Возвращает число загруженных точек."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self.index(data)

    def index(self, data: dict) -> int:
        """Индексирует уже загруженный dict. Возвращает число загруженных точек."""
        discipline = data.get("discipline", "unknown")
        log.info("Индексация: «%s»", discipline)

        blocks = self._builder.build(data)
        log.info("Построено %d блоков.", len(blocks))

        if self._summarizer:
            for i, block in enumerate(blocks, 1):
                log.info("[%d/%d] блок: %s", i, len(blocks), block.block_name)

                summary = self._summarizer.summarize(block)

                log.info("        summary: %s", (summary or ""))

                block.summary = summary

        points = self._build_points(blocks)
        self._store.upsert(points)
        log.info("Готово: %d точек → коллекция «%s» (итого в БД: %d).",
                 len(points), self._store._collection, self._store.count())
        return len(points)

    def _build_points(self, blocks: list[Block]) -> list[PointStruct]:
        # Эмбединг названия раздела — короткий, всегда влезает в лимит модели
        names     = [b.block_name          for b in blocks]
        # Эмбединг резюме — семантически плотное представление содержимого
        # Fallback на block_name если резюме не было сгенерировано
        summaries = [b.summary or b.block_name for b in blocks]

        text_vecs    = self._batch_encode(names)
        summary_vecs = self._batch_encode(summaries)

        points = []
        for block, tv, sv in zip(blocks, text_vecs, summary_vecs):
            payload = {
                "block_type": block.block_type,
                "block_name": block.block_name,
                "text":       block.text,
                "summary":    block.summary,
                "parent_id":  block.parent_id,
                **block.metadata,
            }
            points.append(PointStruct(
                id      = block.block_id,
                vector = {VEC_TEXT: tv, VEC_SUMMARY: sv},
                payload = payload,
            ))
        return points

    def _batch_encode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            result.extend(self._embedder.encode(batch))
            log.debug("Эмбединги: %d/%d", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Индексация JSON-файла или папки с JSON-файлами"
    )
    parser.add_argument(
        "input",
        help="Путь к JSON-файлу или папке с JSON-файлами"
    )
    parser.add_argument("--qdrant",       default=QDRANT_URL,        help="URL Qdrant")
    parser.add_argument("--collection",   default=QDRANT_COLLECTION, help="Имя коллекции")
    parser.add_argument("--skip-summary", action="store_true",       help="Пропустить LLM-резюме")
    parser.add_argument("--memory",       action="store_true",       help="In-memory Qdrant (тест)")
    args = parser.parse_args()

    indexer = Indexer(
        qdrant_url=":memory:" if args.memory else args.qdrant,
        collection=args.collection,
        skip_summary=args.skip_summary,
    )

    input_path = Path(args.input)
    total = 0

    # 📄 Один файл
    if input_path.is_file():
        total += indexer.index_file(str(input_path))

    # 📁 Папка с файлами
    elif input_path.is_dir():
        files = list(input_path.glob("*.json"))

        if not files:
            print("❌ В папке нет JSON-файлов")
            exit(1)

        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Индексация: {f}")
            try:
                n = indexer.index_file(str(f))
                total += n
                print(f"  → {n} блоков")
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")

    else:
        print("❌ Указанный путь не существует")
        exit(1)

    print(f"\n✅ Итого проиндексировано: {total} блоков")
