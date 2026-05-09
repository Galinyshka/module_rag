"""
DisciplineCatalog — лёгкий in-memory индекс всех РПД.
Строится один раз при старте из JSON-файлов в test_data/.
Используется для MULTI_GLOBAL запросов без обращения в Qdrant.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------

@dataclass
class CompetencyInfo:
    code: str
    name: str
    indicators: list[str] = field(default_factory=list)


@dataclass
class TopicInfo:
    index: int
    title: str
    # склеенный текст всех смысловых полей темы — для семантического поиска
    searchable_text: str


@dataclass
class DisciplineCard:
    name: str
    year: int
    department: str
    hours_total: int
    has_coursework: bool
    assessment_forms: list[str]      # ["экзамен"], ["зачёт"], ["зачёт с оценкой"]
    control_forms: list[str]         # ["проектная работа", "контрольная работа"]
    semesters: list[int]
    competencies: list[CompetencyInfo]
    topics: list[TopicInfo]

    # быстрые срезы, чтобы не перебирать списки каждый раз
    @property
    def competency_codes(self) -> list[str]:
        return [c.code for c in self.competencies]

    @property
    def competency_names_text(self) -> str:
        """Все наименования компетенций одной строкой — для семантического поиска."""
        return " | ".join(
            f"{c.code}: {c.name}" for c in self.competencies
        )

    @property
    def topic_titles(self) -> list[str]:
        return [t.title for t in self.topics]

    @property
    def topics_searchable_text(self) -> str:
        """Всё содержимое тем одной строкой — для семантического поиска."""
        return " ".join(t.searchable_text for t in self.topics)


# ---------------------------------------------------------------------------
# Парсер JSON → DisciplineCard
# ---------------------------------------------------------------------------

class _Parser:
    """Парсит один JSON-файл РПД в DisciplineCard. Устойчив к отсутствующим полям."""

    def parse(self, raw: dict) -> DisciplineCard:
        ci = raw.get("course_info") or {}
        competencies = self._parse_competencies(raw.get("competencies") or [])
        topics = self._parse_topics(raw.get("topics") or [])
        assessment_forms, control_forms, semesters, has_coursework = (
            self._parse_semesters(ci.get("семестры") or {})
        )

        return DisciplineCard(
            name=raw.get("discipline", ""),
            year=int(raw.get("year", 0)),
            department=ci.get("department", ""),
            hours_total=int(ci.get("трудоемкость_в_часах_всего") or 0),
            has_coursework=has_coursework,
            assessment_forms=assessment_forms,
            control_forms=control_forms,
            semesters=semesters,
            competencies=competencies,
            topics=topics,
        )

    # --- внутренние методы ---

    def _parse_competencies(self, raw_list: list[dict]) -> list[CompetencyInfo]:
        result = []
        for c in raw_list:
            code = (c.get("код_компетенции") or "").strip()
            if not code:
                continue
            indicators = [
                ind.get("наименование_индикатора", "")
                for ind in (c.get("индикаторы_достижения") or [])
                if ind.get("наименование_индикатора")
            ]
            result.append(CompetencyInfo(
                code=code,
                name=(c.get("наименование_компетенции") or "").strip(),
                indicators=indicators,
            ))
        return result

    def _parse_topics(self, raw_list: list[dict]) -> list[TopicInfo]:
        result = []
        for t in raw_list:
            title = (t.get("название_темы") or "").strip()
            if not title:
                continue

            # собираем весь смысловой текст темы
            parts = [
                t.get("содержание_темы") or "",
                (t.get("практические_занятия") or {}).get(
                    "перечень_вопросов_для_обсуждения", ""
                ),
                (t.get("самостоятельная_работа") or {}).get(
                    "перечень_вопросов_для_освоения", ""
                ),
            ]
            searchable = " ".join(p for p in parts if p).strip()

            result.append(TopicInfo(
                index=int(t.get("topic_index", 0)),
                title=title,
                searchable_text=searchable or title,
            ))
        return result

    def _parse_semesters(
        self, semesters_dict: dict
    ) -> tuple[list[str], list[str], list[int], bool]:
        assessment_forms: set[str] = set()
        control_forms: set[str] = set()
        semester_numbers: list[int] = []
        has_coursework = False

        for sem_num, sem_data in semesters_dict.items():
            if not isinstance(sem_data, dict):
                continue
            try:
                semester_numbers.append(int(sem_num))
            except ValueError:
                pass

            af = sem_data.get("вид_промежуточной_аттестации")
            if af:
                assessment_forms.add(af.strip())

            cf = sem_data.get("вид_текущего_контроля")
            if cf:
                control_forms.add(cf.strip())

            if sem_data.get("наличие_курсовой_работы") or sem_data.get(
                "наличие_курсового_проекта"
            ):
                has_coursework = True

        return (
            sorted(assessment_forms),
            sorted(control_forms),
            sorted(semester_numbers),
            has_coursework,
        )


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class DisciplineCatalog:
    """
    In-memory каталог всех дисциплин.

    Использование:
        catalog = DisciplineCatalog("test_data")

        # фильтрация по коду компетенции
        cards = catalog.filter_by_competency_code("DL-1")

        # контекст для LLM
        ctx = catalog.as_llm_context()

        # поиск по дисциплинам для верификации
        exists = catalog.discipline_exists("Рекомендательные системы...")
    """

    def __init__(self, data_dir: str = "test_data"):
        self._cards: dict[str, DisciplineCard] = {}   # name → card
        self._load(Path(data_dir))
        logger.info("DisciplineCatalog: загружено %d дисциплин", len(self._cards))

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("DisciplineCatalog: директория %s не найдена", path)
            return

        parser = _Parser()
        for f in sorted(path.glob("*.json")):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                card = parser.parse(raw)
                if card.name:
                    self._cards[card.name] = card
                else:
                    logger.warning("Пропущен файл %s: нет поля discipline", f.name)
            except Exception as e:
                logger.error("Ошибка при загрузке %s: %s", f.name, e)

    # ------------------------------------------------------------------
    # Базовые свойства
    # ------------------------------------------------------------------

    @property
    def all_cards(self) -> list[DisciplineCard]:
        return list(self._cards.values())

    @property
    def all_names(self) -> list[str]:
        return list(self._cards.keys())

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, name: str) -> Optional[DisciplineCard]:
        return self._cards.get(name)

    def discipline_exists(self, name: str) -> bool:
        return name in self._cards

    # ------------------------------------------------------------------
    # Фильтрация — детерминированная, без LLM
    # ------------------------------------------------------------------

    def filter_by_competency_code(self, code: str) -> list[DisciplineCard]:
        """Точное совпадение кода компетенции (регистронезависимо)."""
        code = code.upper().strip()
        return [c for c in self._cards.values() if code in c.competency_codes]

    def filter_by_assessment_form(self, form: str) -> list[DisciplineCard]:
        """Например: 'экзамен', 'зачёт'."""
        form = form.lower().strip()
        return [
            c for c in self._cards.values()
            if any(af.lower() == form for af in c.assessment_forms)
        ]

    def filter_by_has_coursework(self) -> list[DisciplineCard]:
        return [c for c in self._cards.values() if c.has_coursework]

    def filter_by_department(self, dept: str) -> list[DisciplineCard]:
        dept = dept.upper().strip()
        return [c for c in self._cards.values() if c.department.upper() == dept]

    # ------------------------------------------------------------------
    # Контекст для LLM
    # ------------------------------------------------------------------

    def as_llm_context(self, mode: str = "full") -> str:
        """
        Компактное представление каталога для передачи в LLM.

        mode="compact"  — только названия + коды компетенций + заголовки тем
                          (~50-80 токенов на дисциплину, ~3-4K на 50 дисциплин)

        mode="full"     — компетенции с формулировками + темы с содержимым
                          (~150-250 токенов на дисциплину, ~8-12K на 50 дисциплин)
        """
        if mode == "compact":
            return self._context_compact()
        return self._context_full()

    def _context_compact(self) -> str:
        lines = []
        for i, card in enumerate(self._cards.values(), 1):
            topics_str = "; ".join(card.topic_titles[:5])
            if len(card.topics) > 5:
                topics_str += f" ...ещё {len(card.topics) - 5}"
            lines.append(
                f"{i}. **{card.name}** ({card.year})\n"
                f"   Компетенции: {', '.join(card.competency_codes) or '—'}\n"
                f"   Темы: {topics_str}\n"
                f"   Контроль: {', '.join(card.assessment_forms) or '—'}"
            )
        return "\n\n".join(lines)

    def _context_full(self) -> str:
        lines = []
        for i, card in enumerate(self._cards.values(), 1):
            comp_str = "\n   ".join(
                f"{c.code}: {c.name}" for c in card.competencies
            ) or "—"
            topics_str = "\n   ".join(
                f"Тема {t.index}: {t.title}" for t in card.topics
            ) or "—"
            lines.append(
                f"{i}. **{card.name}** ({card.year}, {card.department})\n"
                f"   Часов: {card.hours_total} | "
                f"Контроль: {', '.join(card.assessment_forms) or '—'}\n"
                f"   Компетенции:\n   {comp_str}\n"
                f"   Темы:\n   {topics_str}"
            )
        return "\n\n".join(lines)

    def cards_as_llm_context(self, cards: list[DisciplineCard], mode: str = "full") -> str:
        """Контекст только для заданного набора карточек (после фильтрации)."""
        tmp = self._cards
        self._cards = {c.name: c for c in cards}
        result = self.as_llm_context(mode)
        self._cards = tmp
        return result

    # ------------------------------------------------------------------
    # Верификация ответа LLM
    # ------------------------------------------------------------------

    def extract_and_verify_disciplines(
        self, answer: str
    ) -> tuple[list[str], list[str]]:
        """
        Находит в тексте ответа упоминания дисциплин из каталога.
        Возвращает (found_valid, hallucinated).

        Алгоритм: ищем точные вхождения известных имён в тексте ответа.
        """
        found: list[str] = []
        hallucinated: list[str] = []

        for name in self._cards:
            if name.lower() in answer.lower():
                found.append(name)

        # Грубый поиск кандидатов — имена в кавычках или после «дисциплина»
        candidates = re.findall(
            r'[«"\'](.*?)[»"\']|(?:дисциплин[аеуы]\s+)([А-ЯA-Z][^\n,;.]+)',
            answer,
            flags=re.IGNORECASE,
        )
        for groups in candidates:
            for candidate in groups:
                candidate = candidate.strip()
                if candidate and candidate not in self._cards and len(candidate) > 5:
                    hallucinated.append(candidate)

        return found, hallucinated

    def verify_competency_answer(
        self, answer: str, expected_code: str
    ) -> tuple[bool, list[str]]:
        """
        Проверяет, что все дисциплины в ответе действительно
        содержат компетенцию с кодом expected_code.
        """
        valid_cards = {c.name for c in self.filter_by_competency_code(expected_code)}
        mentioned, _ = self.extract_and_verify_disciplines(answer)
        wrong = [d for d in mentioned if d not in valid_cards]
        return len(wrong) == 0, wrong

    # ------------------------------------------------------------------
    # Debug / introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        total_topics = sum(len(c.topics) for c in self._cards.values())
        all_codes: set[str] = set()
        for c in self._cards.values():
            all_codes.update(c.competency_codes)
        return (
            f"Дисциплин: {len(self._cards)}\n"
            f"Тем всего: {total_topics}\n"
            f"Уникальных кодов компетенций: {len(all_codes)}\n"
            f"Коды: {', '.join(sorted(all_codes))}"
        )