"""
Модели данных RAG-пайплайна.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    SINGLE_SIMPLE  = "single.simple"   # точечный факт по одной дисциплине
    SINGLE_GLOBAL  = "single.global"   # развёрнутый вопрос по одной дисциплине
    MULTI_RELATION = "multi.relation"  # сравнение конкретных дисциплин
    MULTI_GLOBAL   = "multi.global"    # запрос по всему корпусу


@dataclass
class RouteResult:
    query_type:    QueryType
    disciplines:   list[str]   # названия дисциплин, извлечённые из запроса
    is_time_query: bool         # вопрос о часах/ЗЕ/семестрах/форме контроля
    reasoning:     str = ""


@dataclass
class ExpandedQuery:
    original:    str
    paraphrases: list[str]        # альтернативные формулировки
    sub_queries: list[str]        # подзапросы (для multi.relation)
    disciplines: list[str]        # разрешённые названия дисциплин из индекса
    query_type:  QueryType
    is_time_query: bool = False


@dataclass
class RetrievedChunk:
    block_id:   str
    block_type: str
    block_name: str
    text:       str
    summary:    str
    discipline: str
    score:      float
    metadata:   dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    is_valid:  bool
    note:      str = ""
    retry:     bool = False   # True — стоит повторить поиск


@dataclass
class RAGResponse:
    answer:           str
    query_type:       QueryType
    is_verified:      bool
    chunks_used:      list[RetrievedChunk]
    verification_note: str = ""
