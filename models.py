from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class QueryType(str, Enum):
    SINGLE_SIMPLE  = "single.simple"
    SINGLE_GLOBAL  = "single.global"
    MULTI_RELATION = "multi.relation"
    MULTI_GLOBAL   = "multi.global"
    NOT_FOUND      = "not_found"      
    IRRELEVANT     = "irrelevant"  
    CLARIFY        = "clarify"  


@dataclass
class RouteResult:
    query_type:  QueryType
    disciplines: list[str]
    message:   str = ""


@dataclass
class ExpandedQuery:
    original:    str
    paraphrases: list[str]
    sub_queries: list[str]
    disciplines: list[str]
    query_type:  QueryType
    hyde_text:   str = ""    

@dataclass
class ExpandedQuery:
    original: str
    paraphrases: list[str]
    sub_queries: list[str]
    disciplines: list[str]
    query_type: QueryType
    hyde_text: str
    sub_queries_expanded: list[dict[str, Any]] = field(default_factory=list)

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
    is_valid: bool
    note:     str  = ""
    retry:    bool = False


@dataclass
class RAGResponse:
    answer:                   str
    query_type:               QueryType
    is_verified:              bool
    chunks_used:              list
    fact_extracted:           bool
    verification_note:        str
    clarification_candidates: list[str] = field(default_factory=list)
    disciplines:              list[str] = field(default_factory=list)

