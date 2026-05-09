from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class QueryType(str, Enum):
    SINGLE_SIMPLE      = "single.simple"
    SINGLE_GLOBAL      = "single.global"
    MULTI_RELATION     = "multi.relation"
    MULTI_COMPARE      = "multi.compare"
    CLARIFY            = "clarify"
    NOT_FOUND          = "not_found"
    IRRELEVANT         = "irrelevant"

    MULTI_GLOBAL       = "multi.global"
    MULTI_GLOBAL_CATALOG            = "multi.global.catalog"
    MULTI_GLOBAL_COMPETENCY_EXACT   = "multi.global.competency_exact"
    MULTI_GLOBAL_COMPETENCY_SEMANTIC = "multi.global.competency_semantic"
    MULTI_GLOBAL_TOPIC              = "multi.global.topic"
    MULTI_GLOBAL_SEMANTIC           = "multi.global.semantic"


@dataclass
class RouteResult:
    query_type:  QueryType
    disciplines: list[str]
    message:     str = ""
    global_entity:      str = "" 


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
    hyde_text: str = ""
    sub_expanded: list[ExpandedQuery] = field(default_factory=list)  

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
    verification_note:        str
    clarification_candidates: list[str] = field(default_factory=list)
    disciplines:              list[str] = field(default_factory=list)

