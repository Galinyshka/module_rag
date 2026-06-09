import os
from dotenv import load_dotenv
import json
import pathlib
import re

load_dotenv()
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_BASE_URL     = os.getenv("LLM_BASE_URL",     "https://rus-gpt.com/api/v1")
LLM_API_KEY      = os.getenv("LLM_API_KEY",      "")
LLM_API_KEY_ROUTER = os.getenv("LLM_API_KEY_ROUTER", "")

LLM_MODEL_FAST   = os.getenv("LLM_MODEL_FAST",   "qwen/qwen3-30b-a3b-instruct-2507")
LLM_MODEL_VERIFY = os.getenv("LLM_MODEL_VERIFY", "deepseek/deepseek-v3.2")
LLM_MODEL_MAIN   = os.getenv("LLM_MODEL_MAIN",   "deepseek/deepseek-v3.2")
LLM_MODEL_IDX    = os.getenv("LLM_MODEL_IDX",    "deepseek/deepseek-v3.2")
# Модель-судья для оценки качества — должна отличаться от generation-модели
# Рекомендуется: gpt-4o, claude-sonnet-4.5, qwen3.5-397b-a17b
LLM_MODEL_EVAL   = os.getenv("LLM_MODEL_EVAL",   "qwen/qwen3.5-397b-a17b")

LLM_MAX_TOKENS_ROUTER   = 300
LLM_MAX_TOKENS_VERIFY = 1000
LLM_MAX_TOKENS_MAIN   = 10000
LLM_MAX_TOKENS_IDX    = 400
LLM_MAX_TOKENS_HYDE   = 200
LLM_MAX_TOKENS_EVAL   = 300   # оценка по одной метрике: score + rationale

LLM_RETRY_DELAY = 2
LLM_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL        = os.getenv("QDRANT_URL",        "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "discipline_chunks")

VEC_QUESTIONS  = "questions_vec"   
VEC_SUMMARY    = "summary_vec"   
QUESTIONS_COUNT = 3        

# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "paraphrase-multilingual-mpnet-base-v2")
EMBED_DIM        = 768
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
# оригинальные названия дисциплин
RPD_NAMES_PATH = 'src/rag/utils/rpd_names.json'
if RPD_NAMES_PATH:
    with open(RPD_NAMES_PATH, "r", encoding="utf-8") as f:
        RPD_NAMES = json.load(f)
else:
    RPD_NAMES = []

# настройки fuzzy поиска
FUZZY_THRESHOLD = 50   # минимальный score для попадания в кандидаты
FUZZY_TOP_K     = 5    # максимум кандидатов, передаваемых в LLM

# ---------------------------------------------------------------------------
# Expander
# ---------------------------------------------------------------------------
PARAPHRASES_COUNT = 3 # Количество перефразировок запроса

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_SINGLE          = 6 # сколько чанков брать для single запросов
TOP_K_GLOBAL = 150   # 3 чанка × 50 дисциплин с запасом
# RRF prefetch размер — сколько кандидатов берём из каждого вектора
# перед слиянием. Должен быть >= итогового top_k.
RRF_PREFETCH_K = 50

ALL_BLOCKS = [
    "course_info", "topics", "competencies", "topic", "competency",
    "self_study_resources", "self_study_section",
    "assessment_fund", "literature",
    "online_resources", "other_sections", "other_section",
] # полный список типов блоков для сортировки полного документа


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
# Multilingual cross-encoder, поддерживает русский язык
#RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_K = 5 # сколько чанков оставить после реранкинга
#RERANKER_TOP_K_BALANCE = 5 # для multi запросов с балансировкой: сколько чанков оставить после реранкинга, гарантируя представительство каждой дисциплины (если хватает релевантных кандидатов)


#OP_K_STAGE1          = 30 # сколько чанков брать на первом этапе для multi запросов, до реранкинга
#TOP_K_PER_DISC        = 8 # для multi запросов: сколько чанков брать с каждой дисциплины, до реранкинга
#MAX_DISCIPLINES_MULTI = 10  # для multi запросов: максимальное количество дисциплин, которые будут представлены в результатах (по количеству релевантных чанков)    
#OVERVIEW_BLOCKS       = ["course_info", "topics", "competencies"] 


