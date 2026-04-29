"""
Централизованная конфигурация RAG-пайплайна и индексатора.

Переменные окружения:
    LLM_BASE_URL      — базовый URL OpenAI-совместимого API
    LLM_API_KEY       — API-ключ
    LLM_MODEL_FAST    — лёгкая модель (router, expander, time_filter)
    LLM_MODEL_VERIFY  — модель верификации
    LLM_MODEL_MAIN    — основная модель (generation)
    LLM_MODEL_IDX     — модель для индексации (summarizer)
    QDRANT_URL        — URL Qdrant (или ":memory:" для тестов)
    QDRANT_COLLECTION — имя коллекции
    EMBED_MODEL       — sentence-transformers модель для эмбедингов
    EMBED_BATCH_SIZE  — размер батча при кодировании
    RERANKER_MODEL    — cross-encoder модель для реранкинга
    RERANKER_TOP_K    — сколько чанков оставить после реранкинга
"""
import os
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_BASE_URL     = os.getenv("LLM_BASE_URL",     "https://api.openai.com/v1")
LLM_API_KEY      = os.getenv("LLM_API_KEY",      "")

LLM_MODEL_FAST   = os.getenv("LLM_MODEL_FAST",   "qwen3-30b-a3b-instruct-2507")
LLM_MODEL_VERIFY = os.getenv("LLM_MODEL_VERIFY", "deepseek-v3.2")
LLM_MODEL_MAIN   = os.getenv("LLM_MODEL_MAIN",   "deepseek-v3.2")
LLM_MODEL_IDX    = os.getenv("LLM_MODEL_IDX",    "deepseek-v3.2")
# Модель-судья для оценки качества — должна отличаться от generation-модели
# Рекомендуется: gpt-4o, claude-sonnet-4.5, qwen3.5-397b-a17b
LLM_MODEL_EVAL   = os.getenv("LLM_MODEL_EVAL",   "qwen/qwen3.5-397b-a17b")

LLM_MAX_TOKENS_FAST   = 300
LLM_MAX_TOKENS_VERIFY = 400
LLM_MAX_TOKENS_MAIN   = 1200
LLM_MAX_TOKENS_IDX    = 400
#LLM_MAX_TOKENS_TIME   = 150
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
# Embeddings
# ---------------------------------------------------------------------------
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "paraphrase-multilingual-mpnet-base-v2")
EMBED_DIM        = 768
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
# Multilingual cross-encoder, поддерживает русский язык
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "6"))


RPD_NAMES = ['алгоритмы и структуры данных в языке python', 'технологии обработки данных', 'технологии обработки больших данных', 'рекомендательные системы и коллаборативная фильтрация', 'прикладная теория графов', 'обработка текстов на естественных языках', 'машинное зрение', 'глубокое обучение', 'фундаментальные и генеративные модели', 'семантические технологии', 'технологии и алгоритмы анализа сетевых моделей', 'машинное обучение в семантическом и сетевом анализе', 'прикладные задачи машинного обучения']
