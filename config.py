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
"""
import os
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_BASE_URL     = os.getenv("LLM_BASE_URL",     "https://rus-gpt.com/api/v1")
LLM_API_KEY      = os.getenv("LLM_API_KEY",      "")

# Лёгкая быстрая модель — JSON-классификация, короткий контекст
# Используется: Router, QueryExpander, TimeFilter
LLM_MODEL_FAST   = os.getenv("LLM_MODEL_FAST",   "deepseek/deepseek-v3.2")

# Модель верификации — сопоставляет ответ с контекстом, детектирует галлюцинации
# Используется: VerificationModule
LLM_MODEL_VERIFY = os.getenv("LLM_MODEL_VERIFY", "deepseek/deepseek-v3.2")

# Основная модель — синтез ответа из большого контекста
# Используется: GenerationModule
LLM_MODEL_MAIN   = os.getenv("LLM_MODEL_MAIN",   "deepseek/deepseek-v3.2")

# Модель для индексации — суммаризация блоков (запускается офлайн)
# Используется: Summarizer в indexer.py
LLM_MODEL_IDX    = os.getenv("LLM_MODEL_IDX",    "deepseek/deepseek-v3.2")
LLM_MODEL_IDX = "deepseek/deepseek-v3.2"
# ---------------------------------------------------------------------------
# Max tokens по ролям
# ---------------------------------------------------------------------------
LLM_MAX_TOKENS_FAST   = 300
LLM_MAX_TOKENS_VERIFY = 200
LLM_MAX_TOKENS_MAIN   = 1200
LLM_MAX_TOKENS_IDX    = 400
LLM_MAX_TOKENS_TIME   = 150

LLM_RETRY_DELAY = 2   # базовая пауза между попытками (умножается на номер попытки)
LLM_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL        = os.getenv("QDRANT_URL",        "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "discipline_chunks")

# Названия именованных векторов в коллекции
VEC_TEXT    = "text"
VEC_SUMMARY = "summary"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "paraphrase-multilingual-mpnet-base-v2")
EMBED_DIM        = 768
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
