"""
Streamlit-интерфейс RAG-системы.
Запуск: streamlit run app.py
"""

import time
import streamlit as st

# ---------------------------------------------------------------------------
# Конфиг страницы
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG · Рабочие программы",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Стили
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Убираем лишние отступы сверху */
.block-container { padding-top: 1rem; }

/* Карточка ответа - красивый стиль */
.answer-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-left: 5px solid #4f46e5;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #1f2937;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.answer-card ul {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.answer-card li {
    margin: 0.3rem 0;
}

.answer-card ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.answer-card ol li {
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Сайдбар — параметры
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Параметры")

    qdrant_url  = "http://localhost:6333"
    collection  = "discipline_chunks"
    top_k_single   = 6
    reranker_top_k = 6

    st.divider()

    no_hyde       = st.checkbox("Отключить перефразирование запроса", value=False)
    no_paraphrase = st.checkbox("Отключить расширение запроса", value=False)
    no_reranker   = st.checkbox("Отключить реранкер", value=False)

    st.divider()
    st.caption("⚡ RAG система для рабочих программ")


# ---------------------------------------------------------------------------
# Кеш пайплайна — не пересоздавать при каждом запросе
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline(qdrant_url, collection, no_reranker, no_hyde, no_paraphrase,
                 top_k_single, reranker_top_k):
    import rag.retrieval.retrieval as ret
    import rag.retrieval.expander  as exp
    import rag.config.config       as cfg
    from rag.pipeline.pipeline import RAGPipeline

    ret.TOP_K_SINGLE   = top_k_single
    cfg.RERANKER_TOP_K = reranker_top_k

    if no_hyde:
        exp.HYDE_QUERY_TYPES = set()

    if no_paraphrase:
        orig = exp.QueryExpander.expand
        def _no_para(self, q, r, rs):
            res = orig(self, q, r, rs)
            res.paraphrases = []
            return res
        exp.QueryExpander.expand = _no_para

    pipeline = RAGPipeline(qdrant_url=qdrant_url, collection=collection)

    if no_reranker:
        class _Noop:
            def rerank(self, q, c): return c
        pipeline._reranker = _Noop()

    return pipeline


# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------

st.title("Интеллектуальная система поддержки принятия решений в управлении образовательными программами")
st.caption("Поиск по учебным планам и программам дисциплин")

# ---------------------------------------------------------------------------
# История в session_state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []      # list of dicts
if "pending_clarify" not in st.session_state:
    st.session_state.pending_clarify = None   # ждём уточнения


# ---------------------------------------------------------------------------
# Вспомогательный рендер одного ответа
# ---------------------------------------------------------------------------

def render_response(entry: dict) -> None:
    """Рисует карточку одного ответа из истории."""
    q        = entry["query"]
    response = entry["response"]

    from rag.domain.models import QueryType

    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        if response.query_type == QueryType.CLARIFY:
            st.warning(response.answer)
            if response.clarification_candidates:
                st.write("**Варианты:**")
                for i, c in enumerate(response.clarification_candidates, 1):
                    st.write(f"&nbsp;&nbsp;{i}. {c}")
        else:
            # Форматируем ответ с поддержкой \n и других символов
            formatted_answer = response.answer.replace("\\n", "\n")
            st.markdown(
                f'<div class="answer-card">{formatted_answer}</div>',
                unsafe_allow_html=True,
            )



# ---------------------------------------------------------------------------
# Рендер истории
# ---------------------------------------------------------------------------

for entry in st.session_state.history:
    render_response(entry)


# ---------------------------------------------------------------------------
# Если ждём уточнения — показываем кнопки-варианты
# ---------------------------------------------------------------------------

if st.session_state.pending_clarify:
    pc = st.session_state.pending_clarify
    candidates = pc["candidates"]
    original   = pc["original_query"]

    st.info("Уточните запрос, выбрав вариант или введя свой:")

    cols = st.columns(min(len(candidates), 4))
    for i, (col, cand) in enumerate(zip(cols, candidates)):
        if col.button(cand, key=f"clarify_{i}"):
            st.session_state.pending_clarify = None
            st.session_state._run_query = f"{original} — {cand}"
            st.rerun()

    custom = st.text_input("Или введите уточнение вручную:", key="custom_clarify")
    if st.button("Отправить уточнение") and custom:
        st.session_state.pending_clarify = None
        st.session_state._run_query = f"{original} — {custom}"
        st.rerun()


# ---------------------------------------------------------------------------
# Поле ввода
# ---------------------------------------------------------------------------

query = st.chat_input("Задайте вопрос по рабочим программам…")

# Если сработало уточнение — берём из session_state
if not query and "_run_query" in st.session_state:
    query = st.session_state.pop("_run_query")


# ---------------------------------------------------------------------------
# Обработка запроса
# ---------------------------------------------------------------------------

if query:
    from rag.domain.models import QueryType

    with st.spinner("Ищу ответ…"):
        try:
            pipeline = get_pipeline(
                qdrant_url, collection,
                no_reranker, no_hyde, no_paraphrase,
                top_k_single, reranker_top_k,
            )
            t0       = time.perf_counter()
            response = pipeline.ask(query)
            elapsed  = time.perf_counter() - t0
        except Exception as e:
            st.error(f"Ошибка пайплайна: {e}")
            st.stop()

    entry = {"query": query, "response": response, "elapsed": elapsed}
    st.session_state.history.append(entry)

    # Если нужно уточнение — запоминаем кандидатов
    if response.query_type == QueryType.CLARIFY and response.clarification_candidates:
        st.session_state.pending_clarify = {
            "original_query": query,
            "candidates": response.clarification_candidates,
        }

    st.rerun()


# ---------------------------------------------------------------------------
# Кнопка очистки истории
# ---------------------------------------------------------------------------

if st.session_state.history:
    st.divider()
    if st.button("🗑 Очистить историю"):
        st.session_state.history = []
        st.session_state.pending_clarify = None
        st.rerun()
