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
.block-container { padding-top: 2rem; }

/* Карточка ответа */
.answer-card {
    background: #475569;
    border-left: 4px solid #4f6df5;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin-top: .5rem;
    font-size: 1.02rem;
    line-height: 1.7;
}

/* Чип типа запроса */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .04em;
    margin-right: 6px;
}
.badge-verified   { background:#d1fae5; color:#065f46; }
.badge-unverified { background:#fee2e2; color:#991b1b; }
.badge-clarify    { background:#fef3c7; color:#92400e; }
.badge-type       { background:#e0e7ff; color:#3730a3; }

/* Строка блока */
.chunk-row {
    display: flex;
    align-items: baseline;
    gap: .6rem;
    padding: 5px 0;
    border-bottom: 1px solid #eee;
    font-size: .88rem;
}
.chunk-score {
    font-family: monospace;
    font-size: .82rem;
    color: #6b7280;
    min-width: 52px;
}
.chunk-disc  { font-weight: 600; color: #1e293b; }
.chunk-block { color: #475569; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Сайдбар — параметры
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Параметры")

    qdrant_url  = st.text_input("Qdrant URL", value="http://localhost:6333")
    collection  = st.text_input("Коллекция",  value="discipline_chunks")

    st.divider()

    top_k_single   = st.slider("top-k (поиск)",    1, 20, 6)
    reranker_top_k = st.slider("top-k (реранкер)", 1, 20, 6)

    st.divider()

    no_reranker   = st.checkbox("Отключить реранкер")
    no_hyde       = st.checkbox("Отключить HyDE")
    no_paraphrase = st.checkbox("Отключить перефразирование")

    st.divider()
    verbose = st.checkbox("Показывать блоки", value=True)


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

st.title("🎓 RAG · Рабочие программы дисциплин")
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
    elapsed  = entry["elapsed"]

    from rag.domain.models import QueryType

    # Заголовок с бейджами
    badges = f'<span class="badge badge-type">{response.query_type}</span>'
    if response.query_type == QueryType.CLARIFY:
        badges += '<span class="badge badge-clarify">уточнение</span>'
    elif response.is_verified:
        badges += '<span class="badge badge-verified">✓ верифицирован</span>'
    else:
        badges += '<span class="badge badge-unverified">✗ не верифицирован</span>'

    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        st.markdown(badges, unsafe_allow_html=True)

        if response.query_type == QueryType.CLARIFY:
            st.warning(response.answer)
            if response.clarification_candidates:
                st.write("**Варианты:**")
                for i, c in enumerate(response.clarification_candidates, 1):
                    st.write(f"&nbsp;&nbsp;{i}. {c}")
        else:
            st.markdown(
                f'<div class="answer-card">{response.answer}</div>',
                unsafe_allow_html=True,
            )
            if response.verification_note:
                st.caption(f"💬 {response.verification_note}")

            if verbose and response.chunks_used:
                with st.expander(f"📄 Блоки ({len(response.chunks_used)})", expanded=False):
                    for c in response.chunks_used:
                        st.markdown(
                            f'<div class="chunk-row">'
                            f'<span class="chunk-score">{c.score:+.3f}</span>'
                            f'<span class="chunk-disc">{c.discipline}</span>'
                            f'<span class="chunk-block">/ {c.block_name}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.caption(f"⏱ {elapsed:.2f} с")


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
