# chat_rag_ui.py
# Run: streamlit run chat_rag_ui.py

from __future__ import annotations
import streamlit as st
from pathlib import Path
from rag_core_engine import RAGEngine, extract_query_tickers, docs_by_ticker

# ---- Streamlit config ----
st.set_page_config(page_title="Chat RAG — UI", layout="wide")
st.title("💬 Chat RAG — UI (Chroma/FAISS + KG)")

# ---- Sidebar controls ----
with st.sidebar:
    st.subheader("RAG Settings")
    use_all_chroma = st.checkbox("Use ALL Chroma collections", value=True)
    prefer_single = st.text_input("Or prefer single collection (leave blank to auto)", value="")
    top_k = st.slider("Top-K retrieval", 2, 12, 6, 1)
    include_graph = st.checkbox("Include KG neighborhood context", value=True)
    strict_rag = st.checkbox("Forbid no-context answers", value=True)
    st.markdown("---")
    st.subheader("LLM Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI (API)", "Ollama (local)"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("Set OPENAI_API_KEY for OpenAI. For Ollama, run 'ollama serve' and pull a model (e.g., llama3).")

# ---- Cached engine factory ----
@st.cache_resource(show_spinner=True)
def get_engine(use_all: bool, prefer_single: str):
    prefer = prefer_single.strip() or None
    eng = RAGEngine(use_all_chroma=use_all, prefer_single_collection=prefer)
    return eng

eng = get_engine(use_all_chroma, prefer_single)

# ---- Display backend info & diagnostics ----
st.success(f"Embeddings: **{eng.embedding_name}** | KG: {eng.G.number_of_nodes()} nodes / {eng.G.number_of_edges()} edges")
with st.expander("🧪 Vector store diagnostics", expanded=True):
    st.write(eng.debug_collections())

# ---- LLM picker (kept here to avoid Streamlit import inside engine) ----
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def pick_llm(provider: str, temperature: float = 0.0):
    if provider == "Ollama (local)":
        return ChatOllama(model="llama3", temperature=temperature)
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

def answer_with_llm(llm, context: str, question: str, strict: bool = True) -> str:
    if strict and not context.strip():
        return ("I couldn’t find relevant documents in your corpus for this question. "
                "Try adding the company/ticker explicitly (e.g., “ADBE financials 2023”).")
    system = (
        "You are a precise financial analyst. Use the given context faithfully. "
        "Prioritize factual accuracy, cite tickers inline when helpful, and avoid speculation."
    )
    prompt = (
        f"{system}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly. If the answer is not in the context, say so briefly and suggest what data would help."
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", resp)

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- Chat input & flow ----
user_q = st.chat_input("Ask about companies/sectors you ingested (e.g., 'Tell me about ADBE financials')...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve from engine
    docs = eng.retrieve(user_q, k=top_k)

    # Boost with direct ticker docs from JSON
    q_tickers = set(extract_query_tickers(user_q, eng.G))
    direct = docs_by_ticker(eng.docs, q_tickers, k_per=5) if q_tickers else []
    # Merge (dedupe)
    seen = set(); merged = []
    for d in direct + docs:
        key = ((d.metadata or {}).get("ticker"), d.page_content[:160])
        if key not in seen:
            merged.append(d); seen.add(key)

    # Build context & answer
    context = eng.build_context(merged, include_graph=include_graph)
    llm = pick_llm(provider, temperature=temperature)
    answer = answer_with_llm(llm, context, user_q, strict=strict_rag)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- Last context viewer ----
with st.expander("🔎 Last retrieved context"):
    try:
        last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user:
            docs_dbg = eng.retrieve(last_user, k=top_k)
            for i, d in enumerate(docs_dbg, 1):
                md = d.metadata or {}
                st.markdown(f"**Doc {i}** — Ticker: `{md.get('ticker')}` | Sector: `{md.get('sector')}` | Source: `{md.get('source')}`")
                st.code((d.page_content or "")[:1200] + ("..." if len(d.page_content or "") > 1200 else ""))
        else:
            st.write("No user query yet.")
    except Exception as e:
        st.write(f"Context display error: {e}")
