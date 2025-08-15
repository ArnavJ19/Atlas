# chat_rag_app_debug.py
# Run: streamlit run chat_rag_app_debug.py
# Works with Chroma v0.6+ (auto-detects collections) + FAISS fallback
# Includes: multi-collection retrieval, KG context, and rich diagnostics.

from __future__ import annotations
import os, re, json, pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st
import networkx as nx
from langchain_core.documents import Document

# Embeddings + vectorstores
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma as LCChroma

# LLMs
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
STORE = BASE_DIR / "store"
CHROMA_DIR = STORE / "chromadb"  # matches your screenshot
FAISS_DIR = STORE / "comprehensive_vectorstore"
GRAPH_FILE = STORE / "comprehensive_knowledge_graph.pkl"
DOCS_JSON = STORE / "comprehensive_stocks_data.json"
SUMMARY_JSON = STORE / "ingestion_summary.json"

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chat RAG — Debug", layout="wide")
st.title("💬 Chat RAG — with Diagnostics (Chroma/FAISS + KG)")

with st.sidebar:
    st.subheader("LLM Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI (API)", "Ollama (local)"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Top-K retrieval", 2, 12, 6, 1)
    include_graph = st.checkbox("Include KG neighborhood context", value=True)
    strict_rag = st.checkbox("Forbid no-context answers", value=True)
    st.markdown("---")
    multi_col = st.checkbox("Use ALL Chroma collections (merge results)", value=True)
    st.caption("Set OPENAI_API_KEY for OpenAI. For Ollama, run 'ollama serve' and pull a model (e.g., llama3).")

# ---------- Helpers (NO Streamlit calls here) ----------
def _embedding_name_from_summary(default=DEFAULT_EMBED) -> str:
    try:
        if SUMMARY_JSON.exists():
            return json.load(open(SUMMARY_JSON, "r", encoding="utf-8")).get("embedding_model", default)
    except Exception:
        pass
    return default

@st.cache_resource(show_spinner=True)
def load_embeddings_name():
    return _embedding_name_from_summary()

@st.cache_resource(show_spinner=True)
def load_embeddings(emb_name: str):
    return HuggingFaceEmbeddings(model_name=emb_name)

@st.cache_resource(show_spinner=True)
def load_graph() -> nx.Graph:
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {GRAPH_FILE}")
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.Graph):
        raise TypeError("Loaded object is not a networkx.Graph")
    return G

@st.cache_resource(show_spinner=True)
def load_docs_as_documents() -> List[Document]:
    if not DOCS_JSON.exists():
        return []
    data = json.load(open(DOCS_JSON, "r", encoding="utf-8"))
    return [Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {})) for d in data]

# ---------- Chroma loaders (v0.6+) ----------
def _list_chroma_collections(path: Path) -> List[str]:
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    names = client.list_collections()  # v0.6 returns List[str]
    return names

@st.cache_resource(show_spinner=True)
def load_chroma_vectorstore_for_collection(_emb, collection_name: str):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.get_collection(collection_name)  # ensure exists
    vs = LCChroma(client=client, collection_name=collection_name, embedding_function=_emb,
                  persist_directory=str(CHROMA_DIR))
    # smoke test
    try:
        _ = vs.similarity_search("ping", k=1)
    except Exception:
        pass
    return vs

@st.cache_resource(show_spinner=True)
def load_all_chroma_vectorstores(_emb):
    """Return list of (name, vs) for all non-empty collections."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    names = client.list_collections()
    pairs = []
    for name in names:
        col = client.get_collection(name)
        try:
            if col.count() > 0:
                vs = LCChroma(client=client, collection_name=name, embedding_function=_emb,
                              persist_directory=str(CHROMA_DIR))
                pairs.append((name, vs))
        except Exception:
            continue
    if not pairs:
        raise RuntimeError("No non-empty Chroma collections found.")
    return pairs

@st.cache_resource(show_spinner=True)
def load_faiss_vectorstore(_emb):
    if not FAISS_DIR.exists():
        raise FileNotFoundError(f"FAISS directory not found: {FAISS_DIR}")
    vs = FAISS.load_local(str(FAISS_DIR), _emb, allow_dangerous_deserialization=True)
    try:
        _ = vs.similarity_search("ping", k=1)
    except Exception:
        pass
    return vs

# ---------- KG utilities ----------
def get_company_tickers(G: nx.Graph) -> Set[str]:
    out = set()
    for n, a in G.nodes(data=True):
        t = (a.get("node_type") or a.get("type") or "").lower()
        if t in {"company", "equity", "stock"}:
            out.add(n)
    return out

def neighbors_summary(G: nx.Graph, ticker: str, limit: int = 12) -> str:
    if ticker not in G:
        return ""
    info = G.nodes[ticker]
    label = info.get("name") or ticker
    bits = [f"{label} ({ticker}) neighborhood:"]
    rels = []
    for u, v, e in G.edges(ticker, data=True):
        other = v if u == ticker else u
        other_data = G.nodes[other]
        olabel = other_data.get("ticker") or other_data.get("name") or other
        rel = e.get("relationship") or "related_to"
        rels.append(f"- {rel} → {olabel}")
        if len(rels) >= limit:
            break
    return "\n".join(bits + rels) if rels else "\n".join(bits)

# ---------- Retrieval helpers ----------
def _extract_query_tickers(query: str, G: nx.Graph) -> List[str]:
    tickers = set(re.findall(r"\b[A-Z]{1,5}\b", query))
    ql = query.lower()
    for n, a in G.nodes(data=True):
        t = (a.get("node_type") or a.get("type") or "").lower()
        if t in {"company", "equity", "stock"}:
            name = (a.get("name") or "").lower()
            if name and (name in ql or ql in name):
                tk = a.get("ticker")
                if tk:
                    tickers.add(tk)
    return list(tickers)

def _docs_by_ticker(all_docs: List[Document], wanted: Set[str], k_per: int = 5) -> List[Document]:
    hits = []
    for t in wanted:
        c = 0
        for d in all_docs:
            if (d.metadata or {}).get("ticker") == t:
                hits.append(d); c += 1
                if c >= k_per:
                    break
    return hits

def multi_retrieve(vss: List[Tuple[str, Any]], query: str, k: int = 6) -> List[Tuple[str, Document, float]]:
    """Query multiple Chroma collections and merge results with light de-dup."""
    PREF_ORDER = ["sp500_financials", "sp500_companies", "sp500_trading", "sp500_sectors"]
    vss_sorted = sorted(vss, key=lambda p: PREF_ORDER.index(p[0]) if p[0] in PREF_ORDER else 999)
    hits: List[Tuple[str, Document, float]] = []
    seen = set()
    for name, vs in vss_sorted:
        try:
            for d, score in vs.similarity_search_with_score(query, k=min(4, k)):
                key = ((d.metadata or {}).get("ticker"), d.page_content[:160])
                if key not in seen:
                    hits.append((name, d, float(score)))
                    seen.add(key)
        except Exception:
            pass
        if len(hits) >= k:
            break
    return hits[:k]

def build_context(docs: List[Document], G: nx.Graph, include_graph: bool = True) -> str:
    parts = []
    tickers_seen: Set[str] = set()
    for d in docs:
        md = d.metadata or {}
        t = md.get("ticker")
        src = md.get("source") or md.get("origin") or "source"
        head = f"[{t or 'N/A'} | {src}]"
        parts.append(f"{head}\n{d.page_content}")
        if t:
            tickers_seen.add(t)
    if include_graph and tickers_seen:
        parts.append("\n--- Knowledge Graph Hints ---")
        for t in list(tickers_seen)[:5]:
            desc = neighbors_summary(G, t, limit=8)
            if desc:
                parts.append(desc)
    return "\n\n".join(parts)

def pick_llm(provider: str, temperature: float = 0.0):
    if provider == "Ollama (local)":
        return ChatOllama(model="llama3.2:3b", temperature=temperature)
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

# ---------- Load resources ----------
try:
    emb_name = load_embeddings_name()
    emb = load_embeddings(emb_name)
    G = load_graph()
    all_docs = load_docs_as_documents()
except Exception as e:
    st.error(str(e))
    st.stop()

# ---- Chroma / FAISS setup with diagnostics ----
chroma_ready = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
faiss_ready = FAISS_DIR.exists()

chroma_info = {}
if chroma_ready:
    try:
        names = _list_chroma_collections(CHROMA_DIR)
        chroma_info = {"dir": str(CHROMA_DIR), "collections": names}
    except Exception as e:
        chroma_info = {"dir": str(CHROMA_DIR), "error": str(e)}

st.success(f"Embeddings: **{emb_name}** | KG: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

with st.expander("🧪 Chroma path & collections (v0.6+)", expanded=True):
    st.write({"CHROMA_DIR": str(CHROMA_DIR)})
    st.write(chroma_info or "No chroma folder found.")

# Load vectorstores (multi-collection if enabled)
vss: List[Tuple[str, Any]] = []
backend_label = ""
try:
    if chroma_ready:
        if multi_col:
            vss = load_all_chroma_vectorstores(emb)
            backend_label = " + ".join([n for n, _ in vss])
            st.info(f"Using Chroma collections: {backend_label}")
        else:
            # single best guess: sp500_financials if present else first
            names = chroma_info.get("collections", [])
            prefer = "sp500_financials" if "sp500_financials" in names else (names[0] if names else None)
            if not prefer:
                raise RuntimeError("No Chroma collections found.")
            single_vs = load_chroma_vectorstore_for_collection(emb, prefer)
            vss = [(prefer, single_vs)]
            backend_label = prefer
            st.info(f"Using Chroma collection: {prefer}")
    elif faiss_ready:
        vs = load_faiss_vectorstore(emb)
        vss = [("faiss", vs)]
        backend_label = "faiss"
        st.info("Using FAISS vector store.")
    else:
        raise FileNotFoundError(
            "No vector store found. Expected either:\n"
            f" • Chroma at: {CHROMA_DIR}\n"
            f" • FAISS  at: {FAISS_DIR}\n"
            "Re-run ingestion or update paths at the top of this file."
        )
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------- Retrieval quick test ----------
with st.expander("⚙️ Retrieval quick test"):
    test_q = st.text_input("Test query (try 'Adobe ADBE financials')", "Adobe ADBE financials")
    if st.button("Run retrieval test"):
        try:
            rows = []
            if backend_label.lower() == "faiss":
                try:
                    hits = vss[0][1].similarity_search_with_score(test_q, k=6)
                    for d, s in hits:
                        rows.append({"store": "faiss", "score": float(s),
                                     "ticker": (d.metadata or {}).get("ticker"),
                                     "sector": (d.metadata or {}).get("sector"),
                                     "source": (d.metadata or {}).get("source"),
                                     "preview": d.page_content[:160]})
                except Exception as e:
                    st.write(f"FAISS search error: {e}")
            else:
                # multi-collection
                merged = multi_retrieve(vss, test_q, k=6)
                for name, d, score in merged:
                    rows.append({"store": name, "score": score,
                                 "ticker": (d.metadata or {}).get("ticker"),
                                 "sector": (d.metadata or {}).get("sector"),
                                 "source": (d.metadata or {}).get("source"),
                                 "preview": d.page_content[:160]})
            st.write(rows)
        except Exception as e:
            st.write(f"debug error: {e}")

# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about companies/sectors you ingested (e.g., 'Tell me about ADBE financials')...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve
    try:
        # Build doc list depending on backend
        if backend_label.lower() == "faiss":
            docs_scored = vss[0][1].similarity_search_with_score(user_q, k=top_k)
            docs = [d for d, _ in docs_scored]
        else:
            docs = [d for _, d, _ in multi_retrieve(vss, user_q, k=top_k)]
        # Boost with ticker-directed docs from JSON (if present)
        q_tickers = set(_extract_query_tickers(user_q, G))
        direct = _docs_by_ticker(all_docs, q_tickers, k_per=5) if q_tickers else []
        # Merge de-duplicated
        seen = set(); merged = []
        for d in direct + docs:
            key = ((d.metadata or {}).get("ticker"), d.page_content[:160])
            if key not in seen:
                merged.append(d); seen.add(key)
        context = build_context(merged, G, include_graph=include_graph)
    except Exception as e:
        context = ""
        with st.chat_message("assistant"):
            st.error(f"Retrieval error: {e}")

    # LLM
    llm = pick_llm(provider, temperature=temperature)
    try:
        answer = answer_with_llm(llm, context, user_q, strict=strict_rag)
    except Exception as e:
        answer = f"LLM error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- Last context viewer ----------
with st.expander("🔎 Last retrieved context"):
    try:
        last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user:
            if backend_label.lower() == "faiss":
                docs_scored = vss[0][1].similarity_search_with_score(last_user, k=top_k)
                docs_dbg = [d for d, _ in docs_scored]
            else:
                docs_dbg = [d for _, d, _ in multi_retrieve(vss, last_user, k=top_k)]
            for i, d in enumerate(docs_dbg, 1):
                md = d.metadata or {}
                st.markdown(f"**Doc {i}** — Ticker: `{md.get('ticker')}` | Sector: `{md.get('sector')}` | Source: `{md.get('source')}`")
                st.code((d.page_content or "")[:1200] + ("..." if len(d.page_content or "") > 1200 else ""))
        else:
            st.write("No user query yet.")
    except Exception as e:
        st.write(f"Context display error: {e}")
