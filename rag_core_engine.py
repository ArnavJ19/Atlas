# rag_core_engine.py
from __future__ import annotations
import json, pickle, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import networkx as nx
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma as LCChroma

# ---- Defaults (override via constructor if you want) ----
BASE_DIR = Path(__file__).parent
DEFAULT_STORE = BASE_DIR / "store"
DEFAULT_CHROMA_DIR = DEFAULT_STORE / "chromadb"                 # matches your project
DEFAULT_FAISS_DIR = DEFAULT_STORE / "comprehensive_vectorstore"
DEFAULT_GRAPH_FILE = DEFAULT_STORE / "comprehensive_knowledge_graph.pkl"
DEFAULT_DOCS_JSON = DEFAULT_STORE / "comprehensive_stocks_data.json"
DEFAULT_SUMMARY_JSON = DEFAULT_STORE / "ingestion_summary.json"
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Utilities (no Streamlit here) ----
def embedding_name_from_summary(summary_json: Path, default=DEFAULT_EMBED) -> str:
    if summary_json.exists():
        try:
            return json.load(open(summary_json, "r", encoding="utf-8")).get("embedding_model", default)
        except Exception:
            pass
    return default

def make_embeddings(name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=name)

def load_graph(graph_file: Path) -> nx.Graph:
    if not graph_file.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {graph_file}")
    G = pickle.load(open(graph_file, "rb"))
    if not isinstance(G, nx.Graph):
        raise TypeError("Loaded object is not a networkx.Graph")
    return G

def load_docs_json(docs_json: Path) -> List[Document]:
    if not docs_json.exists():
        return []
    data = json.load(open(docs_json, "r", encoding="utf-8"))
    return [Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {})) for d in data]

# -------- Chroma v0.6+ helpers --------
def chroma_list_collections(chroma_dir: Path) -> List[str]:
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.list_collections()  # List[str]

def build_chroma_vectorstore(emb: HuggingFaceEmbeddings, chroma_dir: Path, collection_name: str):
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.get_collection(collection_name)  # ensure exists
    return LCChroma(client=client, collection_name=collection_name, embedding_function=emb,
                    persist_directory=str(chroma_dir))

def build_all_chroma_vectorstores(emb: HuggingFaceEmbeddings, chroma_dir: Path) -> List[Tuple[str, Any]]:
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    names = client.list_collections()
    pairs = []
    for name in names:
        col = client.get_collection(name)
        try:
            if col.count() > 0:
                vs = LCChroma(client=client, collection_name=name, embedding_function=emb,
                              persist_directory=str(chroma_dir))
                pairs.append((name, vs))
        except Exception:
            continue
    return pairs  # may be []

def build_faiss_vectorstore(emb: HuggingFaceEmbeddings, faiss_dir: Path):
    if not faiss_dir.exists():
        return None
    return FAISS.load_local(str(faiss_dir), emb, allow_dangerous_deserialization=True)

# -------- Retrieval helpers (engine-agnostic) --------
PREF_ORDER = ["sp500_financials", "sp500_companies", "sp500_trading", "sp500_sectors"]

def extract_query_tickers(query: str, G: nx.Graph) -> List[str]:
    tickers = set(re.findall(r"\b[A-Z]{1,5}\b", query))
    ql = query.lower()
    for n, a in G.nodes(data=True):
        t = (a.get("node_type") or a.get("type") or "").lower()
        if t in {"company","equity","stock"}:
            name = (a.get("name") or "").lower()
            if name and (name in ql or ql in name):
                tk = a.get("ticker")
                if tk: tickers.add(tk)
    return list(tickers)

def docs_by_ticker(all_docs: List[Document], wanted: Set[str], k_per: int = 5) -> List[Document]:
    hits = []
    for t in wanted:
        c = 0
        for d in all_docs:
            if (d.metadata or {}).get("ticker") == t:
                hits.append(d); c += 1
                if c >= k_per: break
    return hits

def neighbors_summary(G: nx.Graph, ticker: str, limit: int = 12) -> str:
    if ticker not in G: return ""
    info = G.nodes[ticker]
    label = info.get("name") or ticker
    lines = [f"{label} ({ticker}) neighborhood:"]
    for u, v, e in G.edges(ticker, data=True):
        other = v if u == ticker else u
        other_data = G.nodes[other]
        olabel = other_data.get("ticker") or other_data.get("name") or other
        rel = e.get("relationship") or "related_to"
        lines.append(f"- {rel} → {olabel}")
        if len(lines) - 1 >= limit: break
    return "\n".join(lines)

def build_context(docs: List[Document], G: nx.Graph, include_graph: bool = True) -> str:
    parts = []
    tickers_seen: Set[str] = set()
    for d in docs:
        md = d.metadata or {}
        t = md.get("ticker")
        src = md.get("source") or md.get("origin") or "source"
        parts.append(f"[{t or 'N/A'} | {src}]\n{d.page_content}")
        if t: tickers_seen.add(t)
    if include_graph and tickers_seen:
        parts.append("\n--- Knowledge Graph Hints ---")
        for t in list(tickers_seen)[:5]:
            desc = neighbors_summary(G, t, limit=8)
            if desc: parts.append(desc)
    return "\n\n".join(parts)

def multi_collection_retrieve(vss: List[Tuple[str, Any]], query: str, k: int = 6) -> List[Tuple[str, Document, float]]:
    """Query multiple vectorstores and merge results with light de-dup, preferring financials/companies."""
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
        if len(hits) >= k: break
    return hits[:k]

# -------- Engine --------
class RAGEngine:
    """
    Holds embeddings, vectorstores (Chroma collections and/or FAISS), KG, and documents.
    Provides retrieve() and answer() utilities. No Streamlit imports here.
    """
    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        faiss_dir: Path = DEFAULT_FAISS_DIR,
        graph_file: Path = DEFAULT_GRAPH_FILE,
        docs_json: Path = DEFAULT_DOCS_JSON,
        summary_json: Path = DEFAULT_SUMMARY_JSON,
        embedding_name: Optional[str] = None,
        use_all_chroma: bool = True,
        prefer_single_collection: Optional[str] = None,
    ):
        self.store_dir = store_dir
        self.chroma_dir = chroma_dir
        self.faiss_dir = faiss_dir
        self.graph_file = graph_file
        self.docs_json = docs_json
        self.summary_json = summary_json

        self.embedding_name = embedding_name or embedding_name_from_summary(summary_json)
        self.emb = make_embeddings(self.embedding_name)
        self.G = load_graph(graph_file)
        self.docs = load_docs_json(docs_json)

        self.vss: List[Tuple[str, Any]] = []  # list of (name, vectorstore)
        self.backend_label = ""

        # Try Chroma first
        self._init_vectorstores(use_all_chroma, prefer_single_collection)

    def _init_vectorstores(self, use_all: bool, prefer_single: Optional[str]):
        chroma_ok = self.chroma_dir.exists() and any(self.chroma_dir.iterdir())
        if chroma_ok:
            pairs = []
            if use_all:
                pairs = build_all_chroma_vectorstores(self.emb, self.chroma_dir)
            else:
                names = chroma_list_collections(self.chroma_dir)
                name = prefer_single or ("sp500_financials" if "sp500_financials" in names else (names[0] if names else None))
                if name:
                    vs = build_chroma_vectorstore(self.emb, self.chroma_dir, name)
                    pairs = [(name, vs)]
            if pairs:
                self.vss = pairs
                self.backend_label = " + ".join([n for n, _ in pairs])
                return

        # FAISS fallback
        vs_faiss = build_faiss_vectorstore(self.emb, self.faiss_dir)
        if vs_faiss:
            self.vss = [("faiss", vs_faiss)]
            self.backend_label = "faiss"
            return

        raise FileNotFoundError(
            "No vector store found. Expected either:\n"
            f" • Chroma at: {self.chroma_dir}\n"
            f" • FAISS  at: {self.faiss_dir}\n"
            "Re-run ingestion or update paths."
        )

    # ---- Public API ----
    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        if not self.vss:
            return []
        if len(self.vss) == 1 and self.vss[0][0].lower() == "faiss":
            docs_scored = self.vss[0][1].similarity_search_with_score(query, k=k)
            return [d for d, _ in docs_scored]
        merged = multi_collection_retrieve(self.vss, query, k=k)
        return [d for _, d, _ in merged]

    def build_context(self, docs: List[Document], include_graph: bool = True) -> str:
        # Optional KG-directed boost from local docs, based on query tickers is done in UI layer
        return build_context(docs, self.G, include_graph=include_graph)

    def debug_collections(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"backend": self.backend_label, "collections": []}
        if "faiss" in self.backend_label.lower():
            return info
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            names = client.list_collections()
            for name in names:
                col = client.get_collection(name)
                cnt = col.count()
                info["collections"].append({"name": name, "count": cnt})
        except Exception as e:
            info["error"] = str(e)
        return info
