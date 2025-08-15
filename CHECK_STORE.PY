# check_store.py
from pathlib import Path
import pickle, json

BASE = Path(__file__).parent
STORE = BASE / "store"
CHROMA = STORE / "chromadb"
FAISS  = STORE / "comprehensive_vectorstore"
GRAPH  = STORE / "comprehensive_knowledge_graph.pkl"
DOCS   = STORE / "comprehensive_stocks_data.json"
SUMMARY = STORE / "ingestion_summary.json"

print(f"\nStore path: {STORE.resolve()}")
print("Exists:", STORE.exists())
print("\nContents:")
for p in sorted(STORE.glob("**/*")):
    if p.is_file():
        print(" - FILE ", p.relative_to(STORE))
    else:
        print(" - DIR  ", p.relative_to(STORE))

# Graph check
if GRAPH.exists():
    try:
        import networkx as nx
        G = pickle.load(open(GRAPH,"rb"))
        print(f"\n[OK] Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    except Exception as e:
        print(f"\n[ERR] Failed to load graph: {e}")
else:
    print("\n[MISS] Graph pickle not found:", GRAPH)

# Chroma/FAISS presence
print("\nChroma dir present:", CHROMA.exists(), "has files:", any(CHROMA.iterdir()) if CHROMA.exists() else False)
print("FAISS dir present:", FAISS.exists(), "has files:", any(FAISS.iterdir()) if FAISS.exists() else False)

# Docs / summary
if DOCS.exists():
    try:
        data = json.load(open(DOCS,"r",encoding="utf-8"))
        print(f"\nDocs JSON found: {DOCS.name} (items={len(data)})")
    except Exception as e:
        print(f"\n[ERR] Docs JSON load error: {e}")
else:
    print("\n[MISS] Docs JSON not found:", DOCS)

if SUMMARY.exists():
    try:
        meta = json.load(open(SUMMARY,"r",encoding="utf-8"))
        print(f"\nSummary found: {SUMMARY.name}")
        print("  keys:", list(meta.keys()))
    except Exception as e:
        print(f"\n[ERR] Summary load error: {e}")
else:
    print("\n[MISS] Summary not found:", SUMMARY)
