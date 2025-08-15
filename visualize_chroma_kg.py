# visualize_chroma_kg.py
"""
Interactive visualization for the comprehensive S&P 500 Knowledge Graph
produced by chromadb_sp500_ingest.py

Usage examples:
  python visualize_chroma_kg.py
  python visualize_chroma_kg.py --out kg.html
  python visualize_chroma_kg.py --sector "Information Technology" --min-degree 2 --out it_kg.html
  python visualize_chroma_kg.py --industry "Semiconductors & Semiconductor Equipment" --limit-nodes 400
  python visualize_chroma_kg.py --node-type company --highlight NVDA AAPL MSFT --out focus.html
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import Optional, Set, List

import networkx as nx

try:
    from pyvis.network import Network
    HAVE_PYVIS = True
except Exception:
    HAVE_PYVIS = False

# ---------- Config ----------
STORE = Path(__file__).parent / "store"
GRAPH_FILE = STORE / "comprehensive_knowledge_graph.pkl"

NODE_COLORS = {
    "index": "#1f77b4",
    "sector": "#9467bd",
    "industry": "#8c564b",
    "country": "#2ca02c",
    "exchange": "#bcbd22",
    "company": "#ff7f0e",
}
EDGE_COLORS = {
    "contains": "#7f7f7f",
    "operates_in": "#17becf",
    "belongs_to": "#17becf",
    "headquartered_in": "#2ca02c",
    "listed_on": "#bcbd22",
    "includes_companies_from": "#2ca02c",
    "trades_on": "#bcbd22",
    "similar_market_cap": "#d62728",
}

def _company_size(attrs: dict) -> float:
    if (attrs.get("node_type") or attrs.get("type")) != "company":
        return 18.0
    mc = attrs.get("market_cap")
    if isinstance(mc, (int, float)) and mc > 0:
        return 10.0 + 6.0 * math.log10(mc)
    score = attrs.get("composite_score")
    if isinstance(score, (int, float)) and score > 0:
        return 8.0 + 0.2 * score
    return 10.0

def _pick_color(attrs: dict) -> str:
    t = attrs.get("node_type") or attrs.get("type")
    return NODE_COLORS.get(t, "#999999")

def _edge_color(rel: Optional[str]) -> str:
    return EDGE_COLORS.get(rel or "", "#cccccc")

def _tooltip(n: str, a: dict) -> str:
    t = [f"<b>{a.get('name', n)}</b>"]
    nt = a.get("node_type") or a.get("type")
    if nt: t.append(f"Type: {nt}")
    for key in ("ticker","sector","industry","country","exchange"):
        if a.get(key): t.append(f"{key.capitalize()}: {a.get(key)}")
    if a.get("market_cap") is not None:
        try:
            t.append(f"Market Cap: {int(a['market_cap']):,}")
        except Exception:
            t.append(f"Market Cap: {a['market_cap']}")
    for key in ("composite_score","beta","dividend_yield","profit_margin","roe","roa"):
        if a.get(key) is not None: t.append(f"{key.replace('_',' ').title()}: {a[key]}")
    return "<br>".join(t)

def _filter_graph(
    G: nx.Graph,
    sector: Optional[str],
    industry: Optional[str],
    node_type: Optional[str],
    min_degree: int,
    limit_nodes: Optional[int],
) -> nx.Graph:
    H = G.copy()

    # If sector/industry set, keep the connected component around that node
    if sector:
        sid = f"SECTOR_{sector.replace(' ', '_').upper()}"
        if sid in H:
            keep = nx.node_connected_component(H, sid)
            H = H.subgraph(keep).copy()
    if industry:
        iid = f"INDUSTRY_{industry.replace(' ', '_').upper()}"
        if iid in H:
            keep = nx.node_connected_component(H, iid)
            H = H.subgraph(keep).copy()

    if node_type:
        keep_nodes = [n for n, d in H.nodes(data=True)
                      if (d.get("node_type") or d.get("type")) == node_type]
        H = H.subgraph(keep_nodes).copy()

    if min_degree > 0:
        drop = [n for n, deg in dict(H.degree()).items() if deg < min_degree]
        H.remove_nodes_from(drop)

    if limit_nodes and H.number_of_nodes() > limit_nodes:
        ranked = sorted(H.degree, key=lambda x: x[1], reverse=True)
        keep = {n for n, _ in ranked[:limit_nodes]}
        H = H.subgraph(keep).copy()

    return H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kg.html", help="Output HTML (or PNG if matplotlib fallback)")
    ap.add_argument("--sector", default=None, help="Exact sector name to filter")
    ap.add_argument("--industry", default=None, help="Exact industry name to filter")
    ap.add_argument("--node-type", default=None,
                    choices=["index","sector","industry","country","exchange","company"],
                    help="Filter to a single node type")
    ap.add_argument("--min-degree", type=int, default=0, help="Remove nodes with degree < min-degree")
    ap.add_argument("--limit-nodes", type=int, default=None, help="Keep top-N nodes by degree")
    ap.add_argument("--highlight", nargs="*", help="Tickers to highlight (larger size + white border)")
    args = ap.parse_args()

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_FILE}. Run ingestion first.")

    with open(GRAPH_FILE, "rb") as f:
        G: nx.Graph = pickle.load(f)

    H = _filter_graph(G, args.sector, args.industry, args.node_type, args.min_degree, args.limit_nodes)

    # Optional: mark highlights
    highlights: Set[str] = set([t.upper() for t in (args.highlight or [])]) if args.highlight else set()

    if HAVE_PYVIS:
        net = Network(height="850px", width="100%", bgcolor="#111111", font_color="#EEEEEE", directed=False)
        # Stable, nice layout
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=200,
                               spring_strength=0.01, damping=0.8, overlap=1)

        for n, a in H.nodes(data=True):
            label = a.get("ticker") or a.get("name") or n
            size = _company_size(a)
            color = _pick_color(a)
            title = _tooltip(n, a)
            border_width = 5 if (a.get("ticker") or n) in highlights else 0
            net.add_node(
                n, label=label, title=title, color=color, size=size,
                borderWidth=border_width, borderWidthSelected=6
            )

        for u, v, e in H.edges(data=True):
            rel = e.get("relationship")
            net.add_edge(u, v, color=_edge_color(rel), title=rel or "")

        # Write HTML directly (avoids Jinja/Notebook issues)
        net.write_html(args.out, open_browser=False, notebook=False)
        print(f"[OK] Wrote interactive graph to {args.out}")

    else:
        # Static fallback if PyVis unavailable
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(H, seed=42, k=0.45)
        node_colors = [_pick_color(H.nodes[n]) for n in H.nodes()]
        node_sizes = [_company_size(H.nodes[n]) * 10 for n in H.nodes()]
        edge_cols = [_edge_color(H.edges[e].get("relationship")) for e in H.edges()]

        plt.figure(figsize=(14, 10))
        nx.draw_networkx_edges(H, pos, edge_color=edge_cols, alpha=0.35, width=1.0)
        nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.0)
        labels = {n: (H.nodes[n].get("ticker") or H.nodes[n].get("name") or n)
                  for n in H.nodes()
                  if (H.nodes[n].get("node_type") or H.nodes[n].get("type")) == "company"
                  and H.degree[n] >= max(2, args.min_degree)}
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)
        plt.axis("off")
        out_png = args.out if args.out.lower().endswith(".png") else args.out.replace(".html", ".png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print(f"[OK] Wrote static graph to {out_png}\n(Hint: pip install pyvis for interactive HTML)")
        
if __name__ == "__main__":
    main()
