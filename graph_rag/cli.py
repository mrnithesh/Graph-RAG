from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .system import GraphRAG
from .util import json_dumps


def _print_results(console: Console, payload: dict) -> None:
    console.print("\n[bold]Seed entities[/bold]")
    for s in payload.get("seed_entities", []):
        console.print(f"- {s}")

    table = Table(title="Top Candidates (Graph-RAG)")
    table.add_column("Rank", justify="right")
    table.add_column("Candidate")
    table.add_column("Score", justify="right")
    table.add_column("Graph", justify="right")
    table.add_column("Embed", justify="right")
    table.add_column("Direct matches")

    for i, r in enumerate(payload.get("results", []), start=1):
        table.add_row(
            str(i),
            f"{r.get('candidate_name','')} ({r.get('candidate_id','')})",
            f"{r.get('score',0):.3f}",
            f"{r.get('graph_score',0):.3f}",
            f"{r.get('embed_score',0):.3f}",
            "\n".join(r.get("direct_matches", [])[:6]),
        )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]Evidence paths[/bold] (why each candidate matched)")
    for r in payload.get("results", []):
        console.print(f"\n[bold]{r.get('candidate_name','')}[/bold] — score={r.get('score',0):.3f}")
        eps: dict = r.get("evidence_paths", {}) or {}
        # show up to 5 evidence paths
        for k in list(eps.keys())[:5]:
            console.print(f"- {k}")
            console.print("  " + " -> ".join(eps[k]))


def main() -> None:
    p = argparse.ArgumentParser(description="Graph-RAG recruitment prototype (energy sector)")
    p.add_argument("--candidates", default="data/candidates.json", help="Path to candidates JSON")
    p.add_argument("--roles", default="data/roles.json", help="Path to roles JSON")
    p.add_argument("--role-id", default=None, help="Role id from roles.json (e.g. role_001)")
    p.add_argument("--query", default=None, help="Free-text requirement query (alternative to --role-id)")
    p.add_argument("--free-role-title", default=None, help="Free-text role title (LLM extraction)")
    p.add_argument("--free-role-desc", default=None, help="Free-text role description (LLM extraction)")
    p.add_argument("--top", type=int, default=5, help="Top candidates to return")
    p.add_argument("--depth", type=int, default=2, help="Neighborhood depth for graph expansion")
    p.add_argument("--cache", default=".cache/graph_rag/embeddings.npz", help="Embedding cache path")
    p.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = p.parse_args()

    console = Console()
    rag = GraphRAG.from_files(
        candidates_path=args.candidates,
        roles_path=args.roles,
        cache_path=args.cache,
        neighbor_depth=args.depth,
        top_candidates=args.top,
    )

    if args.free_role_title and args.free_role_desc:
        payload = rag.retrieve_from_free_text_role(title=args.free_role_title, description=args.free_role_desc)
    else:
        payload = rag.retrieve(role_id=args.role_id, query=args.query)

    if args.json:
        console.print(json_dumps(payload))
    else:
        _print_results(console, payload)


if __name__ == "__main__":
    main()

