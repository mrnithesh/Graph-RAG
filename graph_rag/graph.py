from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from .types import EdgeType, NodeType
from .util import normalize_text


def _node_key(ntype: NodeType, name_or_id: str) -> str:
    if ntype in (NodeType.CANDIDATE, NodeType.ROLE):
        return f"{ntype.value}:{name_or_id}"
    return f"{ntype.value}:{normalize_text(name_or_id)}"


@dataclass
class KnowledgeGraph:
    g: nx.MultiDiGraph

    @classmethod
    def empty(cls) -> "KnowledgeGraph":
        return cls(g=nx.MultiDiGraph())

    def upsert_node(self, ntype: NodeType, name_or_id: str, **attrs: Any) -> str:
        nid = _node_key(ntype, name_or_id)
        existing = self.g.nodes.get(nid, {})
        merged = {**existing, **attrs, "type": ntype.value}
        self.g.add_node(nid, **merged)
        return nid

    def add_edge(self, src: str, dst: str, etype: EdgeType, **attrs: Any) -> None:
        self.g.add_edge(src, dst, key=etype.value, type=etype.value, **attrs)

    def nodes_of_type(self, ntype: NodeType) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == ntype.value]

    def neighbors(self, node_id: str, *, depth: int = 1) -> set[str]:
        if depth <= 0:
            return {node_id}
        frontier = {node_id}
        seen = {node_id}
        for _ in range(depth):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(self.g.successors(n))
                nxt.update(self.g.predecessors(n))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return seen

    def candidate_evidence(self, candidate_id: str, seed_nodes: Iterable[str]) -> dict[str, list[str]]:
        """
        For each seed node, returns a shortest path from candidate to seed (if exists).
        Paths are node-id strings, suitable for explanation.
        """
        ev: dict[str, list[str]] = {}
        # Use undirected view for connectivity explanation.
        ug = self.g.to_undirected(as_view=True)
        for s in seed_nodes:
            if candidate_id == s:
                ev[s] = [candidate_id]
                continue
            try:
                p = nx.shortest_path(ug, source=candidate_id, target=s)
                ev[s] = p
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        return ev

