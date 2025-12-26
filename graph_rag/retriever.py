from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .graph import KnowledgeGraph
from .openai_client import OpenAIClient, OpenAIConfigError
from .types import EdgeType, NodeType
from .util import EmbeddingCache, cosine_sim_matrix, normalize_text


def _node_display(kg: KnowledgeGraph, nid: str) -> str:
    d = kg.g.nodes.get(nid, {})
    t = d.get("type", "")
    name = d.get("name") or d.get("title") or nid
    return f"{t}:{name}"


def _candidate_profile_text(kg: KnowledgeGraph, candidate_id: str) -> str:
    n = kg.g.nodes[candidate_id]
    parts = [
        f"Candidate {n.get('name','')}",
        f"Location: {n.get('location','')}",
        f"Years experience: {n.get('years_experience',0)}",
        f"Summary: {n.get('summary','')}",
    ]
    # Include 1-hop neighbor names for richer embedding.
    neigh = []
    for v in kg.g.successors(candidate_id):
        vd = kg.g.nodes.get(v, {})
        neigh.append(f"{vd.get('type','')}: {vd.get('name') or vd.get('title') or v}")
    if neigh:
        parts.append("Entities: " + "; ".join(sorted(set(neigh))))
    return "\n".join(parts).strip()


@dataclass(frozen=True)
class RetrievalConfig:
    neighbor_depth: int = 2
    top_seeds: int = 12
    top_candidates: int = 5
    # weights
    w_graph: float = 0.75
    w_embed: float = 0.25


class GraphRetriever:
    def __init__(
        self,
        kg: KnowledgeGraph,
        *,
        cache: EmbeddingCache | None = None,
        openai: OpenAIClient | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.kg = kg
        self.cache = cache
        self.openai = openai
        self.config = config or RetrievalConfig()

        self._embeddings: dict[str, np.ndarray] = {}
        if self.cache:
            loaded = self.cache.load()
            if loaded:
                self._embeddings.update(loaded)

    def _get_or_create_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """
        Returns aligned vectors for texts. Uses cache if available.
        If OpenAI is unavailable (no key), returns None.
        """
        try:
            if not self.openai:
                self.openai = OpenAIClient()
        except OpenAIConfigError:
            return None

        model = self.openai.models.embedding
        keys = [normalize_text(f"{model}::{t}") for t in texts]
        # We hash in OpenAIClient; here we keep same stable hash format by recomputing.
        from .util import stable_hash

        hashed = [stable_hash(k) for k in keys]
        missing_idx = [i for i, h in enumerate(hashed) if h not in self._embeddings]
        if missing_idx:
            miss_texts = [texts[i] for i in missing_idx]
            miss_keys, miss_vecs = self.openai.embed_texts(miss_texts, model=model)
            for k, v in zip(miss_keys, miss_vecs, strict=True):
                self._embeddings[k] = v
            if self.cache:
                self.cache.save(self._embeddings)

        return np.stack([self._embeddings[h] for h in hashed], axis=0)

    def _role_seed_nodes(self, role_id: str) -> list[str]:
        role_node = f"{NodeType.ROLE.value}:{role_id}"
        if role_node not in self.kg.g:
            raise KeyError(f"Role {role_id!r} not found in graph")

        seeds: list[str] = []
        for _, dst, edata in self.kg.g.out_edges(role_node, data=True):
            et = edata.get("type")
            if et in (EdgeType.REQUIRES_SKILL.value, EdgeType.REQUIRES_CERT.value, EdgeType.REQUIRES_DOMAIN.value):
                seeds.append(dst)
        return sorted(set(seeds))

    def _text_seed_nodes(self, query: str) -> list[str]:
        """
        Seed by embedding similarity over entity nodes (skills/certs/domains/titles/companies).
        Falls back to substring match if embeddings are unavailable.
        """
        entity_types = {NodeType.SKILL.value, NodeType.CERTIFICATION.value, NodeType.DOMAIN.value, NodeType.TITLE.value}
        entity_nodes = [n for n, d in self.kg.g.nodes(data=True) if d.get("type") in entity_types]
        entity_texts = [_node_display(self.kg, n) for n in entity_nodes]

        vecs = self._get_or_create_embeddings([query] + entity_texts)
        if vecs is None:
            q = normalize_text(query)
            hits = []
            for nid in entity_nodes:
                nm = normalize_text(str(self.kg.g.nodes[nid].get("name", "")))
                if nm and (nm in q or q in nm):
                    hits.append(nid)
            return hits[: self.config.top_seeds]

        qv = vecs[0]
        mat = vecs[1:]
        sims = cosine_sim_matrix(qv, mat)
        idx = np.argsort(-sims)[: self.config.top_seeds]
        return [entity_nodes[int(i)] for i in idx.tolist()]

    def _graph_score(self, candidate_id: str, seeds: set[str]) -> tuple[float, dict[str, Any]]:
        """
        Graph score: fraction of seed nodes reachable from candidate within neighbor_depth,
        with extra weight for direct HAS_* matches.
        """
        reachable = self.kg.neighbors(candidate_id, depth=self.config.neighbor_depth)
        matched = sorted(seeds.intersection(reachable))
        if not seeds:
            return 0.0, {"matched": [], "direct": []}

        # direct matches are 1-hop outgoing edges from candidate
        direct = set(self.kg.g.successors(candidate_id)).intersection(seeds)
        # score: 0.7 for matched reachability + 0.3 for directness
        s_reach = len(matched) / max(1, len(seeds))
        s_direct = len(direct) / max(1, len(seeds))
        score = 0.7 * s_reach + 0.3 * s_direct
        return float(score), {"matched": matched, "direct": sorted(direct)}

    def _embed_score(self, role_text: str, candidate_ids: list[str]) -> dict[str, float]:
        vecs = self._get_or_create_embeddings([role_text] + [_candidate_profile_text(self.kg, c) for c in candidate_ids])
        if vecs is None:
            return {c: 0.0 for c in candidate_ids}

        qv = vecs[0]
        mat = vecs[1:]
        sims = cosine_sim_matrix(qv, mat)
        return {c: float(s) for c, s in zip(candidate_ids, sims.tolist(), strict=True)}

    def retrieve(self, *, role_id: str | None = None, query: str | None = None) -> dict[str, Any]:
        """
        Graph-RAG retrieval:
        - choose seed nodes (from role requirements graph edges OR from query embedding -> entity nodes)
        - expand via neighborhoods to find candidate nodes that connect
        - score candidates with graph overlap + optional embedding similarity
        - return ranked candidates + evidence paths
        """
        if not role_id and not query:
            raise ValueError("Provide role_id or query")

        if role_id:
            seeds = self._role_seed_nodes(role_id)
            role_node = f"{NodeType.ROLE.value}:{role_id}"
            role_title = self.kg.g.nodes[role_node].get("title", role_id)
            role_text = f"{role_title}\n{self.kg.g.nodes[role_node].get('summary','')}".strip()
        else:
            seeds = self._text_seed_nodes(query or "")
            role_text = query or ""

        seed_set = set(seeds)
        # candidate shortlist: candidates that reach at least 1 seed within depth
        candidates = self.kg.nodes_of_type(NodeType.CANDIDATE)
        graph_scores: dict[str, tuple[float, dict[str, Any]]] = {}
        for c in candidates:
            score, details = self._graph_score(c, seed_set)
            if score > 0:
                graph_scores[c] = (score, details)

        # If nothing matched, fall back to embedding similarity across all candidates.
        shortlist = sorted(graph_scores.keys())
        if not shortlist:
            shortlist = candidates

        embed_scores = self._embed_score(role_text, shortlist)

        results = []
        for c in shortlist:
            gscore, details = graph_scores.get(c, (0.0, {"matched": [], "direct": []}))
            escore = embed_scores.get(c, 0.0)
            final = self.config.w_graph * gscore + self.config.w_embed * ((escore + 1) / 2)  # normalize [-1,1] -> [0,1]

            ev = self.kg.candidate_evidence(c, details.get("matched", []))
            results.append(
                {
                    "candidate_id": c,
                    "candidate_name": self.kg.g.nodes[c].get("name", ""),
                    "score": float(final),
                    "graph_score": float(gscore),
                    "embed_score": float(escore),
                    "matched_entities": [_node_display(self.kg, n) for n in details.get("matched", [])],
                    "direct_matches": [_node_display(self.kg, n) for n in details.get("direct", [])],
                    "evidence_paths": { _node_display(self.kg,k): [ _node_display(self.kg, x) for x in v ] for k, v in ev.items() },
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return {
            "seed_entities": [_node_display(self.kg, n) for n in seeds],
            "results": results[: self.config.top_candidates],
        }

