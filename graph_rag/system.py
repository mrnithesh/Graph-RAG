from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import KnowledgeGraph
from .ingest import add_candidate, add_role, add_temporary_role_from_requirements
from .openai_client import OpenAIClient, OpenAIConfigError
from .retriever import GraphRetriever, RetrievalConfig
from .util import EmbeddingCache


@dataclass
class GraphRAG:
    kg: KnowledgeGraph
    retriever: GraphRetriever
    openai: OpenAIClient | None = None

    @classmethod
    def from_files(
        cls,
        *,
        candidates_path: str | Path,
        roles_path: str | Path,
        cache_path: str | Path = ".cache/graph_rag/embeddings.npz",
        neighbor_depth: int = 2,
        top_candidates: int = 5,
    ) -> "GraphRAG":
        kg = KnowledgeGraph.empty()
        candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
        roles = json.loads(Path(roles_path).read_text(encoding="utf-8"))

        for c in candidates:
            add_candidate(kg, c)
        for r in roles:
            add_role(kg, r)

        cache = EmbeddingCache(path=Path(cache_path))
        config = RetrievalConfig(neighbor_depth=neighbor_depth, top_candidates=top_candidates)
        retriever = GraphRetriever(kg, cache=cache, config=config)
        return cls(kg=kg, retriever=retriever, openai=None)

    def parse_requirements_with_llm(self, text: str) -> dict[str, Any]:
        try:
            if not self.openai:
                self.openai = OpenAIClient()
        except OpenAIConfigError:
            # No LLM available: return a minimal empty structure.
            return {"summary": text, "skills": [], "certifications": [], "domains": []}

        schema_hint = """
{
  "summary": "string",
  "skills": ["string", "..."],
  "certifications": ["string", "..."],
  "domains": ["string", "..."]
}
""".strip()
        system = (
            "You extract structured hiring requirements for the energy sector. "
            "Only include concrete skills/certifications/domains that matter for matching."
        )
        out = self.openai.chat_json(system=system, user=text, schema_hint=schema_hint)
        # Make it safe/normalized
        return {
            "summary": str(out.get("summary", text)),
            "skills": [str(x) for x in (out.get("skills") or [])],
            "certifications": [str(x) for x in (out.get("certifications") or [])],
            "domains": [str(x) for x in (out.get("domains") or [])],
        }

    def retrieve(self, *, role_id: str | None = None, query: str | None = None) -> dict[str, Any]:
        return self.retriever.retrieve(role_id=role_id, query=query)

    def retrieve_from_free_text_role(self, *, title: str, description: str) -> dict[str, Any]:
        req = self.parse_requirements_with_llm(f"{title}\n{description}".strip())
        tmp_role_id = "role_tmp_llm"
        add_temporary_role_from_requirements(self.kg, role_id=tmp_role_id, title=title, requirements=req)
        return self.retrieve(role_id=tmp_role_id)

