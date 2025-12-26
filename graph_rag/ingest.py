from __future__ import annotations

from typing import Any

from .graph import KnowledgeGraph
from .types import CandidateJSON, EdgeType, NodeType, RoleJSON


def add_candidate(kg: KnowledgeGraph, cand: CandidateJSON) -> str:
    cid = str(cand.get("id") or "")
    if not cid:
        raise ValueError("candidate missing id")
    cnode = kg.upsert_node(
        NodeType.CANDIDATE,
        cid,
        name=str(cand.get("name", "")),
        location=str(cand.get("location", "")),
        years_experience=float(cand.get("years_experience") or 0.0),
        summary=str(cand.get("summary", "")),
    )

    for s in cand.get("skills", []) or []:
        snode = kg.upsert_node(NodeType.SKILL, s, name=str(s))
        kg.add_edge(cnode, snode, EdgeType.HAS_SKILL)

    for c in cand.get("certifications", []) or []:
        ctnode = kg.upsert_node(NodeType.CERTIFICATION, c, name=str(c))
        kg.add_edge(cnode, ctnode, EdgeType.HAS_CERT)

    for d in cand.get("domains", []) or []:
        dnode = kg.upsert_node(NodeType.DOMAIN, d, name=str(d))
        kg.add_edge(cnode, dnode, EdgeType.HAS_DOMAIN)

    for exp in cand.get("experience", []) or []:
        if not isinstance(exp, dict):
            continue
        company = exp.get("company")
        title = exp.get("title")
        years = exp.get("years")
        if company:
            comp = kg.upsert_node(NodeType.COMPANY, str(company), name=str(company))
            kg.add_edge(cnode, comp, EdgeType.WORKED_AT, years=float(years or 0.0))
        if title:
            t = kg.upsert_node(NodeType.TITLE, str(title), name=str(title))
            kg.add_edge(cnode, t, EdgeType.HAD_TITLE, years=float(years or 0.0))

    return cnode


def add_role(kg: KnowledgeGraph, role: RoleJSON) -> str:
    rid = str(role.get("id") or "")
    if not rid:
        raise ValueError("role missing id")

    rnode = kg.upsert_node(
        NodeType.ROLE,
        rid,
        title=str(role.get("title", "")),
        location=str(role.get("location", "")),
        seniority=str(role.get("seniority", "")),
        summary=str(role.get("summary", "")),
    )

    for s in role.get("required_skills", []) or []:
        snode = kg.upsert_node(NodeType.SKILL, s, name=str(s))
        kg.add_edge(rnode, snode, EdgeType.REQUIRES_SKILL, weight=1.0)

    for s in role.get("preferred_skills", []) or []:
        snode = kg.upsert_node(NodeType.SKILL, s, name=str(s))
        kg.add_edge(rnode, snode, EdgeType.REQUIRES_SKILL, weight=0.4)

    for c in role.get("required_certifications", []) or []:
        ctnode = kg.upsert_node(NodeType.CERTIFICATION, c, name=str(c))
        kg.add_edge(rnode, ctnode, EdgeType.REQUIRES_CERT, weight=1.0)

    for d in role.get("domains", []) or []:
        dnode = kg.upsert_node(NodeType.DOMAIN, d, name=str(d))
        kg.add_edge(rnode, dnode, EdgeType.REQUIRES_DOMAIN, weight=0.8)

    return rnode


def add_temporary_role_from_requirements(
    kg: KnowledgeGraph, *, role_id: str, title: str, requirements: dict[str, Any]
) -> str:
    rnode = kg.upsert_node(NodeType.ROLE, role_id, title=title, summary=requirements.get("summary", ""))
    for s in requirements.get("skills", []) or []:
        snode = kg.upsert_node(NodeType.SKILL, str(s), name=str(s))
        kg.add_edge(rnode, snode, EdgeType.REQUIRES_SKILL, weight=1.0)
    for c in requirements.get("certifications", []) or []:
        ctnode = kg.upsert_node(NodeType.CERTIFICATION, str(c), name=str(c))
        kg.add_edge(rnode, ctnode, EdgeType.REQUIRES_CERT, weight=1.0)
    for d in requirements.get("domains", []) or []:
        dnode = kg.upsert_node(NodeType.DOMAIN, str(d), name=str(d))
        kg.add_edge(rnode, dnode, EdgeType.REQUIRES_DOMAIN, weight=0.8)
    return rnode

