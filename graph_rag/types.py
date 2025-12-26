from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict


class NodeType(str, Enum):
    CANDIDATE = "candidate"
    ROLE = "role"
    SKILL = "skill"
    CERTIFICATION = "certification"
    DOMAIN = "domain"
    COMPANY = "company"
    TITLE = "title"


class EdgeType(str, Enum):
    HAS_SKILL = "HAS_SKILL"
    HAS_CERT = "HAS_CERT"
    HAS_DOMAIN = "HAS_DOMAIN"
    WORKED_AT = "WORKED_AT"
    HAD_TITLE = "HAD_TITLE"
    REQUIRES_SKILL = "REQUIRES_SKILL"
    REQUIRES_CERT = "REQUIRES_CERT"
    REQUIRES_DOMAIN = "REQUIRES_DOMAIN"


class CandidateJSON(TypedDict, total=False):
    id: str
    name: str
    location: str
    years_experience: float
    summary: str
    skills: list[str]
    certifications: list[str]
    domains: list[str]
    experience: list[dict[str, Any]]


class RoleJSON(TypedDict, total=False):
    id: str
    title: str
    location: str
    seniority: Literal["junior", "mid", "senior", "lead"]
    summary: str
    required_skills: list[str]
    preferred_skills: list[str]
    required_certifications: list[str]
    domains: list[str]

