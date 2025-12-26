from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # query_vec: (d,), mat: (n, d)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return (m @ q).astype(np.float32)


@dataclass(frozen=True)
class EmbeddingCache:
    path: Path

    def load(self) -> dict[str, np.ndarray] | None:
        if not self.path.exists():
            return None
        data = np.load(self.path, allow_pickle=False)
        keys = data["keys"].tolist()
        vectors = data["vectors"]
        out: dict[str, np.ndarray] = {}
        for i, k in enumerate(keys):
            out[str(k)] = vectors[i]
        return out

    def save(self, embeddings: dict[str, np.ndarray]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(embeddings.keys()), dtype=object)
        vectors = np.stack([embeddings[k] for k in embeddings.keys()], axis=0)
        np.savez_compressed(self.path, keys=keys, vectors=vectors)


def chunked(xs: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

