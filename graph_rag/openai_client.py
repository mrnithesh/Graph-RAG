from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from openai import OpenAI

from .util import chunked, stable_hash


class OpenAIConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAIModels:
    embedding: str = "text-embedding-3-small"
    chat: str = "gpt-4o-mini"


class OpenAIClient:
    """
    Thin wrapper around the official OpenAI Python SDK.

    - Embeddings: client.embeddings.create(...)
    - Chat Completions: client.chat.completions.create(...)
    """

    def __init__(self, models: OpenAIModels | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigError(
                "OPENAI_API_KEY is not set. Export it, e.g.:\n"
                "  export OPENAI_API_KEY='...'\n"
                "Then re-run."
            )
        self._client = OpenAI(api_key=api_key)
        self.models = models or OpenAIModels()

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> tuple[list[str], np.ndarray]:
        m = model or self.models.embedding
        # OpenAI embeddings API supports batching; keep chunks reasonable.
        vectors: list[np.ndarray] = []
        keys: list[str] = []
        for batch in chunked(texts, 128):
            resp = self._client.embeddings.create(model=m, input=batch)
            # Returned order matches input order.
            for t, item in zip(batch, resp.data, strict=True):
                v = np.array(item.embedding, dtype=np.float32)
                vectors.append(v)
                keys.append(stable_hash(f"{m}::{t}"))
        return keys, np.stack(vectors, axis=0)

    def chat_json(self, *, system: str, user: str, schema_hint: str, model: str | None = None) -> dict[str, Any]:
        """
        Requests the model to output JSON. We keep it simple and robust:
        - force JSON-only response (best-effort)
        - parse using Python json
        """
        import json

        m = model or self.models.chat
        prompt = (
            "Return ONLY valid JSON (no markdown). "
            "Schema:\n"
            f"{schema_hint}\n\n"
            f"User input:\n{user}\n"
        )
        resp = self._client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model did not return valid JSON. Raw content:\n{content}") from e

