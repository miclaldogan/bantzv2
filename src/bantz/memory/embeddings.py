"""
Bantz v3 — Embedding Client

Wraps Ollama's ``/api/embeddings`` endpoint for local vector generation.
Falls back gracefully if Ollama is unreachable or the model is missing.

Usage:
    from bantz.memory.embeddings import embedder

    vec = await embedder.embed("hello world")          # list[float] | None
    vecs = await embedder.embed_batch(["a", "b", "c"]) # list[list[float]]
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from bantz.config import config

log = logging.getLogger("bantz.embeddings")

# Dimension of the default nomic-embed-text model
DEFAULT_DIM = 768


class Embedder:
    """Async wrapper around Ollama /api/embeddings (or /api/embed)."""

    def __init__(self) -> None:
        self._model: str = ""
        self._dim: int = DEFAULT_DIM
        self._available: Optional[bool] = None
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        # Lazy-load shared client to leverage HTTP connection pooling
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    @property
    def model(self) -> str:
        return self._model or config.embedding_model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> Optional[list[float]]:
        """Embed a single text.  Returns None on failure."""
        if not text or not text.strip():
            return None
        try:
            # Try /api/embed first (newer Ollama), fall back to /api/embeddings
            resp = await self.client.post(
                f"{config.ollama_base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=30.0,
            )
            if resp.status_code == 404:
                # Older Ollama version — use /api/embeddings
                resp = await self.client.post(
                    f"{config.ollama_base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30.0,
                )
            resp.raise_for_status()
            data = resp.json()

            # /api/embed returns {"embeddings": [[...]]}
            if "embeddings" in data and data["embeddings"]:
                vec = data["embeddings"][0]
            # /api/embeddings returns {"embedding": [...]}
            elif "embedding" in data:
                vec = data["embedding"]
            else:
                log.warning("Unexpected embedding response: %s", list(data.keys()))
                return None

            if vec:
                self._dim = len(vec)
                self._available = True
            return vec

        except httpx.HTTPStatusError as exc:
            log.warning("Embedding HTTP error %s: %s", exc.response.status_code, exc)
            self._available = False
            return None
        except Exception as exc:
            log.debug("Embedding error: %s", exc)
            self._available = False
            return None

    async def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed multiple texts.  Returns list aligned with input (None for failures)."""
        results: list[Optional[list[float]]] = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results

    async def is_available(self) -> bool:
        """Check if the embedding model is reachable."""
        if self._available is not None:
            return self._available
        _ = await self.embed("test")
        return self._available or False


# Singleton
embedder = Embedder()
