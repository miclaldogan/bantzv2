"""
Bantz v3 — Deep Memory Probe (#170)

Spontaneous vector memory retrieval for conversational depth.
Probabilistically surfaces relevant past memories during casual chat
to make Bantz feel genuinely *remembering* rather than just echoing.

Three senior engineering fixes applied:
  1. Alzheimer Math Fix  — threshold on *raw* cosine, decay for ranking only
  2. Déjà Vu Fix         — track recently-used memory IDs, skip repeats
  3. Butler Lore Fix     — inject memories as 1920s human recollections

Usage:
    from bantz.memory.deep_probe import deep_probe
    hint = await deep_probe.probe("I need to study for that exam")
    # → "You once mentioned an upcoming exam …" or ""
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Optional

log = logging.getLogger("bantz.deep_probe")


# ── Butler-lore injection preamble ─────────────────────────────────────────

_INJECT_PREAMBLE = (
    "Spontaneous Human Memory — the following recollections surfaced "
    "naturally in your mind as a human butler. Weave them in subtly, "
    "as if you personally recall overhearing or being told these things. "
    "Do NOT say 'according to my records' or anything robotic.\n"
)


class DeepMemoryProbe:
    """Probabilistic vector memory retrieval for conversational depth.

    Designed to *augment* chat, not replace ``_vector_context`` which
    serves explicit search / hybrid-search.  ``probe()`` runs on a
    rate-limited cadence (every N messages) and only surfaces memories
    that pass a raw cosine threshold, ranked by time-decayed score.
    """

    def __init__(
        self,
        *,
        cosine_threshold: float = 0.72,
        max_results: int = 3,
        rate_every_n: int = 3,
        decay_half_life_days: float = 30.0,
        dedup_ttl_seconds: float = 600.0,
    ) -> None:
        self._cosine_threshold = cosine_threshold
        self._max_results = max_results
        self._rate_every_n = rate_every_n
        self._decay_half_life = decay_half_life_days
        self._dedup_ttl = dedup_ttl_seconds

        self._call_counter: int = 0
        # Déjà Vu guard: {message_id: timestamp_used}
        self._recently_used: dict[int, float] = {}

    # ── Public API ──────────────────────────────────────────────────────

    async def probe(self, user_message: str) -> str:
        """Return a deep-memory hint string, or '' if nothing surfaces.

        Call this on every user turn; rate-limiting is handled internally.
        """
        from bantz.config import config

        if not getattr(config, "deep_memory_enabled", True):
            return ""

        # ── Rate-limit: probe only every N messages ─────────────────
        self._call_counter += 1
        if self._call_counter % self._rate_every_n != 0:
            return ""

        # Read config overrides
        threshold = getattr(config, "deep_memory_threshold", self._cosine_threshold)
        max_results = getattr(config, "deep_memory_max_results", self._max_results)

        # ── Embed the user message ──────────────────────────────────
        try:
            from bantz.memory.embeddings import embedder

            if not config.embedding_enabled:
                return ""

            query_vec = await embedder.embed(user_message)
            if query_vec is None:
                return ""
        except Exception as exc:
            log.debug("Deep probe embedding failed: %s", exc)
            return ""

        # ── Brute-force cosine search ───────────────────────────────
        try:
            from bantz.core.memory import memory

            vs = memory.vectors
            if vs is None or vs.count() == 0:
                return ""

            candidates = self._search_raw(vs, query_vec, threshold)
        except Exception as exc:
            log.debug("Deep probe search failed: %s", exc)
            return ""

        if not candidates:
            return ""

        # ── Rank by temporal-decay score ────────────────────────────
        ranked = self._rank_with_decay(candidates)

        # ── Déjà Vu filter ──────────────────────────────────────────
        self._evict_stale()
        filtered = [
            r for r in ranked
            if r["message_id"] not in self._recently_used
        ]
        if not filtered:
            return ""

        # ── Take top-N and mark as used ─────────────────────────────
        top = filtered[:max_results]
        now = time.monotonic()
        for r in top:
            self._recently_used[r["message_id"]] = now

        # ── Format butler-lore hint ─────────────────────────────────
        return self._format_hint(top)

    def reset(self) -> None:
        """Reset internal state (useful for testing)."""
        self._call_counter = 0
        self._recently_used.clear()

    # ── Internal helpers ────────────────────────────────────────────────

    def _search_raw(
        self,
        vs,  # VectorStore
        query_vec: list[float],
        threshold: float,
    ) -> list[dict]:
        """Search with raw cosine threshold — NO time-decay in the filter.

        Alzheimer Math Fix: the threshold is applied to the *raw* cosine
        similarity so genuinely relevant old memories are never discarded
        simply because they are old.  Temporal decay is used only for
        *ranking* among the survivors.
        """
        from bantz.memory.vector_store import _cosine_similarity, _blob_to_vec

        rows = vs._conn.execute(
            """SELECT mv.message_id, mv.embedding, mv.dim,
                      m.role, m.content, m.tool_used, m.created_at,
                      m.conversation_id
               FROM message_vectors mv
               JOIN messages m ON m.id = mv.message_id"""
        ).fetchall()

        results: list[dict] = []
        for row in rows:
            vec = _blob_to_vec(row["embedding"], row["dim"])
            cosine = _cosine_similarity(query_vec, vec)

            if cosine < threshold:
                continue

            results.append({
                "message_id": row["message_id"],
                "cosine": round(cosine, 4),
                "role": row["role"],
                "content": row["content"],
                "tool_used": row["tool_used"],
                "created_at": row["created_at"],
                "conv_id": row["conversation_id"],
            })

        return results

    def _rank_with_decay(self, candidates: list[dict]) -> list[dict]:
        """Rank candidates by ``cosine * exp(-age_days / half_life)``.

        Recent memories get a natural boost but old-yet-relevant ones
        are never killed — they just compete with a gentle handicap.
        """
        now = datetime.utcnow()
        scored: list[tuple[float, dict]] = []

        for c in candidates:
            age_days = 0.0
            if c.get("created_at"):
                try:
                    created = datetime.fromisoformat(
                        c["created_at"].replace("Z", "+00:00").replace("+00:00", "")
                    )
                    age_days = max(
                        (now - created).total_seconds() / 86400, 0.0
                    )
                except (ValueError, TypeError):
                    pass

            decay = math.exp(-age_days / self._decay_half_life)
            rank_score = c["cosine"] * decay
            c["rank_score"] = round(rank_score, 4)
            scored.append((rank_score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def _evict_stale(self) -> None:
        """Remove expired entries from the déjà-vu set."""
        now = time.monotonic()
        stale = [
            mid for mid, ts in self._recently_used.items()
            if now - ts > self._dedup_ttl
        ]
        for mid in stale:
            del self._recently_used[mid]

    @staticmethod
    def _format_hint(memories: list[dict]) -> str:
        """Format memories as a butler-lore injection block.

        The LLM sees this as system-level context and naturally weaves
        the recollections into its response without sounding robotic.
        """
        lines: list[str] = [_INJECT_PREAMBLE]
        for m in memories:
            role_label = "Ma'am" if m["role"] == "user" else "You"
            snippet = m["content"][:250].strip()
            age_str = ""
            if m.get("created_at"):
                try:
                    created = datetime.fromisoformat(
                        m["created_at"].replace("Z", "+00:00").replace("+00:00", "")
                    )
                    age_days = (datetime.utcnow() - created).total_seconds() / 86400
                    if age_days < 1:
                        age_str = "earlier today"
                    elif age_days < 2:
                        age_str = "yesterday"
                    elif age_days < 7:
                        age_str = f"{int(age_days)} days ago"
                    elif age_days < 30:
                        age_str = f"about {int(age_days / 7)} weeks ago"
                    else:
                        age_str = f"roughly {int(age_days / 30)} months ago"
                except (ValueError, TypeError):
                    age_str = "some time ago"

            ts_part = f" ({age_str})" if age_str else ""
            lines.append(
                f"- {role_label} once said{ts_part}: \"{snippet}\""
            )
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────

deep_probe = DeepMemoryProbe()
