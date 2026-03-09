"""
Bantz v3 — Spatial Cache (#121)

Persistent SQLite cache for UI element coordinates per application.
Prevents repeated AT-SPI / VLM lookups for the same element.

Features:
  - Resolution-aware cache keys (element may move on resolution change)
  - TTL-based invalidation (entries > 24h get re-verified)
  - Confidence scoring: AT-SPI=1.0, VLM=0.7, cached decays 0.05/day
  - Hit-count tracking for analytics
  - LRU eviction at max 1000 entries
  - Cache warming support

Schema:
    CREATE TABLE spatial_cache (
        id INTEGER PRIMARY KEY,
        app_name TEXT NOT NULL,
        element_label TEXT NOT NULL,
        resolution_w INTEGER NOT NULL,
        resolution_h INTEGER NOT NULL,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL,
        width INTEGER DEFAULT 0,
        height INTEGER DEFAULT 0,
        role TEXT DEFAULT 'other',
        confidence REAL DEFAULT 1.0,
        source TEXT NOT NULL,  -- 'atspi', 'vlm', 'manual'
        last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        hit_count INTEGER DEFAULT 0,
        UNIQUE(app_name, element_label, resolution_w, resolution_h)
    );

Usage:
    from bantz.vision.spatial_cache import spatial_db

    spatial_db.init(db_path)  # call once at startup

    # Lookup (< 1ms)
    hit = spatial_db.lookup("firefox", "Send", 1920, 1080)

    # Store after successful find
    spatial_db.store("firefox", "Send", 1920, 1080,
                     x=1340, y=680, width=80, height=30,
                     source="atspi", confidence=1.0, role="push button")

    # Stats
    stats = spatial_db.stats()
"""
from __future__ import annotations

import logging
import sqlite3
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("bantz.vision.spatial_cache")

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ENTRIES = 1000
TTL_HOURS = 24
CONFIDENCE_DECAY_PER_DAY = 0.05

# Source → base confidence mapping
SOURCE_CONFIDENCE = {
    "atspi": 1.0,
    "vlm": 0.7,
    "manual": 0.9,
}


# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A cached UI element coordinate."""
    app_name: str
    element_label: str
    resolution_w: int
    resolution_h: int
    x: int
    y: int
    width: int
    height: int
    role: str
    confidence: float
    source: str
    last_verified: str
    hit_count: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def age_hours(self) -> float:
        """Hours since last verification."""
        try:
            verified = datetime.fromisoformat(self.last_verified)
            return (datetime.now() - verified).total_seconds() / 3600
        except Exception:
            return 999.0

    @property
    def effective_confidence(self) -> float:
        """Confidence decays 0.05 per day since last verification."""
        days = self.age_hours / 24.0
        decayed = self.confidence - (CONFIDENCE_DECAY_PER_DAY * days)
        return max(0.0, min(1.0, decayed))

    @property
    def is_expired(self) -> bool:
        """True if entry is older than TTL_HOURS."""
        return self.age_hours > TTL_HOURS

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "element_label": self.element_label,
            "resolution": f"{self.resolution_w}x{self.resolution_h}",
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "center": self.center,
            "role": self.role,
            "confidence": self.confidence,
            "effective_confidence": round(self.effective_confidence, 3),
            "source": self.source,
            "last_verified": self.last_verified,
            "hit_count": self.hit_count,
            "expired": self.is_expired,
        }


# ── Screen resolution detection ──────────────────────────────────────────

def get_screen_resolution() -> tuple[int, int]:
    """
    Detect current screen resolution.
    Returns (width, height), defaults to (1920, 1080) if detection fails.
    """
    # Try xrandr (X11)
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            import re
            match = re.search(r"current\s+(\d+)\s*x\s*(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try xdpyinfo (X11)
    try:
        result = subprocess.run(
            ["xdpyinfo"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            import re
            match = re.search(r"dimensions:\s+(\d+)x(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try wlr-randr (Wayland)
    try:
        result = subprocess.run(
            ["wlr-randr"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            import re
            match = re.search(r"(\d+)x(\d+)\s+px", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return (1920, 1080)


# ── SQLite Spatial Cache ──────────────────────────────────────────────────

class SpatialCacheDB:
    """
    Persistent SQLite-backed spatial cache for UI element coordinates.

    Resolution-aware: coordinates are keyed to the screen resolution at
    the time of capture, so a resolution change invalidates stale entries.

    Thread-safe with a lock around all writes.
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False

    def init(self, db_path: Path) -> None:
        """Initialize the spatial cache table.  Call once at startup."""
        if self._initialized:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self._initialized = True
        log.debug("Spatial cache initialized: %s", db_path)

    def _migrate(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS spatial_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                app_name        TEXT NOT NULL,
                element_label   TEXT NOT NULL,
                resolution_w    INTEGER NOT NULL,
                resolution_h    INTEGER NOT NULL,
                x               INTEGER NOT NULL,
                y               INTEGER NOT NULL,
                width           INTEGER DEFAULT 0,
                height          INTEGER DEFAULT 0,
                role            TEXT DEFAULT 'other',
                confidence      REAL DEFAULT 1.0,
                source          TEXT NOT NULL,
                last_verified   TEXT DEFAULT (datetime('now', 'localtime')),
                hit_count       INTEGER DEFAULT 0,
                UNIQUE(app_name, element_label, resolution_w, resolution_h)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spatial_app_label
            ON spatial_cache(app_name, element_label)
        """)

    # ── Lookup ────────────────────────────────────────────────────────────

    def lookup(
        self,
        app_name: str,
        element_label: str,
        resolution_w: int | None = None,
        resolution_h: int | None = None,
    ) -> Optional[CacheEntry]:
        """
        Look up a cached element coordinate.

        Returns CacheEntry if found and not expired, None otherwise.
        Automatically increments hit_count on success.
        """
        if not self._initialized or not self._conn:
            return None

        if resolution_w is None or resolution_h is None:
            resolution_w, resolution_h = get_screen_resolution()

        row = self._conn.execute(
            """SELECT * FROM spatial_cache
               WHERE app_name = ? AND element_label = ?
                 AND resolution_w = ? AND resolution_h = ?""",
            (app_name.lower(), element_label.lower(), resolution_w, resolution_h),
        ).fetchone()

        if row is None:
            return None

        entry = self._row_to_entry(row)

        # Check TTL
        if entry.is_expired:
            log.debug("Spatial cache expired: %s/%s", app_name, element_label)
            return None

        # Check effective confidence (too decayed = unreliable)
        if entry.effective_confidence < 0.3:
            log.debug("Spatial cache confidence too low: %s/%s (%.2f)",
                       app_name, element_label, entry.effective_confidence)
            return None

        # Increment hit count
        with self._lock:
            self._conn.execute(
                "UPDATE spatial_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (row["id"],),
            )

        entry.hit_count += 1
        return entry

    # ── Store ─────────────────────────────────────────────────────────────

    def store(
        self,
        app_name: str,
        element_label: str,
        resolution_w: int | None = None,
        resolution_h: int | None = None,
        *,
        x: int,
        y: int,
        width: int = 0,
        height: int = 0,
        role: str = "other",
        source: str = "atspi",
        confidence: float | None = None,
    ) -> None:
        """
        Store or update a cached element coordinate.

        If confidence is None, uses SOURCE_CONFIDENCE[source] default.
        Performs LRU eviction if at max capacity.
        """
        if not self._initialized or not self._conn:
            return

        if resolution_w is None or resolution_h is None:
            resolution_w, resolution_h = get_screen_resolution()

        if confidence is None:
            confidence = SOURCE_CONFIDENCE.get(source, 0.5)

        now = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            # Upsert
            self._conn.execute("""
                INSERT INTO spatial_cache
                    (app_name, element_label, resolution_w, resolution_h,
                     x, y, width, height, role, confidence, source, last_verified, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(app_name, element_label, resolution_w, resolution_h)
                DO UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    width = excluded.width,
                    height = excluded.height,
                    role = excluded.role,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    last_verified = excluded.last_verified,
                    hit_count = hit_count + 1
            """, (
                app_name.lower(), element_label.lower(),
                resolution_w, resolution_h,
                x, y, width, height, role, confidence, source, now,
            ))

            # LRU eviction if over max
            count = self._conn.execute(
                "SELECT COUNT(*) FROM spatial_cache"
            ).fetchone()[0]
            if count > MAX_ENTRIES:
                excess = count - MAX_ENTRIES
                self._conn.execute("""
                    DELETE FROM spatial_cache
                    WHERE id IN (
                        SELECT id FROM spatial_cache
                        ORDER BY last_verified ASC
                        LIMIT ?
                    )
                """, (excess,))
                log.debug("Evicted %d LRU entries from spatial cache", excess)

    # ── Invalidation ──────────────────────────────────────────────────────

    def invalidate_app(self, app_name: str) -> int:
        """Remove all cached entries for an application."""
        if not self._initialized or not self._conn:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM spatial_cache WHERE app_name = ?",
                (app_name.lower(),),
            )
            return cur.rowcount

    def invalidate_resolution(self, resolution_w: int, resolution_h: int) -> int:
        """Remove all entries for a specific resolution."""
        if not self._initialized or not self._conn:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM spatial_cache WHERE resolution_w = ? AND resolution_h = ?",
                (resolution_w, resolution_h),
            )
            return cur.rowcount

    def invalidate_all(self) -> int:
        """Clear the entire spatial cache."""
        if not self._initialized or not self._conn:
            return 0
        with self._lock:
            cur = self._conn.execute("DELETE FROM spatial_cache")
            return cur.rowcount

    def invalidate_expired(self) -> int:
        """Remove entries older than TTL_HOURS."""
        if not self._initialized or not self._conn:
            return 0
        cutoff = (datetime.now() - timedelta(hours=TTL_HOURS)).isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM spatial_cache WHERE last_verified < ?",
                (cutoff,),
            )
            return cur.rowcount

    # ── Verify (refresh timestamp) ────────────────────────────────────────

    def verify(
        self,
        app_name: str,
        element_label: str,
        resolution_w: int | None = None,
        resolution_h: int | None = None,
        *,
        x: int | None = None,
        y: int | None = None,
        confidence: float | None = None,
    ) -> bool:
        """
        Re-verify an existing cache entry (refresh last_verified).
        Optionally update coordinates if they changed.
        Returns True if entry existed and was updated.
        """
        if not self._initialized or not self._conn:
            return False

        if resolution_w is None or resolution_h is None:
            resolution_w, resolution_h = get_screen_resolution()

        now = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            if x is not None and y is not None:
                cur = self._conn.execute("""
                    UPDATE spatial_cache
                    SET x = ?, y = ?, last_verified = ?,
                        confidence = COALESCE(?, confidence)
                    WHERE app_name = ? AND element_label = ?
                      AND resolution_w = ? AND resolution_h = ?
                """, (x, y, now, confidence,
                      app_name.lower(), element_label.lower(),
                      resolution_w, resolution_h))
            else:
                cur = self._conn.execute("""
                    UPDATE spatial_cache
                    SET last_verified = ?,
                        confidence = COALESCE(?, confidence)
                    WHERE app_name = ? AND element_label = ?
                      AND resolution_w = ? AND resolution_h = ?
                """, (now, confidence,
                      app_name.lower(), element_label.lower(),
                      resolution_w, resolution_h))
            return cur.rowcount > 0

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cache statistics for the CLI command."""
        if not self._initialized or not self._conn:
            return {
                "total_entries": 0,
                "apps": {},
                "sources": {},
                "expired": 0,
                "total_hits": 0,
                "top_elements": [],
            }

        total = self._conn.execute(
            "SELECT COUNT(*) FROM spatial_cache"
        ).fetchone()[0]

        # Per-app counts
        apps = {}
        for row in self._conn.execute(
            "SELECT app_name, COUNT(*) as cnt FROM spatial_cache GROUP BY app_name ORDER BY cnt DESC"
        ):
            apps[row["app_name"]] = row["cnt"]

        # Per-source counts
        sources = {}
        for row in self._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM spatial_cache GROUP BY source ORDER BY cnt DESC"
        ):
            sources[row["source"]] = row["cnt"]

        # Expired count
        cutoff = (datetime.now() - timedelta(hours=TTL_HOURS)).isoformat(timespec="seconds")
        expired = self._conn.execute(
            "SELECT COUNT(*) FROM spatial_cache WHERE last_verified < ?",
            (cutoff,),
        ).fetchone()[0]

        # Total hits
        total_hits = self._conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM spatial_cache"
        ).fetchone()[0]

        # Top elements by hit count
        top = []
        for row in self._conn.execute(
            "SELECT app_name, element_label, hit_count, source, confidence "
            "FROM spatial_cache ORDER BY hit_count DESC LIMIT 10"
        ):
            top.append({
                "app": row["app_name"],
                "label": row["element_label"],
                "hits": row["hit_count"],
                "source": row["source"],
                "confidence": row["confidence"],
            })

        return {
            "total_entries": total,
            "max_entries": MAX_ENTRIES,
            "apps": apps,
            "sources": sources,
            "expired": expired,
            "total_hits": total_hits,
            "top_elements": top,
            "ttl_hours": TTL_HOURS,
        }

    def all_entries(self, app_name: str | None = None) -> list[CacheEntry]:
        """List all cache entries, optionally filtered by app."""
        if not self._initialized or not self._conn:
            return []

        if app_name:
            rows = self._conn.execute(
                "SELECT * FROM spatial_cache WHERE app_name = ? ORDER BY last_verified DESC",
                (app_name.lower(),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM spatial_cache ORDER BY last_verified DESC"
            ).fetchall()

        return [self._row_to_entry(row) for row in rows]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            app_name=row["app_name"],
            element_label=row["element_label"],
            resolution_w=row["resolution_w"],
            resolution_h=row["resolution_h"],
            x=row["x"],
            y=row["y"],
            width=row["width"],
            height=row["height"],
            role=row["role"],
            confidence=row["confidence"],
            source=row["source"],
            last_verified=row["last_verified"],
            hit_count=row["hit_count"],
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False


# Module-level singleton
spatial_db = SpatialCacheDB()
