"""
Tests for Bantz v3 — Spatial Cache (#121)

Coverage:
  - SQLite table creation and schema
  - Store / lookup with resolution awareness
  - TTL-based expiration
  - Confidence decay over time
  - LRU eviction at max entries
  - Hit-count tracking
  - Invalidation (app, resolution, all, expired)
  - Verify (refresh) entries
  - Stats output
  - CacheEntry properties
  - Screen resolution detection (mocked)
  - AccessibilityTool spatial integration
  - CLI --cache-stats
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bantz.data.connection_pool import get_pool, SQLitePool
from bantz.vision.spatial_cache import (
    SpatialCacheDB,
    CacheEntry,
    MAX_ENTRIES,
    TTL_HOURS,
    CONFIDENCE_DECAY_PER_DAY,
    SOURCE_CONFIDENCE,
    get_screen_resolution,
    spatial_db,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_pool():
    yield
    SQLitePool.reset()


@pytest.fixture
def db(tmp_path: Path) -> SpatialCacheDB:
    """Fresh spatial cache DB per test."""
    cache = SpatialCacheDB()
    cache.init(tmp_path / "test.db")
    return cache


@pytest.fixture
def populated_db(db: SpatialCacheDB) -> SpatialCacheDB:
    """DB with some entries pre-populated."""
    db.store("firefox", "send", 1920, 1080, x=1340, y=680, width=80, height=30,
             role="push button", source="atspi")
    db.store("firefox", "url bar", 1920, 1080, x=960, y=50, width=800, height=30,
             role="text", source="atspi")
    db.store("vscode", "save", 1920, 1080, x=50, y=10, width=60, height=20,
             role="menu item", source="vlm", confidence=0.65)
    db.store("firefox", "send", 2560, 1440, x=1700, y=900, width=100, height=40,
             role="push button", source="atspi")
    return db


# ── Schema / Init ─────────────────────────────────────────────────────────

class TestSchema:
    def test_table_created(self, db: SpatialCacheDB):
        """spatial_cache table exists with correct columns."""
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='spatial_cache'"
            ).fetchall()
        assert len(rows) == 1

    def test_index_created(self, db: SpatialCacheDB):
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_spatial_app_label'"
            ).fetchall()
        assert len(rows) == 1

    def test_unique_constraint(self, db: SpatialCacheDB):
        """UNIQUE(app_name, element_label, resolution_w, resolution_h)."""
        db.store("app", "button", 1920, 1080, x=100, y=200, source="atspi")
        # Second store with same key should upsert, not duplicate
        db.store("app", "button", 1920, 1080, x=150, y=250, source="vlm")
        with get_pool().connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM spatial_cache").fetchone()[0]
        assert count == 1

    def test_wal_mode(self, db: SpatialCacheDB):
        with get_pool().connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_double_init_safe(self, tmp_path: Path):
        cache = SpatialCacheDB()
        cache.init(tmp_path / "test.db")
        cache.init(tmp_path / "test.db")  # should be no-op
        assert cache._initialized


# ── Store / Lookup ────────────────────────────────────────────────────────

class TestStoreLookup:
    def test_store_and_lookup(self, db: SpatialCacheDB):
        db.store("firefox", "Send", 1920, 1080, x=1340, y=680, width=80, height=30,
                 role="push button", source="atspi")

        entry = db.lookup("firefox", "Send", 1920, 1080)
        assert entry is not None
        assert entry.x == 1340
        assert entry.y == 680
        assert entry.width == 80
        assert entry.height == 30
        assert entry.role == "push button"
        assert entry.source == "atspi"
        assert entry.confidence == 1.0

    def test_case_insensitive(self, db: SpatialCacheDB):
        db.store("Firefox", "SEND", 1920, 1080, x=100, y=200, source="atspi")
        entry = db.lookup("firefox", "send", 1920, 1080)
        assert entry is not None

    def test_lookup_miss(self, db: SpatialCacheDB):
        entry = db.lookup("nonexistent", "button", 1920, 1080)
        assert entry is None

    def test_resolution_awareness(self, populated_db: SpatialCacheDB):
        """Same element at different resolutions have different coordinates."""
        e1 = populated_db.lookup("firefox", "send", 1920, 1080)
        e2 = populated_db.lookup("firefox", "send", 2560, 1440)
        assert e1 is not None and e2 is not None
        assert e1.x != e2.x  # different positions
        assert e1.resolution_w == 1920
        assert e2.resolution_w == 2560

    def test_resolution_mismatch_returns_none(self, populated_db: SpatialCacheDB):
        """Lookup with wrong resolution returns nothing."""
        entry = populated_db.lookup("firefox", "send", 3840, 2160)
        assert entry is None

    def test_upsert_updates_coordinates(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        db.store("app", "btn", 1920, 1080, x=150, y=250, source="vlm")
        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry.x == 150
        assert entry.y == 250

    def test_default_confidence_by_source(self, db: SpatialCacheDB):
        db.store("app", "a", 1920, 1080, x=0, y=0, source="atspi")
        db.store("app", "b", 1920, 1080, x=0, y=0, source="vlm")
        db.store("app", "c", 1920, 1080, x=0, y=0, source="manual")
        assert db.lookup("app", "a", 1920, 1080).confidence == 1.0
        assert db.lookup("app", "b", 1920, 1080).confidence == 0.7
        assert db.lookup("app", "c", 1920, 1080).confidence == 0.9

    @patch("bantz.vision.spatial_cache.get_screen_resolution", return_value=(1920, 1080))
    def test_auto_resolution(self, mock_res, db: SpatialCacheDB):
        """If resolution not specified, auto-detect is used."""
        db.store("app", "btn", x=100, y=200, source="atspi")
        entry = db.lookup("app", "btn")
        assert entry is not None
        assert entry.resolution_w == 1920
        assert entry.resolution_h == 1080

    def test_not_initialized_returns_none(self):
        cache = SpatialCacheDB()
        # Not initialized
        assert cache.lookup("app", "btn", 1920, 1080) is None
        cache.store("app", "btn", 1920, 1080, x=0, y=0, source="atspi")  # no error


# ── Hit Count ─────────────────────────────────────────────────────────────

class TestHitCount:
    def test_hit_count_increments(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        e1 = db.lookup("app", "btn", 1920, 1080)
        assert e1.hit_count == 1  # first lookup increments from 0
        e2 = db.lookup("app", "btn", 1920, 1080)
        assert e2.hit_count == 2

    def test_store_resets_hit_on_upsert(self, db: SpatialCacheDB):
        """Upsert increments hit_count via ON CONFLICT."""
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        db.lookup("app", "btn", 1920, 1080)  # hit_count=1
        db.lookup("app", "btn", 1920, 1080)  # hit_count=2
        # Re-store (upsert) → increments by 1
        db.store("app", "btn", 1920, 1080, x=150, y=250, source="vlm")
        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry.hit_count >= 1


# ── TTL / Expiration ──────────────────────────────────────────────────────

class TestTTL:
    def test_fresh_entry_not_expired(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        entry = db.lookup("app", "btn", 1920, 1080)
        assert not entry.is_expired

    def test_old_entry_expired(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        # Manually age the entry
        old_time = (datetime.now() - timedelta(hours=TTL_HOURS + 1)).isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "UPDATE spatial_cache SET last_verified = ?", (old_time,)
            )
        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry is None  # expired entries return None

    def test_invalidate_expired(self, db: SpatialCacheDB):
        db.store("app", "a", 1920, 1080, x=0, y=0, source="atspi")
        db.store("app", "b", 1920, 1080, x=0, y=0, source="atspi")
        # Age one entry
        old = (datetime.now() - timedelta(hours=TTL_HOURS + 1)).isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute(
                "UPDATE spatial_cache SET last_verified = ? WHERE element_label = 'a'",
                (old,),
            )
        removed = db.invalidate_expired()
        assert removed == 1
        assert db.lookup("app", "a", 1920, 1080) is None
        assert db.lookup("app", "b", 1920, 1080) is not None


# ── Confidence Decay ──────────────────────────────────────────────────────

class TestConfidenceDecay:
    def test_fresh_no_decay(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=0, y=0, source="atspi")
        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry.effective_confidence >= 0.95  # minimal decay

    def test_decay_after_days(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=0, y=0, source="atspi")
        # Age by 10 days (within TTL since we're 10*24 = 240h > 24h, but let's use 12h = 0.5 days)
        aging = (datetime.now() - timedelta(hours=12)).isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute("UPDATE spatial_cache SET last_verified = ?", (aging,))

        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry is not None
        # 0.5 days * 0.05/day = 0.025 decay → ~0.975
        assert 0.95 < entry.effective_confidence < 1.0

    def test_low_confidence_skipped(self, db: SpatialCacheDB):
        """Entries with effective confidence < 0.3 are not returned."""
        db.store("app", "btn", 1920, 1080, x=0, y=0, source="vlm", confidence=0.35)
        # Age by 20h (0.83 days → decay 0.042) → effective ≈ 0.308, still above
        aging = (datetime.now() - timedelta(hours=20)).isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute("UPDATE spatial_cache SET last_verified = ?", (aging,))
        entry = db.lookup("app", "btn", 1920, 1080)
        # Should still be above 0.3
        assert entry is not None


# ── LRU Eviction ──────────────────────────────────────────────────────────

class TestLRUEviction:
    def test_eviction_at_max_entries(self, db: SpatialCacheDB):
        """Old entries evicted when max is exceeded."""
        # Insert MAX_ENTRIES + 10
        for i in range(MAX_ENTRIES + 10):
            db.store("app", f"btn_{i}", 1920, 1080, x=i, y=i, source="atspi")

        with get_pool().connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM spatial_cache").fetchone()[0]
        assert count <= MAX_ENTRIES

    def test_oldest_evicted_first(self, db: SpatialCacheDB):
        """The oldest entries are evicted (LRU by last_verified)."""
        # Insert a few with staggered timestamps
        base = datetime.now() - timedelta(hours=10)
        with get_pool().connection(write=True) as conn:
            for i in range(5):
                ts = (base + timedelta(minutes=i)).isoformat(timespec="seconds")
                conn.execute("""
                    INSERT INTO spatial_cache
                        (app_name, element_label, resolution_w, resolution_h,
                         x, y, source, last_verified)
                    VALUES (?, ?, 1920, 1080, ?, ?, 'atspi', ?)
                """, (f"app", f"btn_{i}", i, i, ts))

        # Now fill to max + overflow
        for i in range(5, MAX_ENTRIES + 2):
            db.store("app", f"btn_{i}", 1920, 1080, x=i, y=i, source="atspi")

        with get_pool().connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM spatial_cache").fetchone()[0]
        assert count <= MAX_ENTRIES

        # The oldest (btn_0) should be evicted
        entry = db.lookup("app", "btn_0", 1920, 1080)
        assert entry is None


# ── Invalidation ──────────────────────────────────────────────────────────

class TestInvalidation:
    def test_invalidate_app(self, populated_db: SpatialCacheDB):
        removed = populated_db.invalidate_app("firefox")
        assert removed >= 2  # two firefox entries at different resolutions
        assert populated_db.lookup("firefox", "send", 1920, 1080) is None
        assert populated_db.lookup("vscode", "save", 1920, 1080) is not None

    def test_invalidate_resolution(self, populated_db: SpatialCacheDB):
        removed = populated_db.invalidate_resolution(2560, 1440)
        assert removed >= 1
        assert populated_db.lookup("firefox", "send", 2560, 1440) is None
        assert populated_db.lookup("firefox", "send", 1920, 1080) is not None

    def test_invalidate_all(self, populated_db: SpatialCacheDB):
        removed = populated_db.invalidate_all()
        assert removed >= 4
        with get_pool().connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM spatial_cache").fetchone()[0]
        assert count == 0

    def test_invalidate_not_initialized(self):
        cache = SpatialCacheDB()
        assert cache.invalidate_all() == 0
        assert cache.invalidate_app("x") == 0
        assert cache.invalidate_resolution(1920, 1080) == 0
        assert cache.invalidate_expired() == 0


# ── Verify ────────────────────────────────────────────────────────────────

class TestVerify:
    def test_verify_refreshes_timestamp(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        # Age it
        old = (datetime.now() - timedelta(hours=20)).isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute("UPDATE spatial_cache SET last_verified = ?", (old,))

        result = db.verify("app", "btn", 1920, 1080)
        assert result is True

        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry is not None
        assert entry.age_hours < 1  # just refreshed

    def test_verify_updates_coordinates(self, db: SpatialCacheDB):
        db.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        db.verify("app", "btn", 1920, 1080, x=150, y=250)
        entry = db.lookup("app", "btn", 1920, 1080)
        assert entry.x == 150
        assert entry.y == 250

    def test_verify_nonexistent_returns_false(self, db: SpatialCacheDB):
        assert db.verify("nonexistent", "btn", 1920, 1080) is False

    def test_verify_not_initialized(self):
        cache = SpatialCacheDB()
        assert cache.verify("app", "btn") is False


# ── Stats ─────────────────────────────────────────────────────────────────

class TestStats:
    def test_empty_stats(self, db: SpatialCacheDB):
        stats = db.stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["expired"] == 0
        assert stats["apps"] == {}
        assert stats["top_elements"] == []

    def test_populated_stats(self, populated_db: SpatialCacheDB):
        # Generate some hits
        populated_db.lookup("firefox", "send", 1920, 1080)
        populated_db.lookup("firefox", "send", 1920, 1080)
        populated_db.lookup("vscode", "save", 1920, 1080)

        stats = populated_db.stats()
        assert stats["total_entries"] == 4
        assert stats["total_hits"] >= 3
        assert "firefox" in stats["apps"]
        assert "atspi" in stats["sources"]
        assert len(stats["top_elements"]) > 0

    def test_not_initialized_stats(self):
        cache = SpatialCacheDB()
        stats = cache.stats()
        assert stats["total_entries"] == 0


# ── CacheEntry ────────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_center(self):
        entry = CacheEntry(
            app_name="app", element_label="btn",
            resolution_w=1920, resolution_h=1080,
            x=100, y=200, width=80, height=30,
            role="button", confidence=1.0, source="atspi",
            last_verified=datetime.now().isoformat(), hit_count=0,
        )
        assert entry.center == (140, 215)

    def test_to_dict(self):
        now = datetime.now().isoformat(timespec="seconds")
        entry = CacheEntry(
            app_name="firefox", element_label="send",
            resolution_w=1920, resolution_h=1080,
            x=100, y=200, width=80, height=30,
            role="button", confidence=1.0, source="atspi",
            last_verified=now, hit_count=5,
        )
        d = entry.to_dict()
        assert d["app_name"] == "firefox"
        assert d["resolution"] == "1920x1080"
        assert d["center"] == (140, 215)
        assert d["hit_count"] == 5
        assert "effective_confidence" in d
        assert "expired" in d

    def test_age_hours(self):
        old = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
        entry = CacheEntry(
            app_name="app", element_label="btn",
            resolution_w=1920, resolution_h=1080,
            x=0, y=0, width=0, height=0,
            role="", confidence=1.0, source="atspi",
            last_verified=old, hit_count=0,
        )
        assert 4.9 < entry.age_hours < 5.1

    def test_effective_confidence_decay(self):
        old = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
        entry = CacheEntry(
            app_name="app", element_label="btn",
            resolution_w=1920, resolution_h=1080,
            x=0, y=0, width=0, height=0,
            role="", confidence=1.0, source="atspi",
            last_verified=old, hit_count=0,
        )
        # 2 days * 0.05/day = 0.10 decay → ~0.90
        assert 0.85 < entry.effective_confidence < 0.95


# ── all_entries ───────────────────────────────────────────────────────────

class TestAllEntries:
    def test_all_entries(self, populated_db: SpatialCacheDB):
        entries = populated_db.all_entries()
        assert len(entries) == 4

    def test_all_entries_by_app(self, populated_db: SpatialCacheDB):
        entries = populated_db.all_entries("firefox")
        assert len(entries) >= 2  # 2 resolutions + url bar
        for e in entries:
            assert e.app_name == "firefox"

    def test_all_entries_empty(self, db: SpatialCacheDB):
        entries = db.all_entries()
        assert len(entries) == 0

    def test_all_entries_not_initialized(self):
        cache = SpatialCacheDB()
        assert cache.all_entries() == []


# ── Screen resolution detection ───────────────────────────────────────────

class TestResolutionDetection:
    @patch("subprocess.run")
    def test_xrandr_detection(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Screen 0: minimum 8 x 8, current 1920 x 1080, maximum 32767 x 32767",
        )
        w, h = get_screen_resolution()
        assert w == 1920
        assert h == 1080

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_fallback_resolution(self, mock_run):
        w, h = get_screen_resolution()
        assert w == 1920
        assert h == 1080


# ── Close / Cleanup ──────────────────────────────────────────────────────

class TestClose:
    def test_close_and_reinit(self, tmp_path: Path):
        cache = SpatialCacheDB()
        cache.init(tmp_path / "test.db")
        cache.store("app", "btn", 1920, 1080, x=100, y=200, source="atspi")
        cache.close()
        assert not cache._initialized

        # Re-init should work
        cache.init(tmp_path / "test.db")
        entry = cache.lookup("app", "btn", 1920, 1080)
        assert entry is not None  # data persisted


# ── Module singleton ──────────────────────────────────────────────────────

class TestSingleton:
    def test_module_singleton_exists(self):
        from bantz.vision.spatial_cache import spatial_db
        assert isinstance(spatial_db, SpatialCacheDB)


# ── Constants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_max_entries(self):
        assert MAX_ENTRIES == 1000

    def test_ttl_hours(self):
        assert TTL_HOURS == 24

    def test_confidence_decay(self):
        assert CONFIDENCE_DECAY_PER_DAY == 0.05

    def test_source_confidence_mapping(self):
        assert SOURCE_CONFIDENCE["atspi"] == 1.0
        assert SOURCE_CONFIDENCE["vlm"] == 0.7
        assert SOURCE_CONFIDENCE["manual"] == 0.9
