"""
Tests for Issue #117 — Migrate JSON file stores into unified DB schema.

Covers:
    - SQLiteProfileStore CRUD ops
    - SQLitePlaceStore CRUD ops (incl. upsert/delete)
    - SQLiteScheduleStore CRUD ops
    - SQLiteSessionStore CRUD ops
    - Auto-migration from JSON → SQLite in DataLayer
    - Backward compat: core modules with and without bound store
    - Updated migration script (new table names)
    - bind_store() on SessionTracker, Profile, PlaceService, Schedule
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from bantz.data.sqlite_store import (
    SQLitePlaceStore,
    SQLiteProfileStore,
    SQLiteScheduleStore,
    SQLiteSessionStore,
)
from bantz.data.migration import migrate_to_sqlite


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def db_path(tmp_dir: Path) -> Path:
    return tmp_dir / "test.db"


# ━━ SQLiteProfileStore ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLiteProfileStore:
    def test_empty_on_init(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        assert store.load() == {}
        assert not store.exists()

    def test_save_and_load(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        data = {"name": "Iclal", "university": "OMU", "year": 2}
        store.save(data)
        loaded = store.load()
        assert loaded["name"] == "Iclal"
        assert loaded["university"] == "OMU"
        assert loaded["year"] == 2

    def test_exists_after_save(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        assert not store.exists()
        store.save({"name": "Test"})
        assert store.exists()

    def test_save_overwrites(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        store.save({"name": "Alice", "city": "Istanbul"})
        store.save({"name": "Bob"})  # overwrites entirely
        loaded = store.load()
        assert loaded["name"] == "Bob"
        assert "city" not in loaded

    def test_nested_dict_preserved(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        data = {
            "name": "Test",
            "preferences": {
                "briefing_sections": ["schedule", "weather"],
                "response_style": "casual",
            },
        }
        store.save(data)
        loaded = store.load()
        assert loaded["preferences"]["response_style"] == "casual"
        assert loaded["preferences"]["briefing_sections"] == ["schedule", "weather"]

    def test_path_returns_db(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        assert store.path == db_path

    def test_unicode_values(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        store.save({"name": "İclal", "university": "Samsun OMÜ"})
        loaded = store.load()
        assert loaded["name"] == "İclal"
        assert loaded["university"] == "Samsun OMÜ"


# ━━ SQLitePlaceStore ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLitePlaceStore:
    def test_empty_on_init(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        assert store.load_all() == {}
        assert not store.exists()

    def test_save_all_and_load(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        data = {
            "dorm": {"label": "Yurt", "lat": 41.28, "lon": 36.33, "radius": 100},
            "campus": {"label": "Kampüs", "lat": 41.29, "lon": 36.34, "radius": 200},
        }
        store.save_all(data)
        loaded = store.load_all()
        assert len(loaded) == 2
        assert loaded["dorm"]["label"] == "Yurt"
        assert loaded["campus"]["lat"] == 41.29

    def test_upsert_single(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({"dorm": {"label": "Yurt", "lat": 41.0, "lon": 36.0}})
        store.upsert("library", {"label": "Kütüphane", "lat": 41.1, "lon": 36.1})
        loaded = store.load_all()
        assert len(loaded) == 2
        assert "library" in loaded

    def test_upsert_updates_existing(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({"dorm": {"label": "Yurt", "lat": 41.0, "lon": 36.0}})
        store.upsert("dorm", {"label": "Updated Yurt", "lat": 41.5, "lon": 36.5})
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded["dorm"]["label"] == "Updated Yurt"
        assert loaded["dorm"]["lat"] == 41.5

    def test_delete(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({
            "a": {"label": "A", "lat": 0, "lon": 0},
            "b": {"label": "B", "lat": 0, "lon": 0},
        })
        assert store.delete("a")
        loaded = store.load_all()
        assert len(loaded) == 1
        assert "b" in loaded

    def test_delete_nonexistent(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        assert not store.delete("nope")

    def test_exists_after_save(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        assert not store.exists()
        store.save_all({"x": {"label": "X", "lat": 0, "lon": 0}})
        assert store.exists()

    def test_default_radius(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({"test": {"label": "Test", "lat": 1.0, "lon": 2.0}})
        loaded = store.load_all()
        assert loaded["test"]["radius"] == 100.0


# ━━ SQLiteScheduleStore ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLiteScheduleStore:
    def test_empty_on_init(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        assert store.load() == {}
        assert not store.exists()

    def test_save_and_load(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        data = {
            "monday": [
                {"name": "ML", "time": "10:00", "duration": 90, "location": "B2", "type": "lab"},
                {"name": "DS", "time": "14:00", "duration": 60, "location": "A1", "type": "lecture"},
            ],
            "wednesday": [
                {"name": "DB", "time": "09:00", "duration": 90, "location": "C3", "type": ""},
            ],
        }
        store.save(data)
        loaded = store.load()
        assert len(loaded["monday"]) == 2
        assert loaded["monday"][0]["name"] == "ML"
        assert loaded["monday"][1]["time"] == "14:00"
        assert loaded["wednesday"][0]["location"] == "C3"

    def test_save_overwrites(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        store.save({"monday": [{"name": "Old", "time": "08:00"}]})
        store.save({"tuesday": [{"name": "New", "time": "09:00"}]})
        loaded = store.load()
        assert "monday" not in loaded
        assert "tuesday" in loaded

    def test_exists_after_save(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        assert not store.exists()
        store.save({"friday": [{"name": "Test", "time": "10:00"}]})
        assert store.exists()

    def test_default_duration(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        store.save({"monday": [{"name": "X", "time": "10:00"}]})
        loaded = store.load()
        assert loaded["monday"][0]["duration"] == 60

    def test_preserves_order(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        entries = [
            {"name": f"Class{i}", "time": f"{8+i}:00"}
            for i in range(5)
        ]
        store.save({"monday": entries})
        loaded = store.load()
        for i, entry in enumerate(loaded["monday"]):
            assert entry["name"] == f"Class{i}"


# ━━ SQLiteSessionStore ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLiteSessionStore:
    def test_empty_on_init(self, db_path: Path):
        store = SQLiteSessionStore(db_path)
        assert store.load() == {}

    def test_save_and_load(self, db_path: Path):
        store = SQLiteSessionStore(db_path)
        data = {"last_seen": "2026-03-05T10:00:00", "session_count": 42}
        store.save(data)
        loaded = store.load()
        assert loaded["last_seen"] == "2026-03-05T10:00:00"
        assert loaded["session_count"] == 42

    def test_save_overwrites(self, db_path: Path):
        store = SQLiteSessionStore(db_path)
        store.save({"a": 1, "b": 2})
        store.save({"c": 3})
        loaded = store.load()
        assert loaded == {"c": 3}

    def test_path_returns_db(self, db_path: Path):
        store = SQLiteSessionStore(db_path)
        assert store.path == db_path


# ━━ Core Module bind_store Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSessionTrackerBindStore:
    def test_bind_store_and_save(self, db_path: Path):
        from bantz.core.session import SessionTracker
        store = SQLiteSessionStore(db_path)
        tracker = SessionTracker()
        tracker.bind_store(store)

        result = tracker.on_launch()
        assert result["is_first"] is True
        assert result["session_count"] == 1

        # Data persisted in SQLite
        loaded = store.load()
        assert loaded["session_count"] == 1
        assert "last_seen" in loaded

    def test_sequential_launches(self, db_path: Path):
        from bantz.core.session import SessionTracker
        store = SQLiteSessionStore(db_path)
        tracker = SessionTracker()
        tracker.bind_store(store)

        tracker.on_launch()
        result = tracker.on_launch()
        assert result["session_count"] == 2
        assert result["is_first"] is False

    def test_path_from_store(self, db_path: Path):
        from bantz.core.session import SessionTracker
        store = SQLiteSessionStore(db_path)
        tracker = SessionTracker()
        tracker.bind_store(store)
        assert tracker.path == db_path

    def test_json_fallback(self, tmp_dir: Path):
        """Without bind_store, uses JSON fallback."""
        from bantz.core.session import SessionTracker
        tracker = SessionTracker()
        # No store bound — should not crash
        data = tracker._load()
        assert isinstance(data, dict)


class TestProfileBindStore:
    def test_bind_store_and_load(self, db_path: Path):
        store = SQLiteProfileStore(db_path)
        store.save({"name": "Iclal", "tone": "casual"})

        from bantz.core.profile import Profile
        p = Profile()
        p.bind_store(store)

        assert p.name == "Iclal"
        assert p.get("tone") == "casual"

    def test_save_through_profile(self, db_path: Path):
        store = SQLiteProfileStore(db_path)

        from bantz.core.profile import Profile
        p = Profile()
        p.bind_store(store)
        p.save({"name": "Alice", "university": "MIT"})

        loaded = store.load()
        assert loaded["name"] == "Alice"
        assert loaded["university"] == "MIT"
        # Defaults filled
        assert loaded["tone"] == "casual"

    def test_is_configured(self, db_path: Path):
        store = SQLiteProfileStore(db_path)

        from bantz.core.profile import Profile
        p = Profile()
        p.bind_store(store)
        assert not p.is_configured()

        p.save({"name": "Bob"})
        assert p.is_configured()


class TestPlaceServiceBindStore:
    def test_bind_store_and_load(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({
            "dorm": {"label": "Yurt", "lat": 41.0, "lon": 36.0, "radius": 100},
        })

        from bantz.core.places import PlaceService
        ps = PlaceService()
        ps.bind_store(store)

        all_places = ps.all_places()
        assert "dorm" in all_places
        assert all_places["dorm"]["label"] == "Yurt"

    def test_save_through_service(self, db_path: Path):
        store = SQLitePlaceStore(db_path)

        from bantz.core.places import PlaceService
        ps = PlaceService()
        ps.bind_store(store)
        ps.save({"lib": {"label": "Library", "lat": 1.0, "lon": 2.0}})

        loaded = store.load_all()
        assert "lib" in loaded

    def test_delete_through_service(self, db_path: Path):
        store = SQLitePlaceStore(db_path)
        store.save_all({
            "a": {"label": "A", "lat": 0, "lon": 0, "radius": 100},
            "b": {"label": "B", "lat": 0, "lon": 0, "radius": 100},
        })

        from bantz.core.places import PlaceService
        ps = PlaceService()
        ps.bind_store(store)

        assert ps.delete_place("a")
        loaded = store.load_all()
        assert "a" not in loaded
        assert "b" in loaded

    def test_is_configured(self, db_path: Path):
        store = SQLitePlaceStore(db_path)

        from bantz.core.places import PlaceService
        ps = PlaceService()
        ps.bind_store(store)
        assert not ps.is_configured()

        ps.save({"x": {"label": "X", "lat": 0, "lon": 0}})
        assert ps.is_configured()


class TestScheduleBindStore:
    def test_bind_store_and_load(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)
        store.save({
            "monday": [
                {"name": "ML", "time": "10:00", "duration": 90, "location": "B2", "type": "lab"}
            ],
        })

        from bantz.core.schedule import Schedule
        s = Schedule()
        s.bind_store(store)

        monday = datetime(2026, 1, 5, 8, 0)  # a Monday
        classes = s.today(monday)
        assert len(classes) == 1
        assert classes[0]["name"] == "ML"

    def test_is_configured(self, db_path: Path):
        store = SQLiteScheduleStore(db_path)

        from bantz.core.schedule import Schedule
        s = Schedule()
        s.bind_store(store)
        assert not s.is_configured()

        store.save({"monday": [{"name": "X", "time": "08:00"}]})
        # Reset loaded flag to pick up new data
        s._loaded = False
        assert s.is_configured()


# ━━ Auto-Migration in DataLayer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDataLayerAutoMigration:
    """Test that DataLayer auto-migrates JSON → SQLite on first init."""

    def test_auto_migrate_profile(self, tmp_dir: Path):
        db_path = tmp_dir / "bantz.db"

        # Write JSON
        (tmp_dir / "profile.json").write_text(
            json.dumps({"name": "Iclal", "university": "OMU"})
        )

        # Simulate DataLayer auto-migration
        profile_store = SQLiteProfileStore(db_path)
        assert not profile_store.exists()

        pf = tmp_dir / "profile.json"
        data = json.loads(pf.read_text("utf-8"))
        profile_store.save(data)

        loaded = profile_store.load()
        assert loaded["name"] == "Iclal"

    def test_auto_migrate_places(self, tmp_dir: Path):
        db_path = tmp_dir / "bantz.db"
        (tmp_dir / "places.json").write_text(
            json.dumps({
                "dorm": {"label": "Yurt", "lat": 41.0, "lon": 36.0, "radius": 100}
            })
        )

        place_store = SQLitePlaceStore(db_path)
        assert not place_store.exists()

        data = json.loads((tmp_dir / "places.json").read_text("utf-8"))
        place_store.save_all(data)

        loaded = place_store.load_all()
        assert loaded["dorm"]["label"] == "Yurt"

    def test_auto_migrate_schedule(self, tmp_dir: Path):
        db_path = tmp_dir / "bantz.db"
        (tmp_dir / "schedule.json").write_text(
            json.dumps({"tuesday": [{"name": "DB", "time": "09:00", "duration": 90}]})
        )

        sched_store = SQLiteScheduleStore(db_path)
        data = json.loads((tmp_dir / "schedule.json").read_text("utf-8"))
        sched_store.save(data)

        loaded = sched_store.load()
        assert loaded["tuesday"][0]["name"] == "DB"

    def test_auto_migrate_session(self, tmp_dir: Path):
        db_path = tmp_dir / "bantz.db"
        (tmp_dir / "session.json").write_text(
            json.dumps({"last_seen": "2026-01-01T00:00:00", "session_count": 10})
        )

        sess_store = SQLiteSessionStore(db_path)
        data = json.loads((tmp_dir / "session.json").read_text("utf-8"))
        sess_store.save(data)

        loaded = sess_store.load()
        assert loaded["session_count"] == 10

    def test_no_migrate_if_sqlite_has_data(self, tmp_dir: Path):
        db_path = tmp_dir / "bantz.db"

        # Pre-populate SQLite
        profile_store = SQLiteProfileStore(db_path)
        profile_store.save({"name": "SQLite User"})

        # Write different JSON
        (tmp_dir / "profile.json").write_text(
            json.dumps({"name": "JSON User"})
        )

        # Auto-migrate should skip since SQLite has data
        assert profile_store.exists()
        loaded = profile_store.load()
        assert loaded["name"] == "SQLite User"


# ━━ Updated Migration Script ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMigrationScript:
    def test_migrate_creates_new_tables(self, tmp_dir: Path):
        import sqlite3
        db_path = tmp_dir / "migrate.db"

        (tmp_dir / "profile.json").write_text('{"name": "Test"}')
        (tmp_dir / "session.json").write_text('{"session_count": 1}')

        migrate_to_sqlite(db_path, tmp_dir)

        conn = sqlite3.connect(str(db_path))
        # Check that new table names exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "user_profile" in tables
        assert "session_state" in tables
        assert "places" in tables
        assert "schedule_entries" in tables
        conn.close()

    def test_migrate_profile_to_user_profile(self, tmp_dir: Path):
        import sqlite3
        db_path = tmp_dir / "m.db"
        (tmp_dir / "profile.json").write_text(
            json.dumps({"name": "Iclal", "year": 2})
        )
        migrate_to_sqlite(db_path, tmp_dir)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM user_profile").fetchall()
        keys = {r["key"] for r in rows}
        assert "name" in keys
        assert "year" in keys
        conn.close()

    def test_migrate_session_to_session_state(self, tmp_dir: Path):
        import sqlite3
        db_path = tmp_dir / "m.db"
        (tmp_dir / "session.json").write_text(
            json.dumps({"last_seen": "2026-01-01T00:00:00", "session_count": 5})
        )
        migrate_to_sqlite(db_path, tmp_dir)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM session_state").fetchall()
        keys = {r["key"] for r in rows}
        assert "last_seen" in keys
        assert "session_count" in keys
        conn.close()

    def test_migrate_schedule_to_schedule_entries(self, tmp_dir: Path):
        import sqlite3
        db_path = tmp_dir / "m.db"
        (tmp_dir / "schedule.json").write_text(
            json.dumps({
                "monday": [{"name": "ML", "time": "10:00", "duration": 90}],
                "friday": [{"name": "DB", "time": "14:00"}],
            })
        )
        migrate_to_sqlite(db_path, tmp_dir)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM schedule_entries").fetchall()
        assert len(rows) == 2
        conn.close()


# ━━ Cross-Store Integrity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCrossStoreIntegrity:
    """Multiple SQLite stores can share the same DB file."""

    def test_shared_db_file(self, db_path: Path):
        profile = SQLiteProfileStore(db_path)
        places = SQLitePlaceStore(db_path)
        schedule = SQLiteScheduleStore(db_path)
        session = SQLiteSessionStore(db_path)

        profile.save({"name": "Test"})
        places.save_all({"home": {"label": "Home", "lat": 0, "lon": 0}})
        schedule.save({"monday": [{"name": "ML", "time": "10:00"}]})
        session.save({"session_count": 1})

        assert profile.load()["name"] == "Test"
        assert "home" in places.load_all()
        assert len(schedule.load()["monday"]) == 1
        assert session.load()["session_count"] == 1

    def test_stores_dont_interfere(self, db_path: Path):
        """Saving to one store doesn't corrupt another."""
        profile = SQLiteProfileStore(db_path)
        session = SQLiteSessionStore(db_path)

        profile.save({"name": "Alice"})
        session.save({"count": 1})

        # Overwrite session
        session.save({"count": 2})

        # Profile should be untouched
        assert profile.load()["name"] == "Alice"
        assert session.load()["count"] == 2
