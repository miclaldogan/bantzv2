"""
Tests for the Bantz v3 Data Access Layer (Issue #115).

Covers:
    - Pydantic data models
    - SQLiteConversationStore
    - SQLiteReminderStore
    - JSON file stores (profile, places, schedule, session)
    - Migration utility (validate + migrate)
    - DataLayer integration
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bantz.data.models import (
    Conversation,
    Message,
    Place,
    Reminder,
    ScheduleEntry,
    SessionInfo,
    UserProfile,
)
from bantz.data.sqlite_store import SQLiteConversationStore, SQLiteReminderStore
from bantz.data.json_store import (
    JSONPlaceStore,
    JSONProfileStore,
    JSONScheduleStore,
    JSONSessionStore,
)
from bantz.data.migration import migrate_to_sqlite, validate_json_files


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def db_path(tmp_dir: Path) -> Path:
    return tmp_dir / "test.db"


@pytest.fixture
def conv_store(db_path: Path) -> SQLiteConversationStore:
    store = SQLiteConversationStore()
    store.init(db_path)
    store.new_session()
    return store


@pytest.fixture
def reminder_store(db_path: Path) -> SQLiteReminderStore:
    store = SQLiteReminderStore()
    store.init(db_path)
    return store


# ━━ Model Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModels:
    def test_message_defaults(self):
        m = Message(role="user", content="hello")
        assert m.id == 0
        assert m.role == "user"
        assert m.content == "hello"
        assert m.tool_used is None
        assert isinstance(m.created_at, datetime)

    def test_conversation_defaults(self):
        c = Conversation()
        assert c.id == 0
        assert c.message_count == 0

    def test_reminder_model(self):
        r = Reminder(title="call dentist", fire_at=datetime(2026, 3, 10, 15, 0))
        assert r.repeat == "none"
        assert r.fired is False
        assert r.trigger_place is None

    def test_place_model(self):
        p = Place(key="dorm", label="Yurt")
        assert p.radius == 100.0
        assert p.lat == 0.0

    def test_schedule_entry_model(self):
        e = ScheduleEntry(name="Makine Öğrenmesi", time="10:00")
        assert e.duration == 60
        assert e.location == ""

    def test_user_profile_model(self):
        u = UserProfile(name="Iclal", university="OMU")
        assert u.pronoun == "casual"
        assert u.preferences == {}

    def test_session_info_model(self):
        s = SessionInfo()
        assert s.is_first is False
        assert s.session_count == 0

    def test_message_serialization(self):
        m = Message(role="assistant", content="hi", tool_used="weather")
        d = m.model_dump()
        assert d["role"] == "assistant"
        assert d["tool_used"] == "weather"


# ━━ SQLite Conversation Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLiteConversationStore:
    def test_new_session(self, conv_store: SQLiteConversationStore):
        sid = conv_store.session_id
        assert sid is not None
        assert sid > 0

    def test_add_and_context(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "hello")
        conv_store.add("assistant", "hi there", tool_used="chat")
        ctx = conv_store.context(n=10)
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "hello"
        assert ctx[1]["role"] == "assistant"

    def test_context_order(self, conv_store: SQLiteConversationStore):
        """Messages returned in chronological order."""
        conv_store.add("user", "first")
        conv_store.add("assistant", "second")
        conv_store.add("user", "third")
        ctx = conv_store.context(n=10)
        assert [m["content"] for m in ctx] == ["first", "second", "third"]

    def test_context_limit(self, conv_store: SQLiteConversationStore):
        for i in range(20):
            conv_store.add("user", f"msg-{i}")
        ctx = conv_store.context(n=5)
        assert len(ctx) == 5

    def test_last_n(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "test1")
        conv_store.add("assistant", "reply1", tool_used="weather")
        msgs = conv_store.last_n(n=5)
        assert len(msgs) == 2
        assert "created_at" in msgs[0]
        assert msgs[1]["tool_used"] == "weather"

    def test_search(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "unique_xyz_search_term")
        conv_store.add("user", "something else")
        results = conv_store.search("unique_xyz_search_term")
        assert len(results) >= 1
        assert "unique_xyz_search_term" in results[0]["content"]

    def test_search_by_date(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "today message")
        results = conv_store.search_by_date(datetime.now())
        assert len(results) >= 1

    def test_conversation_list(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "first msg in conv")
        convs = conv_store.conversation_list()
        assert len(convs) >= 1
        assert convs[0]["message_count"] >= 1
        assert convs[0]["first_message"] == "first msg in conv"

    def test_stats(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "stat test")
        s = conv_store.stats()
        assert s["total_messages"] >= 1
        assert s["total_conversations"] >= 1
        assert s["current_session_id"] is not None

    def test_resume_session(self, conv_store: SQLiteConversationStore):
        sid = conv_store.session_id
        conv_store.add("user", "session msg")

        new_sid = conv_store.new_session()
        assert new_sid != sid

        assert conv_store.resume_session(sid) is True
        assert conv_store.session_id == sid

    def test_resume_nonexistent(self, conv_store: SQLiteConversationStore):
        assert conv_store.resume_session(99999) is False

    def test_prune_keeps_recent(self, conv_store: SQLiteConversationStore):
        conv_store.add("user", "recent msg")
        deleted = conv_store.prune(keep_days=90)
        assert deleted == 0  # session just created

    def test_close_and_reopen(self, db_path: Path):
        store = SQLiteConversationStore()
        store.init(db_path)
        store.new_session()
        store.add("user", "persist test")
        store.close()

        store2 = SQLiteConversationStore()
        store2.init(db_path)
        results = store2.search("persist test")
        assert len(results) >= 1
        store2.close()

    def test_auto_session_on_add(self, db_path: Path):
        """If add() is called without new_session(), one is auto-created."""
        store = SQLiteConversationStore()
        store.init(db_path)
        assert store.session_id is None
        store.add("user", "auto session")
        assert store.session_id is not None


# ━━ SQLite Reminder Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSQLiteReminderStore:
    def test_add_reminder(self, reminder_store: SQLiteReminderStore):
        rid = reminder_store.add("test remind", datetime.now() + timedelta(hours=1))
        assert rid > 0

    def test_list_upcoming(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("upcoming task", datetime.now() + timedelta(hours=1))
        items = reminder_store.list_upcoming()
        assert len(items) >= 1
        assert items[0]["title"] == "upcoming task"

    def test_check_due_marks_fired(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("overdue", datetime.now() - timedelta(hours=1))
        due = reminder_store.check_due()
        assert len(due) >= 1
        assert due[0]["title"] == "overdue"
        # Should be marked as fired — not in upcoming anymore
        upcoming = reminder_store.list_upcoming()
        assert not any(r["title"] == "overdue" for r in upcoming)

    def test_cancel_by_id(self, reminder_store: SQLiteReminderStore):
        rid = reminder_store.add("cancel me", datetime.now() + timedelta(hours=1))
        assert reminder_store.cancel(rid) is True
        assert reminder_store.cancel(rid) is False  # already gone

    def test_cancel_by_title(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("dup title", datetime.now() + timedelta(hours=1))
        reminder_store.add("dup title", datetime.now() + timedelta(hours=2))
        count = reminder_store.cancel_by_title("dup title")
        assert count == 2

    def test_snooze(self, reminder_store: SQLiteReminderStore):
        rid = reminder_store.add("snooze me", datetime.now() - timedelta(minutes=5))
        assert reminder_store.snooze(rid, minutes=15) is True

    def test_due_today(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("today task", datetime.now() + timedelta(hours=1))
        today = reminder_store.due_today()
        assert len(today) >= 1

    def test_count_active(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("active1", datetime.now() + timedelta(hours=1))
        reminder_store.add("active2", datetime.now() + timedelta(hours=2))
        assert reminder_store.count_active() >= 2

    def test_recurring_daily(self, reminder_store: SQLiteReminderStore):
        fire = datetime.now() - timedelta(minutes=5)
        reminder_store.add("daily check", fire, repeat="daily")
        due = reminder_store.check_due()
        assert len(due) >= 1
        assert due[0]["title"] == "daily check"
        # Should be rescheduled, not deleted
        upcoming = reminder_store.list_upcoming()
        assert any(r["title"] == "daily check" for r in upcoming)

    def test_place_trigger(self, reminder_store: SQLiteReminderStore):
        reminder_store.add(
            "buy milk",
            datetime.now() + timedelta(days=7),
            trigger_place="market",
        )
        due = reminder_store.check_place_due("market")
        assert len(due) >= 1
        assert due[0]["title"] == "buy milk"

    def test_list_all_includes_fired(self, reminder_store: SQLiteReminderStore):
        reminder_store.add("past", datetime.now() - timedelta(hours=1))
        reminder_store.check_due()  # marks as fired
        all_items = reminder_store.list_all()
        assert any(r["title"] == "past" for r in all_items)


# ━━ JSON Store Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestJSONProfileStore:
    def test_save_and_load(self, tmp_dir: Path):
        store = JSONProfileStore(tmp_dir)
        store.save({"name": "Iclal", "university": "OMU"})
        data = store.load()
        assert data["name"] == "Iclal"
        assert data["university"] == "OMU"

    def test_exists(self, tmp_dir: Path):
        store = JSONProfileStore(tmp_dir)
        assert store.exists() is False
        store.save({"name": "test"})
        assert store.exists() is True

    def test_empty_load(self, tmp_dir: Path):
        store = JSONProfileStore(tmp_dir)
        assert store.load() == {}

    def test_path_property(self, tmp_dir: Path):
        store = JSONProfileStore(tmp_dir)
        assert store.path == tmp_dir / "profile.json"

    def test_unicode_roundtrip(self, tmp_dir: Path):
        store = JSONProfileStore(tmp_dir)
        store.save({"name": "İclal", "department": "Bilgisayar Mühendisliği"})
        data = store.load()
        assert data["name"] == "İclal"


class TestJSONPlaceStore:
    def test_save_and_load(self, tmp_dir: Path):
        store = JSONPlaceStore(tmp_dir)
        data = {
            "dorm": {"label": "Yurt", "lat": 41.27, "lon": 36.33, "radius": 100},
            "campus": {"label": "Kampüs", "lat": 41.28, "lon": 36.34, "radius": 200},
        }
        store.save_all(data)
        loaded = store.load_all()
        assert "dorm" in loaded
        assert loaded["dorm"]["lat"] == 41.27
        assert len(loaded) == 2

    def test_empty_load(self, tmp_dir: Path):
        store = JSONPlaceStore(tmp_dir)
        assert store.load_all() == {}

    def test_overwrite(self, tmp_dir: Path):
        store = JSONPlaceStore(tmp_dir)
        store.save_all({"a": {"label": "A", "lat": 0, "lon": 0}})
        store.save_all({"b": {"label": "B", "lat": 1, "lon": 1}})
        loaded = store.load_all()
        assert "a" not in loaded
        assert "b" in loaded


class TestJSONScheduleStore:
    def test_save_and_load(self, tmp_dir: Path):
        store = JSONScheduleStore(tmp_dir)
        data = {
            "monday": [
                {"name": "Makine Öğrenmesi", "time": "10:00", "duration": 90},
                {"name": "Veri Tabanı", "time": "14:00", "duration": 60},
            ],
            "wednesday": [
                {"name": "Algoritma", "time": "09:00", "duration": 90},
            ],
        }
        store.save(data)
        loaded = store.load()
        assert len(loaded["monday"]) == 2
        assert loaded["wednesday"][0]["name"] == "Algoritma"

    def test_empty_load(self, tmp_dir: Path):
        store = JSONScheduleStore(tmp_dir)
        assert store.load() == {}


class TestJSONSessionStore:
    def test_save_and_load(self, tmp_dir: Path):
        store = JSONSessionStore(tmp_dir)
        store.save({"last_seen": "2026-03-05T10:00:00", "session_count": 5})
        loaded = store.load()
        assert loaded["session_count"] == 5

    def test_path_property(self, tmp_dir: Path):
        store = JSONSessionStore(tmp_dir)
        assert store.path == tmp_dir / "session.json"


# ━━ Migration Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMigration:
    def test_validate_json_files(self, tmp_dir: Path):
        (tmp_dir / "profile.json").write_text('{"name": "Iclal"}')
        (tmp_dir / "places.json").write_text('{"dorm": {"lat": 41.0}}')

        report = validate_json_files(tmp_dir)
        assert report["profile"]["status"] == "ok"
        assert report["places"]["status"] == "ok"
        assert report["schedule"]["status"] == "missing"
        assert report["session"]["status"] == "missing"

    def test_validate_corrupt_file(self, tmp_dir: Path):
        (tmp_dir / "profile.json").write_text("{invalid json}")
        report = validate_json_files(tmp_dir)
        assert report["profile"]["status"] == "error"

    def test_migrate_to_sqlite(self, tmp_dir: Path):
        db_path = tmp_dir / "migrate.db"

        (tmp_dir / "profile.json").write_text(
            json.dumps({"name": "Iclal", "university": "OMU"})
        )
        (tmp_dir / "places.json").write_text(
            json.dumps({"dorm": {"label": "Yurt", "lat": 41.0, "lon": 36.0}})
        )
        (tmp_dir / "schedule.json").write_text(
            json.dumps(
                {"monday": [{"name": "ML", "time": "10:00", "duration": 90}]}
            )
        )
        (tmp_dir / "session.json").write_text(
            json.dumps({"last_seen": "2026-03-05T10:00:00", "session_count": 5})
        )

        results = migrate_to_sqlite(db_path, tmp_dir)
        assert results["profile"]["status"] == "ok"
        assert results["places"]["status"] == "ok"
        assert results["schedule"]["status"] == "ok"
        assert results["session"]["status"] == "ok"

        # Verify data actually landed in SQLite
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # user_profile table
        rows = conn.execute(
            "SELECT * FROM user_profile"
        ).fetchall()
        assert len(rows) >= 1
        names = {r["key"] for r in rows}
        assert "name" in names

        # places table
        rows = conn.execute("SELECT * FROM places").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["key"] == "dorm"

        # schedule_entries table
        rows = conn.execute("SELECT * FROM schedule_entries").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["name"] == "ML"

        conn.close()

    def test_dry_run_no_write(self, tmp_dir: Path):
        db_path = tmp_dir / "dry.db"
        (tmp_dir / "profile.json").write_text('{"name": "test"}')

        results = migrate_to_sqlite(db_path, tmp_dir, dry_run=True)
        assert results["profile"]["status"] == "ok"

        # DB tables were created but no data
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT * FROM user_profile"
        ).fetchall()
        assert len(rows) == 0
        conn.close()

    def test_migrate_missing_files(self, tmp_dir: Path):
        db_path = tmp_dir / "empty.db"
        results = migrate_to_sqlite(db_path, tmp_dir)
        assert results["profile"]["status"] == "skipped"
        assert results["places"]["status"] == "skipped"


# ━━ ABC Contract Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestABCConformance:
    """Verify that concrete implementations satisfy the ABC contracts."""

    def test_sqlite_conversation_is_conversation_store(self):
        from bantz.data.store import ConversationStore

        assert issubclass(SQLiteConversationStore, ConversationStore)

    def test_sqlite_reminder_is_reminder_store(self):
        from bantz.data.store import ReminderStore

        assert issubclass(SQLiteReminderStore, ReminderStore)

    def test_json_profile_is_profile_store(self):
        from bantz.data.store import ProfileStore

        assert issubclass(JSONProfileStore, ProfileStore)

    def test_json_place_is_place_store(self):
        from bantz.data.store import PlaceStore

        assert issubclass(JSONPlaceStore, PlaceStore)

    def test_json_schedule_is_schedule_store(self):
        from bantz.data.store import ScheduleStore

        assert issubclass(JSONScheduleStore, ScheduleStore)

    def test_json_session_is_session_store(self):
        from bantz.data.store import SessionStore

        assert issubclass(JSONSessionStore, SessionStore)

    def test_sqlite_profile_is_profile_store(self):
        from bantz.data.store import ProfileStore
        from bantz.data.sqlite_store import SQLiteProfileStore

        assert issubclass(SQLiteProfileStore, ProfileStore)

    def test_sqlite_place_is_place_store(self):
        from bantz.data.store import PlaceStore
        from bantz.data.sqlite_store import SQLitePlaceStore

        assert issubclass(SQLitePlaceStore, PlaceStore)

    def test_sqlite_schedule_is_schedule_store(self):
        from bantz.data.store import ScheduleStore
        from bantz.data.sqlite_store import SQLiteScheduleStore

        assert issubclass(SQLiteScheduleStore, ScheduleStore)

    def test_sqlite_session_is_session_store(self):
        from bantz.data.store import SessionStore
        from bantz.data.sqlite_store import SQLiteSessionStore

        assert issubclass(SQLiteSessionStore, SessionStore)

    def test_memory_is_conversation_store(self):
        from bantz.data.store import ConversationStore
        from bantz.core.memory import Memory

        assert issubclass(Memory, ConversationStore)

    def test_scheduler_is_reminder_store(self):
        from bantz.data.store import ReminderStore
        from bantz.core.scheduler import Scheduler

        assert issubclass(Scheduler, ReminderStore)
