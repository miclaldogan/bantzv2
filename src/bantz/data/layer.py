"""
Bantz v3 — Data Access Layer (Singleton)

Composes all stores behind a single entry point.
Initialized once during app startup — replaces scattered init() calls.

    from bantz.data import data_layer

    data_layer.init(config)
    data_layer.conversations.add("user", "hello")

At runtime the DataLayer wires the existing ``Memory`` and ``Scheduler``
singletons (which now inherit from the ABCs) so that every module in the
codebase — old or new — shares the same connections and session.

Profile, places, schedule, and session now go through SQLite stores
(migrated from JSON on first launch).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bantz.data.store import (
    ConversationStore,
    GraphStore,
    PlaceStore,
    ProfileStore,
    ReminderStore,
    ScheduleStore,
    SessionStore,
)
from bantz.data.json_store import (
    JSONPlaceStore,
    JSONProfileStore,
    JSONScheduleStore,
    JSONSessionStore,
)
from bantz.data.sqlite_store import (
    SQLitePlaceStore,
    SQLiteProfileStore,
    SQLiteScheduleStore,
    SQLiteSessionStore,
)

if TYPE_CHECKING:
    from bantz.config import Config

log = logging.getLogger("bantz.data")


class DataLayer:
    """Unified data access — the one-stop shop for all persistence.

    Attributes:
        conversations — messages & sessions      (SQLite via Memory)
        reminders     — scheduled reminders      (SQLite via Scheduler)
        profile       — user identity/prefs      (SQLite, auto-migrated from JSON)
        places        — named GPS locations      (SQLite, auto-migrated from JSON)
        schedule      — weekly timetable         (SQLite, auto-migrated from JSON)
        session       — launch tracking          (SQLite, auto-migrated from JSON)
        graph         — knowledge graph          (Neo4j, optional)
    """

    def __init__(self) -> None:
        self.conversations: ConversationStore = None  # type: ignore[assignment]
        self.reminders: ReminderStore = None  # type: ignore[assignment]
        self.profile: ProfileStore = None  # type: ignore[assignment]
        self.places: PlaceStore = None  # type: ignore[assignment]
        self.schedule: ScheduleStore = None  # type: ignore[assignment]
        self.session: SessionStore = None  # type: ignore[assignment]
        self.graph: Optional[GraphStore] = None
        self._initialized = False

    # ── initialization ────────────────────────────────────────────────────

    def init(self, cfg: "Config") -> None:
        """Initialize all stores.  Call once at app startup.

        Wires the existing ``Memory`` and ``Scheduler`` singletons so that
        they serve double duty as the DAL conversation/reminder stores.
        SQLite stores are created for profile, places, schedule, session,
        with automatic migration from JSON on first run.
        """
        if self._initialized:
            return

        # Import legacy singletons
        from bantz.core.memory import memory
        from bantz.core.scheduler import scheduler

        cfg.ensure_dirs()

        # ── SQLite stores (via legacy singletons) ────────────────────────
        memory.init(cfg.db_path)
        memory.new_session()
        scheduler.init(cfg.db_path)

        self.conversations = memory  # Memory IS-A ConversationStore
        self.reminders = scheduler  # Scheduler IS-A ReminderStore

        # ── SQLite stores for profile / places / schedule / session ──────
        self.profile = SQLiteProfileStore(cfg.db_path)
        self.places = SQLitePlaceStore(cfg.db_path)
        self.schedule = SQLiteScheduleStore(cfg.db_path)
        self.session = SQLiteSessionStore(cfg.db_path)

        # ── Spatial cache for UI element coordinates (#121) ──────────────
        try:
            from bantz.vision.spatial_cache import spatial_db
            spatial_db.init(cfg.db_path)
            log.debug("Spatial cache initialized")
        except Exception as exc:
            log.debug("Spatial cache init skipped: %s", exc)

        # ── Auto-migrate JSON → SQLite if tables are empty ───────────────
        base_dir = (
            Path(cfg.data_dir)
            if cfg.data_dir
            else Path.home() / ".local" / "share" / "bantz"
        )
        self._auto_migrate_json(base_dir)

        # ── Bind stores to core singletons ───────────────────────────────
        from bantz.core.session import session_tracker
        from bantz.core.profile import profile
        from bantz.core.places import places
        from bantz.core.schedule import schedule

        session_tracker.bind_store(self.session)
        profile.bind_store(self.profile)
        places.bind_store(self.places)
        schedule.bind_store(self.schedule)

        self._initialized = True
        log.info("DataLayer initialized  db=%s  data=%s", cfg.db_path, base_dir)

    def _auto_migrate_json(self, base_dir: Path) -> None:
        """Silently import JSON data into SQLite if tables are empty.

        Ensures seamless upgrade for existing v2 users.
        """
        migrated = []

        # ── profile.json → user_profile table ────────────────────────────
        if not self.profile.exists():
            pf = base_dir / "profile.json"
            if pf.exists():
                try:
                    data = json.loads(pf.read_text("utf-8"))
                    if data:
                        self.profile.save(data)
                        migrated.append("profile")
                except Exception as exc:
                    log.warning("Auto-migrate profile.json failed: %s", exc)

        # ── places.json → places table ───────────────────────────────────
        if not self.places.exists():
            pf = base_dir / "places.json"
            if pf.exists():
                try:
                    data = json.loads(pf.read_text("utf-8"))
                    if data:
                        self.places.save_all(data)
                        migrated.append("places")
                except Exception as exc:
                    log.warning("Auto-migrate places.json failed: %s", exc)

        # ── schedule.json → schedule_entries table ───────────────────────
        if not self.schedule.exists():
            pf = base_dir / "schedule.json"
            if pf.exists():
                try:
                    data = json.loads(pf.read_text("utf-8"))
                    if data:
                        self.schedule.save(data)
                        migrated.append("schedule")
                except Exception as exc:
                    log.warning("Auto-migrate schedule.json failed: %s", exc)

        # ── session.json → session_state table ───────────────────────────
        sess_data = self.session.load()
        if not sess_data:
            pf = base_dir / "session.json"
            if pf.exists():
                try:
                    data = json.loads(pf.read_text("utf-8"))
                    if data:
                        self.session.save(data)
                        migrated.append("session")
                except Exception as exc:
                    log.warning("Auto-migrate session.json failed: %s", exc)

        if migrated:
            log.info("Auto-migrated JSON → SQLite: %s", ", ".join(migrated))

    async def init_graph(self) -> None:
        """Initialize the optional graph store (Neo4j).

        Safe to call even if Neo4j is disabled or not installed.
        """
        try:
            from bantz.memory.graph import graph_memory

            if await graph_memory.init():
                self.graph = graph_memory  # type: ignore[assignment]
        except ImportError:
            log.debug("neo4j driver not installed — graph memory off")

    # ── properties ────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── teardown ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down all stores cleanly."""
        if self.conversations is not None:
            self.conversations.close()
        # Close spatial cache (#121)
        try:
            from bantz.vision.spatial_cache import spatial_db
            spatial_db.close()
        except Exception:
            pass
        if self.graph is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.graph.close())
                else:
                    loop.run_until_complete(self.graph.close())
            except Exception:
                pass
        self._initialized = False
        log.info("DataLayer closed")


# Singleton — import this everywhere
data_layer = DataLayer()
