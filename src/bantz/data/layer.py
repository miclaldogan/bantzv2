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
"""
from __future__ import annotations

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

if TYPE_CHECKING:
    from bantz.config import Config

log = logging.getLogger("bantz.data")


class DataLayer:
    """Unified data access — the one-stop shop for all persistence.

    Attributes:
        conversations — messages & sessions      (SQLite via Memory)
        reminders     — scheduled reminders      (SQLite via Scheduler)
        profile       — user identity/prefs      (JSON file)
        places        — named GPS locations      (JSON file)
        schedule      — weekly timetable         (JSON file)
        session       — launch tracking          (JSON file)
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
        JSON stores are created fresh for profile, places, schedule, session.
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

        # ── JSON stores ──────────────────────────────────────────────────
        base_dir = (
            Path(cfg.data_dir)
            if cfg.data_dir
            else Path.home() / ".local" / "share" / "bantz"
        )
        self.profile = JSONProfileStore(base_dir)
        self.places = JSONPlaceStore(base_dir)
        self.schedule = JSONScheduleStore(base_dir)
        self.session = JSONSessionStore(base_dir)

        self._initialized = True
        log.info("DataLayer initialized  db=%s  data=%s", cfg.db_path, base_dir)

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
