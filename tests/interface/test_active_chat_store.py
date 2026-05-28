"""Tests for #436 — _ActiveChatStore: persistent active Telegram chat IDs.

Verifies that _ActiveChatStore survives restarts via SQLiteKVStore, supports
add/iter/bool, and falls back gracefully when the DB is unavailable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Provide a stub for the 'telegram' package so telegram_bot.py can be imported
# in environments where python-telegram-bot is not installed.
def _stub_telegram():
    for mod in [
        "telegram", "telegram.constants", "telegram.ext",
        "telegram.ext._application", "telegram.ext._handlers",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


_stub_telegram()


# ─── helper to build an isolated _ActiveChatStore ────────────────────────────

def _make_store(kv_mock):
    """Return an _ActiveChatStore wired to the given SQLiteKVStore mock."""
    with patch.dict(sys.modules, {"bantz.data.sqlite_store": MagicMock(SQLiteKVStore=lambda _: kv_mock)}):
        # Re-import _ActiveChatStore fresh from module source to pick up the patch
        import importlib
        import bantz.interface.telegram_bot as tg_mod
        importlib.reload(tg_mod)
        return tg_mod._ActiveChatStore()


# ─── tests ───────────────────────────────────────────────────────────────────

class TestActiveChatStorePersistence:
    """_ActiveChatStore reads/writes from SQLiteKVStore."""

    def _make_kv(self, initial: list[int] | None = None) -> MagicMock:
        store = MagicMock()
        data = json.dumps(initial or [])
        store.get.return_value = data
        return store

    def test_add_new_chat_saves_to_kv(self):
        kv = self._make_kv([])
        store = _make_store(kv)
        store.add(42)
        kv.set.assert_called_once()
        saved = json.loads(kv.set.call_args[0][1])
        assert 42 in saved

    def test_add_existing_chat_does_not_double_save(self):
        kv = self._make_kv([42])
        store = _make_store(kv)
        store.add(42)
        kv.set.assert_not_called()

    def test_iter_returns_persisted_ids(self):
        kv = self._make_kv([1, 2, 3])
        store = _make_store(kv)
        result = list(store)
        assert set(result) == {1, 2, 3}

    def test_bool_true_when_chats_present(self):
        kv = self._make_kv([99])
        store = _make_store(kv)
        assert bool(store) is True

    def test_bool_false_when_no_chats(self):
        kv = self._make_kv([])
        store = _make_store(kv)
        assert bool(store) is False

    def test_corrupted_json_returns_empty_set(self):
        kv = MagicMock()
        kv.get.return_value = "not-json!!"
        store = _make_store(kv)
        assert list(store) == []
        assert not store


class TestActiveChatStoreMemoryFallback:
    """Falls back to in-memory set when SQLiteKVStore is unavailable."""

    def test_fallback_add_and_iter(self):
        with patch("bantz.data.sqlite_store.SQLiteKVStore", side_effect=RuntimeError("no db")):
            import importlib
            import bantz.interface.telegram_bot as tg_mod
            importlib.reload(tg_mod)
            store = tg_mod._ActiveChatStore()

        store.add(7)
        store.add(8)
        assert set(store) == {7, 8}
        assert bool(store) is True

    def test_fallback_bool_false_initially(self):
        with patch("bantz.data.sqlite_store.SQLiteKVStore", side_effect=RuntimeError("no db")):
            import importlib
            import bantz.interface.telegram_bot as tg_mod
            importlib.reload(tg_mod)
            store = tg_mod._ActiveChatStore()

        assert not store


class TestBridgeNoRedisReference:
    """memory/bridge.py must not reference Redis in its docstring."""

    def test_redis_removed_from_bridge_docstring(self):
        from bantz.memory import bridge
        import inspect
        src = inspect.getsource(bridge.MemPalaceBridge)
        assert "Redis" not in src, (
            "Redis reference still present in MemPalaceBridge docstring — "
            "should have been removed by #436 fix"
        )
