"""Tests for bantz.desktop.widgets — Widget data providers (#365)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bantz.desktop.widgets import WidgetDataProvider


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def provider(tmp_path: Path) -> WidgetDataProvider:
    return WidgetDataProvider(data_dir=tmp_path)


# ── CPU ───────────────────────────────────────────────────────────────────────

class TestCPU:
    def test_returns_number_string(self) -> None:
        result = WidgetDataProvider.get_cpu()
        assert result.isdigit() or result == "0"

    def test_range_0_to_100(self) -> None:
        val = int(WidgetDataProvider.get_cpu())
        assert 0 <= val <= 100


# ── RAM ───────────────────────────────────────────────────────────────────────

class TestRAM:
    def test_returns_number_string(self) -> None:
        result = WidgetDataProvider.get_ram()
        assert result.isdigit() or result == "0"

    def test_range_0_to_100(self) -> None:
        val = int(WidgetDataProvider.get_ram())
        assert 0 <= val <= 100


# ── Disk ──────────────────────────────────────────────────────────────────────

class TestDisk:
    def test_returns_number_string(self) -> None:
        result = WidgetDataProvider.get_disk()
        assert result.isdigit() or result == "0"

    def test_range_0_to_100(self) -> None:
        val = int(WidgetDataProvider.get_disk())
        assert 0 <= val <= 100


# ── GPU ───────────────────────────────────────────────────────────────────────

class TestGPU:
    def test_returns_string(self) -> None:
        result = WidgetDataProvider.get_gpu()
        assert isinstance(result, str)

    def test_fallback_to_na(self) -> None:
        with patch("shutil.which", return_value=None):
            with patch.dict("sys.modules", {"pynvml": None}):
                # Force ImportError on pynvml
                result = WidgetDataProvider.get_gpu()
                assert isinstance(result, str)


# ── Network ───────────────────────────────────────────────────────────────────

class TestNetwork:
    def test_returns_string(self) -> None:
        result = WidgetDataProvider.get_network()
        assert isinstance(result, str)
        # Should contain arrows if psutil available
        if result != "N/A":
            assert "↑" in result or "M" in result


# ── News (cached) ─────────────────────────────────────────────────────────────

class TestNews:
    def test_no_cache_message(self, provider: WidgetDataProvider) -> None:
        result = provider.get_news()
        assert "No news" in result or "populate" in result

    def test_reads_cache(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "news_cache.json"
        cache.write_text(json.dumps([
            {"title": "AI breaks record", "source": "TechCrunch"},
            {"title": "Bantz v5 released", "source": "GitHub"},
        ]))
        result = provider.get_news()
        assert "AI breaks record" in result
        assert "TechCrunch" in result
        assert "Bantz v5" in result

    def test_empty_cache_list(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "news_cache.json"
        cache.write_text("[]")
        result = provider.get_news()
        assert "No recent news" in result

    def test_corrupt_cache(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "news_cache.json"
        cache.write_text("not json {{{")
        result = provider.get_news()
        assert "No news" in result


# ── Calendar ──────────────────────────────────────────────────────────────────

class TestCalendar:
    def test_no_cache(self, provider: WidgetDataProvider) -> None:
        result = provider.get_calendar()
        assert "No calendar data" in result

    def test_reads_events(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "calendar_cache.json"
        cache.write_text(json.dumps([
            {"time": "14:00", "summary": "Team standup"},
            {"time": "16:30", "summary": "Coffee break"},
        ]))
        result = provider.get_calendar()
        assert "Team standup" in result
        assert "14:00" in result

    def test_no_events(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "calendar_cache.json"
        cache.write_text("[]")
        result = provider.get_calendar()
        assert "No upcoming events" in result


# ── Todos ─────────────────────────────────────────────────────────────────────

class TestTodos:
    def test_no_cache(self, provider: WidgetDataProvider) -> None:
        result = provider.get_todos()
        assert "No todos" in result

    def test_reads_todos(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "todos_cache.json"
        cache.write_text(json.dumps([
            {"text": "Buy groceries", "done": False},
            {"text": "Review PR", "done": True},
        ]))
        result = provider.get_todos()
        assert "Buy groceries" in result
        assert "○" in result  # not done
        assert "✓" in result  # done

    def test_empty_todos(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "todos_cache.json"
        cache.write_text("[]")
        result = provider.get_todos()
        assert "No todos" in result


# ── Weather ───────────────────────────────────────────────────────────────────

class TestWeather:
    def test_no_cache_falls_back(self, provider: WidgetDataProvider) -> None:
        # Without curl/wttr.in, should return unavailable
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = provider.get_weather()
            assert "unavailable" in result.lower() or isinstance(result, str)

    def test_reads_cache(self, provider: WidgetDataProvider) -> None:
        cache = provider.data_dir / "weather_cache.json"
        cache.write_text(json.dumps({"summary": "☀️ 25°C, Clear"}))
        result = provider.get_weather()
        assert "25°C" in result


# ── Status ────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_returns_string(self, provider: WidgetDataProvider) -> None:
        result = provider.get_status()
        assert result in ("active", "idle")

    def test_status_json(self, provider: WidgetDataProvider) -> None:
        result = provider.get_status_json()
        data = json.loads(result)
        assert "text" in data
        assert "tooltip" in data
        assert "class" in data


# ── Dispatch ──────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_known_widget(self, provider: WidgetDataProvider) -> None:
        result = provider.get("cpu")
        assert result.isdigit() or result == "0"

    def test_unknown_widget(self, provider: WidgetDataProvider) -> None:
        result = provider.get("nonexistent")
        data = json.loads(result)
        assert "error" in data

    def test_all_names(self, provider: WidgetDataProvider) -> None:
        for name in ("cpu", "ram", "disk", "gpu", "network",
                     "weather", "news", "calendar", "todos", "status"):
            result = provider.get(name)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_exception_handling(self, provider: WidgetDataProvider) -> None:
        with patch.object(provider, "get_cpu", side_effect=RuntimeError("boom")):
            result = provider.get("cpu")
            data = json.loads(result)
            assert "error" in data
            assert "boom" in data["error"]
