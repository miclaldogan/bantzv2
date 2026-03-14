"""Tests for bantz.core.location_handler (#225)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ── handle_location ──────────────────────────────────────────────────


class TestHandleLocation:
    @pytest.mark.asyncio
    async def test_with_gps_data(self):
        from bantz.core.location_handler import handle_location

        mock_places = MagicMock()
        mock_places.current_place_label.return_value = None

        gps = {"lat": 41.015137, "lon": 28.979530, "accuracy": 12}
        mock_gps_server = MagicMock()
        mock_gps_server.latest = gps

        mock_loc_svc = MagicMock()
        mock_loc_svc.get = AsyncMock(return_value=None)

        with patch("bantz.core.places.places", mock_places), \
             patch("bantz.core.location.location_service", mock_loc_svc), \
             patch("bantz.core.gps_server.gps_server", mock_gps_server):
            result = await handle_location()

        assert "41.015137" in result
        assert "28.979530" in result
        assert "±12m" in result

    @pytest.mark.asyncio
    async def test_with_named_place(self):
        from bantz.core.location_handler import handle_location

        mock_places = MagicMock()
        mock_places.current_place_label.return_value = "Home"

        mock_loc_svc = MagicMock()
        mock_loc_svc.get = AsyncMock(return_value=None)

        with patch("bantz.core.places.places", mock_places), \
             patch("bantz.core.location.location_service", mock_loc_svc), \
             patch.dict("sys.modules", {"bantz.core.gps_server": None}):
            result = await handle_location()

        assert "Home" in result

    @pytest.mark.asyncio
    async def test_no_gps_shows_fallback(self):
        from bantz.core.location_handler import handle_location

        mock_places = MagicMock()
        mock_places.current_place_label.return_value = None

        mock_loc_svc = MagicMock()
        mock_loc_svc.get = AsyncMock(return_value=None)

        with patch("bantz.core.places.places", mock_places), \
             patch("bantz.core.location.location_service", mock_loc_svc), \
             patch.dict("sys.modules", {"bantz.core.gps_server": None}):
            result = await handle_location()

        assert "can't pinpoint" in result

    @pytest.mark.asyncio
    async def test_with_live_location_service(self):
        from bantz.core.location_handler import handle_location

        mock_places = MagicMock()
        mock_places.current_place_label.return_value = None

        loc_obj = SimpleNamespace(
            is_live=True,
            display="Istanbul, Turkey",
            lat=41.0,
            lon=29.0,
            source="GeoClue",
        )

        mock_loc_svc = MagicMock()
        mock_loc_svc.get = AsyncMock(return_value=loc_obj)

        with patch("bantz.core.places.places", mock_places), \
             patch("bantz.core.location.location_service", mock_loc_svc), \
             patch.dict("sys.modules", {"bantz.core.gps_server": None}):
            result = await handle_location()

        assert "Istanbul" in result
        assert "GeoClue" in result


# ── handle_save_place ────────────────────────────────────────────────


class TestHandleSavePlace:
    @pytest.mark.asyncio
    async def test_save_success(self):
        from bantz.core.location_handler import handle_save_place

        mock_places = MagicMock()
        mock_places.save_here.return_value = {"lat": 41.0, "lon": 29.0, "radius": 100}

        with patch("bantz.core.places.places", mock_places):
            result = await handle_save_place("Home")

        assert "Saved 'Home'" in result
        assert "41.000000" in result
        mock_places.save_here.assert_called_once_with("Home")

    @pytest.mark.asyncio
    async def test_save_no_gps(self):
        from bantz.core.location_handler import handle_save_place

        mock_places = MagicMock()
        mock_places.save_here.return_value = None

        with patch("bantz.core.places.places", mock_places):
            result = await handle_save_place("Uni")

        assert "No GPS data" in result


# ── handle_list_places ───────────────────────────────────────────────


class TestHandleListPlaces:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        from bantz.core.location_handler import handle_list_places

        mock_places = MagicMock()
        mock_places.all_places.return_value = {}

        with patch("bantz.core.places.places", mock_places):
            result = await handle_list_places()

        assert "No saved places" in result

    @pytest.mark.asyncio
    async def test_list_with_places(self):
        from bantz.core.location_handler import handle_list_places

        mock_places = MagicMock()
        mock_places.all_places.return_value = {
            "home": {"label": "Home", "lat": 41.0, "lon": 29.0, "radius": 100},
            "uni": {"label": "University", "lat": 41.1, "lon": 29.1, "radius": 200},
        }
        mock_places._current_place_key = "home"

        with patch("bantz.core.places.places", mock_places):
            result = await handle_list_places()

        assert "Home" in result
        assert "University" in result
        assert "you are here" in result


# ── handle_delete_place ──────────────────────────────────────────────


class TestHandleDeletePlace:
    @pytest.mark.asyncio
    async def test_delete_success(self):
        from bantz.core.location_handler import handle_delete_place

        mock_places = MagicMock()
        mock_places.delete_place.return_value = True

        with patch("bantz.core.places.places", mock_places):
            result = await handle_delete_place("Uni")

        assert "'Uni' deleted" in result

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        from bantz.core.location_handler import handle_delete_place

        mock_places = MagicMock()
        mock_places.delete_place.return_value = False

        with patch("bantz.core.places.places", mock_places):
            result = await handle_delete_place("Nowhere")

        assert "No saved place" in result


# ── brain.py delegation ──────────────────────────────────────────────


class TestBrainDelegation:
    """Brain methods should delegate to location_handler functions."""

    @pytest.mark.asyncio
    async def test_brain_handle_location_delegates(self):
        from bantz.core.brain import Brain
        with patch("bantz.core.location_handler.handle_location", new_callable=AsyncMock) as mock:
            mock.return_value = "mocked location"
            b = Brain.__new__(Brain)
            result = await b._handle_location()
        assert result == "mocked location"

    @pytest.mark.asyncio
    async def test_brain_handle_save_place_delegates(self):
        """dispatch_internal routes _save_place to location_handler."""
        from bantz.core.routing_engine import dispatch_internal
        with patch("bantz.core.location_handler.handle_save_place", new_callable=AsyncMock) as mock, \
             patch("bantz.core.routing_engine.data_layer") as dl:
            dl.conversations = MagicMock()
            mock.return_value = "saved"
            result = await dispatch_internal(
                "_save_place", {"name": "Park"}, "", "", {},
            )
        mock.assert_called_once_with("Park")
        assert result is not None
        assert "saved" in result.response

    @pytest.mark.asyncio
    async def test_brain_handle_list_places_delegates(self):
        """dispatch_internal routes _list_places to location_handler."""
        from bantz.core.routing_engine import dispatch_internal
        with patch("bantz.core.location_handler.handle_list_places", new_callable=AsyncMock) as mock, \
             patch("bantz.core.routing_engine.data_layer") as dl:
            dl.conversations = MagicMock()
            mock.return_value = "places list"
            result = await dispatch_internal(
                "_list_places", {}, "", "", {},
            )
        assert result is not None
        assert "places list" in result.response

    @pytest.mark.asyncio
    async def test_brain_handle_delete_place_delegates(self):
        """dispatch_internal routes _delete_place to location_handler."""
        from bantz.core.routing_engine import dispatch_internal
        with patch("bantz.core.location_handler.handle_delete_place", new_callable=AsyncMock) as mock, \
             patch("bantz.core.routing_engine.data_layer") as dl:
            dl.conversations = MagicMock()
            mock.return_value = "deleted"
            result = await dispatch_internal(
                "_delete_place", {"name": "Gym"}, "", "", {},
            )
        mock.assert_called_once_with("Gym")
        assert result is not None
        assert "deleted" in result.response
