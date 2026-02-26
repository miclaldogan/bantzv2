"""
Bantz v2 — Place Service

Known locations with labels — "yurt", "kampüs", "ev", etc.
Compare current GPS to known places.  Travel hints for schedule.

Data: ~/.local/share/bantz/places.json
Setup: bantz --setup places

Usage:
    from bantz.core.places import places
    p = await places.current_place()
    hint = await places.travel_hint("Mimarlık B2", 20)
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from bantz.core.location import Location, location_service

PLACES_PATH = Path.home() / ".local" / "share" / "bantz" / "places.json"

# Radius in metres — if within this range, consider "at" the place
MATCH_RADIUS_M = 500


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two lat/lon points."""
    R = 6_371_000  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class PlaceService:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if PLACES_PATH.exists():
            try:
                self._data = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    def reload(self) -> None:
        """Force reload from disk (after --setup places)."""
        self._loaded = False
        self._data = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    async def current_place(self) -> Optional[dict]:
        """
        Compare current location to known places.
        Returns: {"key": "yurt", "label": "Yurt", "distance_m": 120}
        or None if no match within MATCH_RADIUS_M.
        """
        self._load()
        if not self._data:
            return None

        loc = await location_service.get()
        if loc.lat == 0.0 and loc.lon == 0.0:
            return None

        best: Optional[dict] = None
        best_dist = float("inf")

        for key, place in self._data.items():
            plat = place.get("lat", 0.0)
            plon = place.get("lon", 0.0)
            if plat == 0.0 and plon == 0.0:
                continue
            dist = _haversine_m(loc.lat, loc.lon, plat, plon)
            if dist < best_dist:
                best_dist = dist
                best = {"key": key, "label": place.get("label", key), "distance_m": round(dist)}

        if best and best_dist <= MATCH_RADIUS_M:
            return best
        return None

    async def travel_hint(
        self, destination: str, minutes_until: int
    ) -> Optional[str]:
        """
        Contextual travel hint:
          "yurttasın, derse 20 dk var — kampüse 15 dk yürüyüş"
        Returns None if place unknown or not meaningful.
        """
        place = await self.current_place()
        if not place:
            return None

        label = place["label"]

        if minutes_until <= 0:
            return None  # already started, no point

        # Check if destination matches a known place
        self._load()
        dest_place = self._find_place_by_label(destination)

        if dest_place and place["key"] != dest_place["key"]:
            loc = await location_service.get()
            dist = _haversine_m(
                loc.lat, loc.lon,
                dest_place.get("lat", 0.0), dest_place.get("lon", 0.0),
            )
            walk_min = round(dist / 80)  # ~80 m/min walking speed ≈ 4.8 km/h
            return (
                f"You're at {label}, class in {minutes_until} min"
                f" — ~{walk_min} min walk to {dest_place.get('label', destination)}"
            )
        else:
            # Generic — we know where user is, but not the destination
            if minutes_until <= 15:
                urgency = "⚠️ hurry up"
            elif minutes_until <= 30:
                urgency = "time to go"
            else:
                urgency = "no rush"
            return f"You're at {label}, class in {minutes_until} min — {urgency}"

    def _find_place_by_label(self, text: str) -> Optional[dict]:
        """Find a known place whose label appears in the text (fuzzy)."""
        self._load()
        text_lower = text.lower()
        for key, place in self._data.items():
            label = place.get("label", key).lower()
            if label in text_lower or key.lower() in text_lower:
                return {**place, "key": key}
        return None

    def all_places(self) -> dict[str, dict]:
        self._load()
        return dict(self._data)

    def save(self, data: dict[str, dict]) -> None:
        """Write places to disk."""
        PLACES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLACES_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._data = data
        self._loaded = True

    def is_configured(self) -> bool:
        return PLACES_PATH.exists()

    def status_line(self) -> str:
        """Short summary for --doctor."""
        self._load()
        if not self._data:
            return "not configured  → bantz --setup places"
        n = len(self._data)
        names = ", ".join(p.get("label", k) for k, p in list(self._data.items())[:3])
        if n > 3:
            names += f" +{n - 3}"
        return f"{n} locations: {names}"

    @staticmethod
    def setup_path() -> Path:
        return PLACES_PATH


places = PlaceService()
