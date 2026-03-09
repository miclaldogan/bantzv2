"""
Bantz v2 — Place Service

Known locations with labels — "dorm", "campus", "home", etc.
Compare current GPS to known places.  Travel hints for schedule.
Geofence detection, stationary tracking, proactive place-learning.

Data stored via DAL (SQLite), with JSON fallback for backward compat.
Setup: bantz --setup places

Usage:
    from bantz.core.places import places
    p = await places.current_place()
    hint = await places.travel_hint("Architecture B2", 20)
    places.save_here("dorm")                      # save current GPS as "dorm"
    places.update_gps(lat, lon)                    # feed GPS tick
    notice = places.check_stationary()             # ask about unknown spot
"""
from __future__ import annotations

import json
import logging
import math
import time as _time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bantz.core.location import Location, location_service

if TYPE_CHECKING:
    from bantz.data.store import PlaceStore

log = logging.getLogger("bantz.places")

PLACES_PATH = Path.home() / ".local" / "share" / "bantz" / "places.json"

# Radius in metres — if within this range, consider "at" the place
MATCH_RADIUS_M = 100          # tightened from 500 → 100 for actual geofencing
MOVE_THRESHOLD_M = 100        # must move this far to reset anchor
STATIONARY_MINUTES = 45       # wait this long before asking


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
        self._store: PlaceStore | None = None

        # ── Geofence / stationary state ──────────────────────────────
        self._current_place_key: Optional[str] = None
        self._last_place_key: Optional[str] = None

        # Anchor: the spot where we "settled"
        self._anchor_lat: Optional[float] = None
        self._anchor_lon: Optional[float] = None
        self._anchor_time: float = 0.0
        self._asked_for_anchor: bool = False
        self._last_ask_time: float = 0.0

        # Location-triggered reminders that fired
        self._place_fired: list[dict] = []

    def bind_store(self, store: PlaceStore) -> None:
        """Bind a DAL store for persistence (called by DataLayer)."""
        self._store = store
        self._loaded = False  # force reload on next access

    def _load(self) -> None:
        if self._loaded:
            return
        if self._store:
            self._data = self._store.load_all()
        elif PLACES_PATH.exists():
            try:
                self._data = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    def reload(self) -> None:
        """Force reload from store (after --setup places)."""
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

        return self._find_nearest(loc.lat, loc.lon)

    def find_place_at(self, lat: float, lon: float) -> Optional[dict]:
        """Find the nearest known place within MATCH_RADIUS_M (sync)."""
        self._load()
        return self._find_nearest(lat, lon)

    def _find_nearest(self, lat: float, lon: float) -> Optional[dict]:
        """Internal: find nearest place within radius."""
        best: Optional[dict] = None
        best_dist = float("inf")

        for key, place in self._data.items():
            plat = place.get("lat", 0.0)
            plon = place.get("lon", 0.0)
            if plat == 0.0 and plon == 0.0:
                continue
            radius = place.get("radius", MATCH_RADIUS_M)
            dist = _haversine_m(lat, lon, plat, plon)
            if dist <= radius and dist < best_dist:
                best_dist = dist
                best = {
                    "key": key,
                    "label": place.get("label", key),
                    "distance_m": round(dist),
                }
        return best

    # ── GPS tick: feed every GPS update ───────────────────────────────

    def update_gps(self, lat: float, lon: float) -> Optional[str]:
        """
        Feed a GPS reading.  Returns a transition message if the user
        entered a *new* known place, else None.

        Call this each time new GPS data arrives.
        Also triggers location-based reminders from the scheduler.
        """
        self._load()
        now = _time.time()

        place = self.find_place_at(lat, lon)
        place_key = place["key"] if place else None

        transition: Optional[str] = None

        # Detect place transition
        if place_key != self._current_place_key:
            if place_key and place:
                transition = place["label"]
                log.info("Entered place: %s", place["label"])
                # Fire location-based reminders
                self._fire_place_reminders(place_key, place["label"])
            elif self._current_place_key:
                log.info("Left place: %s", self._current_place_key)
            self._last_place_key = self._current_place_key
            self._current_place_key = place_key

        # Update stationary anchor
        if self._anchor_lat is None or self._anchor_lon is None:
            self._anchor_lat = lat
            self._anchor_lon = lon
            self._anchor_time = now
            self._asked_for_anchor = False
        else:
            dist = _haversine_m(self._anchor_lat, self._anchor_lon, lat, lon)
            if dist > MOVE_THRESHOLD_M:
                # User moved — reset anchor
                self._anchor_lat = lat
                self._anchor_lon = lon
                self._anchor_time = now
                self._asked_for_anchor = False

        return transition

    def current_place_label(self) -> Optional[str]:
        """Return label of the place the user is currently in, or None."""
        if self._current_place_key is None:
            return None
        self._load()
        place = self._data.get(self._current_place_key)
        if place:
            return place.get("label", self._current_place_key)
        return self._current_place_key

    # ── Location-based reminders ──────────────────────────────────────

    def _fire_place_reminders(self, place_key: str, place_label: str) -> None:
        """Check scheduler for reminders triggered by entering this place."""
        try:
            from bantz.core.scheduler import scheduler
            due = scheduler.check_place_due(place_key)
            for item in due:
                item["_place_label"] = place_label
                self._place_fired.append(item)
                log.info(
                    "Location reminder #%d fired at %s: %s",
                    item["id"], place_label, item["title"],
                )
        except Exception as exc:
            log.debug("Place reminder check failed: %s", exc)

    def pop_place_reminders(self) -> list[dict]:
        """Return and clear any location-triggered reminders that fired.

        The TUI and Telegram bot poll this to display notifications.
        """
        fired = self._place_fired[:]
        self._place_fired.clear()
        return fired

    def check_stationary(self) -> Optional[str]:
        """
        If user sat in an unknown location for STATIONARY_MINUTES,
        return a polite prompt.  Otherwise return None.

        Call periodically (every few minutes).
        """
        if self._anchor_lat is None or self._anchor_lon is None:
            return None

        # Already at a known place — no need to ask
        if self._current_place_key is not None:
            return None

        # Already asked about this spot
        if self._asked_for_anchor:
            return None

        now = _time.time()

        # Cooldown: don't ask more than once per 30 min
        if now - self._last_ask_time < 1800:
            return None

        elapsed_min = (now - self._anchor_time) / 60
        if elapsed_min < STATIONARY_MINUTES:
            return None

        self._asked_for_anchor = True
        self._last_ask_time = now

        lat = self._anchor_lat
        lon = self._anchor_lon
        minutes = int(elapsed_min)

        log.info("Stationary %d min at unknown: %.4f, %.4f", minutes, lat, lon)

        return (
            f"I notice you've been at the same spot for about {minutes} minutes "
            f"({lat:.4f}, {lon:.4f}) but it's not in my saved places.\n"
            f"What is this place? Give me a name and I'll save it. "
            f"(e.g. 'save here as dorm' or 'this is the library')"
        )

    # ── Save current location as named place ─────────────────────────

    def save_here(self, name: str, radius: float = MATCH_RADIUS_M) -> Optional[dict]:
        """
        Save the current GPS position as a named place.
        Tries: anchor position → gps_server latest.
        Returns the saved place dict, or None if no GPS.
        """
        lat, lon = None, None

        # 1. Use anchor (most recent stable position)
        if self._anchor_lat is not None and self._anchor_lon is not None:
            lat, lon = self._anchor_lat, self._anchor_lon

        # 2. Fallback to gps_server
        if lat is None:
            try:
                from bantz.core.gps_server import gps_server
                loc = gps_server.latest
                if loc and "lat" in loc and "lon" in loc:
                    lat, lon = loc["lat"], loc["lon"]
            except Exception:
                pass

        if lat is None or lon is None:
            return None

        key = name.lower().strip()
        self._load()
        self._data[key] = {
            "label": name.strip(),
            "lat": lat,
            "lon": lon,
            "radius": radius,
        }
        self._persist()

        # Mark this anchor as known now
        self._current_place_key = key
        self._asked_for_anchor = True

        log.info("Saved place '%s' at %.6f, %.6f (r=%dm)", name, lat, lon, int(radius))
        return self._data[key]

    def delete_place(self, name: str) -> bool:
        """Delete a named place. Returns True if it existed."""
        key = name.lower().strip()
        self._load()
        if key in self._data:
            del self._data[key]
            self._persist()
            return True
        return False

    def _persist(self) -> None:
        """Write current data to store or disk."""
        if self._store:
            self._store.save_all(self._data)
        else:
            PLACES_PATH.parent.mkdir(parents=True, exist_ok=True)
            PLACES_PATH.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self._loaded = True

    async def travel_hint(
        self, destination: str, minutes_until: int
    ) -> Optional[str]:
        """
        Contextual travel hint:
          "You're at dorm, class in 20 min — 15 min walk to campus"
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
        """Write places to store (used by --setup places)."""
        self._data = data
        self._persist()

    def is_configured(self) -> bool:
        if self._store:
            return self._store.exists()
        return PLACES_PATH.exists()

    def status_line(self) -> str:
        """Short summary for --doctor."""
        self._load()
        if not self._data:
            return "not configured  → bantz --setup places"
        n = len(self._data)
        names = ", ".join(p.get("label", k) for k, p in list(self._data.items())[:4])
        if n > 4:
            names += f" +{n - 4}"
        cur = self.current_place_label()
        at = f"  [at: {cur}]" if cur else ""
        return f"{n} places: {names}{at}"

    @staticmethod
    def setup_path() -> Path:
        return PLACES_PATH


places = PlaceService()
