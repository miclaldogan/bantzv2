"""
Bantz v2 — Location Service

Priority order:
1. .env manual override (BANTZ_CITY, BANTZ_LAT, BANTZ_LON)
2. GeoClue2 via D-Bus (system location service, requires user permission)
3. ipinfo.io IP geolocation (works without permission, online only)
4. Hardcoded fallback (Ankara, TR)

Fetched once per session, cached in memory.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from bantz.config import config

logger = logging.getLogger(__name__)

IPINFO_URL = "https://ipinfo.io/json"
TIMEOUT = 5.0

FALLBACK = {
    "city": "Ankara",
    "country": "TR",
    "timezone": "Europe/Istanbul",
    "region": "Ankara",
    "lat": 39.9208,
    "lon": 32.8541,
}


@dataclass
class Location:
    city: str
    country: str
    timezone: str
    region: str = ""
    lat: float = 0.0
    lon: float = 0.0
    source: str = "unknown"   # "config" | "geoclue" | "ipinfo" | "fallback"

    @property
    def is_turkey(self) -> bool:
        return self.country == "TR"

    @property
    def display(self) -> str:
        parts = [p for p in [self.city, self.region, self.country] if p]
        return ", ".join(parts)


class LocationService:
    def __init__(self) -> None:
        self._cache: Optional[Location] = None
        self._lock = asyncio.Lock()

    async def get(self) -> Location:
        if self._cache is not None:
            return self._cache
        async with self._lock:
            if self._cache is not None:
                return self._cache
            self._cache = await self._resolve()
            logger.info(f"Location: {self._cache.display} via {self._cache.source}")
            return self._cache

    async def _resolve(self) -> Location:
        # 1. Manual .env override (BANTZ_CITY / BANTZ_LAT / BANTZ_LON)
        if loc := self._from_config():
            return loc

        # 2. places.json primary location (set via --setup places)
        if loc := self._from_places():
            return loc

        # 3. GeoClue2 (Linux system location)
        if loc := await self._from_geoclue():
            return loc

        # 4. ipinfo.io
        if loc := await self._from_ipinfo():
            return loc

        # 5. Fallback
        logger.warning("All location sources failed — using fallback (Ankara, TR)")
        return Location(**FALLBACK, source="fallback")

    def _from_config(self) -> Optional[Location]:
        """Read BANTZ_CITY / BANTZ_LAT / BANTZ_LON from config."""
        city = getattr(config, "location_city", "") or ""
        if not city:
            return None
        return Location(
            city=city,
            country=getattr(config, "location_country", "TR") or "TR",
            timezone=getattr(config, "location_timezone", "Europe/Istanbul") or "Europe/Istanbul",
            region=getattr(config, "location_region", "") or "",
            lat=float(getattr(config, "location_lat", 0.0) or 0.0),
            lon=float(getattr(config, "location_lon", 0.0) or 0.0),
            source="config",
        )

    def _from_places(self) -> Optional[Location]:
        """Read primary location from places.json (set via --setup places)."""
        import json
        from pathlib import Path

        places_path = Path.home() / ".local" / "share" / "bantz" / "places.json"
        if not places_path.exists():
            return None
        try:
            data = json.loads(places_path.read_text(encoding="utf-8"))
            if not data:
                return None

            # Find primary place, or fall back to first place
            place = None
            for v in data.values():
                if v.get("primary"):
                    place = v
                    break
            if not place:
                place = next(iter(data.values()))

            lat = place.get("lat", 0.0)
            lon = place.get("lon", 0.0)
            if lat == 0.0 and lon == 0.0:
                return None

            label = place.get("label", "Unknown")
            return Location(
                city=label,
                country="TR",
                timezone="Europe/Istanbul",
                lat=lat,
                lon=lon,
                source="places",
            )
        except Exception as exc:
            logger.debug(f"places.json read failed: {exc}")
            return None

    async def _from_geoclue(self) -> Optional[Location]:
        """Try GeoClue2 via D-Bus. Requires geoclue2 installed and user permission."""
        try:
            # Run in executor — dbus calls are blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._geoclue_sync
            )
            return result
        except Exception as exc:
            logger.debug(f"GeoClue2 unavailable: {exc}")
            return None

    def _geoclue_sync(self) -> Optional[Location]:
        """Blocking GeoClue2 lookup via python-dbus or gi.repository."""
        try:
            import gi
            gi.require_version("Geoclue", "2.0")
            from gi.repository import Geoclue, GLib

            loop = GLib.MainLoop()
            result: list = []

            def on_location(client, _):
                loc = client.get_location()
                result.append((loc.get_property("latitude"),
                                loc.get_property("longitude")))
                loop.quit()

            client = Geoclue.Simple.new_sync(
                "bantz", Geoclue.AccuracyLevel.CITY, None
            )
            lat = client.get_location().get_property("latitude")
            lon = client.get_location().get_property("longitude")

            # Reverse geocode lat/lon → city (best effort)
            city, country, tz = self._reverse_geocode_sync(lat, lon)
            return Location(
                city=city, country=country, timezone=tz,
                lat=lat, lon=lon, source="geoclue"
            )
        except Exception:
            return None

    def _reverse_geocode_sync(self, lat: float, lon: float) -> tuple[str, str, str]:
        """Simple reverse geocode via nominatim (no API key)."""
        import urllib.request, json as _json
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Bantz/2.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _json.loads(r.read())
        addr = data.get("address", {})
        city = (addr.get("city") or addr.get("town") or
                addr.get("village") or addr.get("county") or "Unknown")
        country = addr.get("country_code", "").upper() or "TR"
        # Timezone from ipinfo for the lat/lon
        tz = FALLBACK["timezone"]
        return city, country, tz

    async def _from_ipinfo(self) -> Optional[Location]:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(IPINFO_URL)
                resp.raise_for_status()
                data = resp.json()

            lat, lon = 0.0, 0.0
            if loc_str := data.get("loc", ""):
                parts = loc_str.split(",")
                if len(parts) == 2:
                    lat, lon = float(parts[0]), float(parts[1])

            return Location(
                city=data.get("city", FALLBACK["city"]),
                country=data.get("country", FALLBACK["country"]),
                timezone=data.get("timezone", FALLBACK["timezone"]),
                region=data.get("region", ""),
                lat=lat, lon=lon,
                source="ipinfo",
            )
        except Exception as exc:
            logger.debug(f"ipinfo.io failed: {exc}")
            return None

    def reset(self) -> None:
        self._cache = None


location_service = LocationService()