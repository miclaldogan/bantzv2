"""
Bantz v2 — Location Service

Priority order:
1. .env manual override (BANTZ_CITY, BANTZ_LAT, BANTZ_LON)
2. Live GPS from phone (via gps_server) — reverse-geocoded to a real city
3. WiFi SSID → places.json mapping
4. places.json primary location
5. Cached phone GPS city (last good fix, persisted — survives the live TTL
   so the correct city sticks between phone pushes)
6. WiFi-BSSID geolocation (nmcli scan → beaconDB, the free MLS successor;
   far more accurate than IP where the APs are mapped)
7. GeoClue2 (system location service)
8. ipinfo.io IP geolocation (online only; unreliable in TR — ISPs route via
   Istanbul, so this is a last resort)
9. If everything fails: location unknown (no wrong-city guess)

GeoIP is unreliable in Turkey, so phone GPS (live or cached) and WiFi-BSSID
geolocation are preferred over IP. Fetched once per session, cached in memory.
Call reset() to re-resolve (e.g. after receiving new GPS data).
"""
from __future__ import annotations

import asyncio
import json as _json_mod
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from bantz.config import config
from bantz.core.secure_io import secure_write_text

logger = logging.getLogger(__name__)

_shared_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient()
    return _shared_client

IPINFO_URL = "https://ipinfo.io/json"
# beaconDB — free, community MLS successor; no API key required. Accepts the
# Mozilla/Google geolocate schema ({wifiAccessPoints:[{macAddress,signalStrength}]}).
BEACONDB_URL = "https://api.beacondb.net/v1/geolocate"
TIMEOUT = 5.0

# Persisted reverse-geocoded phone-GPS city — lets the correct city survive the
# 30-min live-GPS TTL so it doesn't decay back to a wrong GeoIP guess.
GPS_CITY_CACHE = Path.home() / ".local" / "share" / "bantz" / "gps_city.json"
GPS_CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days — stale enough is still better than IP


def _system_timezone() -> str:
    """Best-effort IANA timezone for this machine.

    Used when geolocation can't name a city but the app still needs a valid
    tz (calendar.py feeds Location.timezone straight into pytz.timezone()).
    Never invents a city — only the local clock's zone.
    """
    # /etc/localtime is usually a symlink into …/zoneinfo/<Area>/<City>.
    try:
        parts = Path("/etc/localtime").resolve().parts
        if "zoneinfo" in parts:
            tz = "/".join(parts[parts.index("zoneinfo") + 1:])
            if tz:
                return tz
    except Exception:
        pass
    import os
    return os.environ.get("TZ") or "UTC"


@dataclass
class Location:
    city: str
    country: str
    timezone: str
    region: str = ""
    lat: float = 0.0
    lon: float = 0.0
    source: str = "unknown"   # "config" | "geoclue" | "ipinfo" | "unknown"

    @property
    def is_live(self) -> bool:
        """True when location comes from a real-time source (GPS, WiFi, GeoClue)."""
        return (
            self.source in ("phone_gps", "geoclue", "wifi")
            or self.source.startswith("wifi:")
        )

    @property
    def is_turkey(self) -> bool:
        return self.country == "TR"

    @property
    def display(self) -> str:
        parts = [p for p in [self.city, self.region, self.country] if p]
        if not parts:
            return "location unknown"
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

        # 2. Live GPS from phone (highest real-time priority)
        if loc := await self._from_live_gps():
            return loc

        # 3. WiFi SSID → places.json match
        if loc := self._from_wifi_ssid():
            return loc

        # 4. places.json primary location (set via --setup places)
        if loc := self._from_places():
            return loc

        # 5. Cached phone GPS city — last good fix, so the correct city sticks
        #    between phone pushes instead of decaying to a wrong GeoIP guess.
        if loc := self._from_gps_cache():
            return loc

        # 6. WiFi-BSSID geolocation (nmcli → beaconDB). Accurate where the
        #    nearby access points are mapped; silently skipped otherwise.
        if loc := await self._from_wifi_geolocation():
            return loc

        # 7. GeoClue2 (Linux system location)
        if loc := await self._from_geoclue():
            return loc

        # 8. ipinfo.io (last resort — unreliable in TR)
        if loc := await self._from_ipinfo():
            return loc

        # 9. Nothing worked — be honest rather than guessing a wrong city.
        #    Keep a valid system timezone so time-dependent tools still run.
        logger.warning("All location sources failed — location unknown")
        return Location(
            city="", country="", region="",
            timezone=_system_timezone(),
            lat=0.0, lon=0.0, source="unknown",
        )

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

    async def _from_live_gps(self) -> Optional[Location]:
        """Read live GPS from the phone relay (gps_server) and name the city.

        GPS is ground truth — the only reliable source where GeoIP/WiFi
        databases are blind (e.g. Elazığ). The raw fix is reverse-geocoded to
        a real city and the result persisted so it survives the live TTL.
        """
        try:
            from bantz.core.gps_server import gps_server
            loc_data = gps_server.latest
            if not loc_data:
                return None
            lat = float(loc_data.get("lat", 0.0))
            lon = float(loc_data.get("lon", 0.0))
            if lat == 0.0 and lon == 0.0:
                return None
            acc = round(loc_data.get("accuracy", 0))
            logger.info(f"Live GPS: {lat:.6f}, {lon:.6f} (±{acc}m)")

            # Reverse-geocode coords → real city/country (best effort, off-loop).
            city, country, tz = await asyncio.get_event_loop().run_in_executor(
                None, self._reverse_geocode_sync, lat, lon
            )
            loc = Location(
                city=city, country=country, timezone=tz,
                lat=lat, lon=lon, source="phone_gps",
            )
            self._save_gps_city(loc)  # persist so the city sticks (see #5)
            return loc
        except Exception as exc:
            logger.debug(f"Live GPS read failed: {exc}")
            return None

    @staticmethod
    def _save_gps_city(loc: "Location") -> None:
        """Persist a reverse-geocoded GPS fix for use beyond the live TTL."""
        try:
            import time
            GPS_CITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            secure_write_text(GPS_CITY_CACHE, _json_mod.dumps({
                "city": loc.city, "country": loc.country,
                "timezone": loc.timezone, "region": loc.region,
                "lat": loc.lat, "lon": loc.lon, "ts": time.time(),
            }, ensure_ascii=False))
        except Exception as exc:
            logger.debug(f"GPS city cache write failed: {exc}")

    def _from_gps_cache(self) -> Optional[Location]:
        """Return the last reverse-geocoded phone-GPS city, if still recent.

        A GPS fix from earlier today (or this week) is a far better guess than
        GeoIP, which mislocates Turkish ISPs to Istanbul. Refreshed whenever
        the phone pushes a new fix.
        """
        try:
            import time
            if not GPS_CITY_CACHE.exists():
                return None
            data = _json_mod.loads(GPS_CITY_CACHE.read_text(encoding="utf-8"))
            age = time.time() - float(data.get("ts", 0))
            if age > GPS_CACHE_MAX_AGE:
                return None
            if not data.get("city"):
                return None
            logger.info(
                "Cached GPS city: %s (age %dh)", data["city"], int(age / 3600)
            )
            return Location(
                city=data.get("city", ""),
                country=data.get("country", ""),
                timezone=data.get("timezone") or _system_timezone(),
                region=data.get("region", ""),
                lat=float(data.get("lat", 0.0)),
                lon=float(data.get("lon", 0.0)),
                source="phone_gps_cached",
            )
        except Exception as exc:
            logger.debug(f"GPS city cache read failed: {exc}")
            return None

    async def _from_wifi_geolocation(self) -> Optional[Location]:
        """Locate via nearby WiFi BSSIDs (nmcli scan → beaconDB).

        Sends MAC addresses + signal strengths of nearby access points to
        beaconDB (the free Mozilla-Location-Service successor). Much more
        accurate than IP geolocation where the access points are mapped;
        returns None when they aren't (beaconDB 404). Bounded by a timeout.
        """
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._wifi_geolocation_sync
                ),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.debug("WiFi geolocation timed out")
            return None
        except Exception as exc:
            logger.debug(f"WiFi geolocation failed: {exc}")
            return None

    def _wifi_geolocation_sync(self) -> Optional[Location]:
        import json as _json
        import urllib.error
        import urllib.request

        aps = self._scan_wifi_aps()
        if len(aps) < 2:  # need a couple APs for a meaningful fix
            logger.debug("WiFi geolocation: only %d AP(s), skipping", len(aps))
            return None

        body = _json.dumps(
            {"considerIp": False, "wifiAccessPoints": aps}
        ).encode()
        req = urllib.request.Request(
            BEACONDB_URL, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                res = _json.loads(r.read())
        except urllib.error.HTTPError as e:
            # 404 = APs not in the database (common outside well-mapped areas)
            logger.debug("beaconDB HTTP %s — APs likely unmapped", e.code)
            return None

        loc = res.get("location") or {}
        lat = loc.get("lat")
        lon = loc.get("lng")
        if lat is None or lon is None:
            return None
        logger.info(
            "WiFi geolocation: %.5f, %.5f (±%sm, %d APs)",
            lat, lon, round(res.get("accuracy", 0)), len(aps),
        )
        city, country, tz = self._reverse_geocode_sync(lat, lon)
        return Location(
            city=city, country=country, timezone=tz,
            lat=float(lat), lon=float(lon), source="wifi",
        )

    @staticmethod
    def _scan_wifi_aps() -> list[dict]:
        """Nearby APs as [{macAddress, signalStrength(dBm)}] via nmcli."""
        aps: list[dict] = []
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "BSSID,SIGNAL", "dev", "wifi", "list"],
                capture_output=True, text=True, timeout=8,
            )
            for line in result.stdout.splitlines():
                line = line.replace("\\:", ":")  # nmcli escapes BSSID colons
                bssid, _, sig = line.rpartition(":")
                if len(bssid) != 17:  # not a MAC
                    continue
                try:
                    pct = int(sig)
                except ValueError:
                    continue
                # nmcli SIGNAL is 0-100%; approximate dBm for the API schema.
                dbm = (pct // 2) - 100
                aps.append({
                    "macAddress": bssid.lower(),
                    "signalStrength": dbm,
                })
        except Exception as exc:
            logger.debug(f"nmcli wifi scan failed: {exc}")
        return aps

    @staticmethod
    def _current_ssid() -> Optional[str]:
        """Get the currently connected WiFi SSID via nmcli."""
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                capture_output=True, text=True, timeout=3,
            )
            for line in result.stdout.splitlines():
                if line.startswith("yes:"):
                    ssid = line.split(":", 1)[1].strip()
                    if ssid:
                        return ssid
        except Exception:
            pass
        return None

    def _from_wifi_ssid(self) -> Optional[Location]:
        """Match current WiFi SSID against places.json ssid field."""
        ssid = self._current_ssid()
        if not ssid:
            return None

        places_path = Path.home() / ".local" / "share" / "bantz" / "places.json"
        if not places_path.exists():
            return None
        try:
            data = _json_mod.loads(places_path.read_text(encoding="utf-8"))
            for name, place in data.items():
                if place.get("ssid") == ssid:
                    lat = place.get("lat", 0.0)
                    lon = place.get("lon", 0.0)
                    label = place.get("label", name)
                    logger.info(f"WiFi SSID '{ssid}' → {label}")
                    return Location(
                        city=label,
                        country="TR",
                        timezone="Europe/Istanbul",
                        lat=lat,
                        lon=lon,
                        source=f"wifi:{ssid}",
                    )
        except Exception as exc:
            logger.debug(f"WiFi SSID lookup failed: {exc}")
        return None

    async def _from_geoclue(self) -> Optional[Location]:
        """Try GeoClue2 (system location service). Primary automatic source.

        Bounded by a timeout so a stalled geoclue can't block startup; on
        timeout/failure the caller falls through to IP geolocation.
        """
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._geoclue_sync
                ),
                timeout=8.0,
            )
        except asyncio.TimeoutError:
            logger.debug("GeoClue2 timed out")
            return None
        except Exception as exc:
            logger.debug(f"GeoClue2 unavailable: {exc}")
            return None

    def _geoclue_sync(self) -> Optional[Location]:
        """Blocking GeoClue2 lookup via gi.repository (Geoclue.Simple).

        The `gdbus` CLI cannot be used here: GeoClue2 binds each Client to the
        D-Bus connection that created it, so a Client made in one `gdbus call`
        is destroyed the instant that process exits ("Object does not exist").
        Geoclue.Simple holds a single connection open and blocks until the
        first fix is available.
        """
        try:
            import gi
            gi.require_version("Geoclue", "2.0")
            from gi.repository import Geoclue
        except Exception as exc:
            logger.debug(f"GeoClue2 gi bindings unavailable: {exc}")
            return None

        try:
            simple = Geoclue.Simple.new_sync(
                "bantz", Geoclue.AccuracyLevel.CITY, None,
            )
            loc = simple.get_location()
            if loc is None:
                logger.debug("GeoClue2 returned no location fix")
                return None

            lat = float(loc.get_property("latitude"))
            lon = float(loc.get_property("longitude"))
            if lat == 0.0 and lon == 0.0:
                return None
            acc = round(loc.get_property("accuracy"))
            logger.info("GeoClue2 fix: %.6f, %.6f (±%sm)", lat, lon, acc)

            # Reverse geocode lat/lon → city (best effort)
            city, country, tz = self._reverse_geocode_sync(lat, lon)
            return Location(
                city=city, country=country, timezone=tz,
                lat=lat, lon=lon, source="geoclue",
            )
        except Exception as exc:
            logger.debug(f"GeoClue2 lookup failed: {exc}")
            return None

    def _reverse_geocode_sync(self, lat: float, lon: float) -> tuple[str, str, str]:
        """Simple reverse geocode via nominatim (no API key)."""
        import json as _json
        import urllib.request
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
        country = addr.get("country_code", "").upper()
        # Nominatim doesn't return a tz; use the machine's local zone.
        tz = _system_timezone()
        return city, country, tz

    async def _from_ipinfo(self) -> Optional[Location]:
        try:
            client = _get_client()
            resp = await client.get(IPINFO_URL, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            lat, lon = 0.0, 0.0
            if loc_str := data.get("loc", ""):
                parts = loc_str.split(",")
                if len(parts) == 2:
                    lat, lon = float(parts[0]), float(parts[1])

            # No usable data → let the caller fall through to "unknown"
            # rather than emitting a blank/half-filled location.
            if not data.get("city") and lat == 0.0 and lon == 0.0:
                return None

            return Location(
                city=data.get("city", ""),
                country=data.get("country", ""),
                timezone=data.get("timezone") or _system_timezone(),
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