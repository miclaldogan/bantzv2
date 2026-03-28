"""
Bantz v2 — Weather Tool
Fetches current weather + 3-day forecast from wttr.in.
Auto-detects city via LocationService. No API key needed.
"""
from __future__ import annotations

from typing import Any

import httpx

from bantz.core.location import location_service
from bantz.tools import BaseTool, ToolResult, registry

TIMEOUT = 8.0

# Shared client to leverage HTTP connection pooling, reducing TCP/TLS overhead
# and speeding up repeated tool executions.
_shared_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient()
    return _shared_client


class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "Fetches current weather conditions and forecast. "
        "Use for: weather, forecast, temperature, is it raining, how's the weather. "
        "Uses the user's configured location if no city is specified."
    )
    risk_level = "safe"

    async def execute(self, city: str = "", **kwargs: Any) -> ToolResult:
        # Resolve city — explicit arg wins, else auto-detect
        if not city:
            loc = await location_service.get()
            city = loc.city
            auto = True
        else:
            auto = False

        try:
            data = await self._fetch(city)
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"Weather fetch failed: {exc}")

        output = self._format(data, city, auto)
        return ToolResult(
            success=True,
            output=output,
            data={"city": city, "raw": data},
        )

    async def _fetch(self, city: str) -> dict:
        """Fetch JSON from wttr.in."""
        url = f"https://wttr.in/{city}?format=j1"
        client = _get_client()
        resp = await client.get(url, headers={"Accept": "application/json"}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _format(self, data: dict, city: str, auto: bool) -> str:
        try:
            current = data["current_condition"][0]
            temp_c  = current["temp_C"]
            feels   = current["FeelsLikeC"]
            desc    = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            wind_kmph = current["windspeedKmph"]

            # 3-day forecast
            forecast_lines = []
            for day in data.get("weather", [])[:3]:
                date     = day["date"]
                max_c    = day["maxtempC"]
                min_c    = day["mintempC"]
                day_desc = day["hourly"][4]["weatherDesc"][0]["value"]  # noon
                forecast_lines.append(f"  {date}: {min_c}°C–{max_c}°C, {day_desc}")

            location_note = " (auto-detected)" if auto else ""
            forecast_str = "\n".join(forecast_lines) if forecast_lines else "  No forecast data"

            return (
                f"📍 {city}{location_note}\n"
                f"🌡  {temp_c}°C (feels like {feels}°C) — {desc}\n"
                f"💧 Humidity: {humidity}%  💨 Wind: {wind_kmph} km/h\n"
                f"\n3-day forecast:\n{forecast_str}"
            )
        except (KeyError, IndexError) as exc:
            return f"Weather data received but could not be parsed: {exc}"


registry.register(WeatherTool())


# ── Digest helper ─────────────────────────────────────────────────────────────

async def tomorrow_forecast() -> str | None:
    """Return a short forecast string for tomorrow, or None on failure."""
    try:
        tool = WeatherTool()
        loc = await location_service.get()
        data = await tool._fetch(loc.city)
        days = data.get("weather", [])
        if len(days) < 2:
            return None
        day = days[1]  # tomorrow
        max_c = day["maxtempC"]
        min_c = day["mintempC"]
        desc = day["hourly"][4]["weatherDesc"][0]["value"]  # noon
        return f"{min_c}°C–{max_c}°C, {desc}"
    except Exception:
        return None