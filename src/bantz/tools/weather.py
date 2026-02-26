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


class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "Fetches current weather conditions and forecast. "
        "Use for: weather, forecast, temperature, is it raining, how's the weather."
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
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
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

            location_note = f" (auto-detected)" if auto else ""
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