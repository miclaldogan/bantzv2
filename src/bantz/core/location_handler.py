"""
Bantz — Location Handler (#225)

Extracts all GPS / place-management side-effects from ``brain.py`` into a
standalone module.  These functions interact with:
  - ``bantz.core.location``   — IP / GeoClue location service
  - ``bantz.core.gps_server`` — phone GPS relay
  - ``bantz.core.places``     — named-place CRUD (PlaceStore)

None of them touch the LLM, routing, or memory — they are pure
side-effect handlers that return formatted strings for the assistant.

Closes #225 (Part 2-B of #218).
"""
from __future__ import annotations

import logging

log = logging.getLogger("bantz.core.location_handler")


async def handle_location() -> str:
    """Handle 'where am i' queries — show GPS / location info."""
    from bantz.core.location import location_service
    from bantz.core.places import places as _places

    # Check phone GPS first — it's the most accurate source
    gps_loc = None
    try:
        from bantz.core.gps_server import gps_server
        gps_loc = gps_server.latest
    except Exception:
        pass

    try:
        loc = await location_service.get()
    except Exception:
        loc = None

    lines: list[str] = []

    # Show current named place first if any
    cur_label = _places.current_place_label()
    if cur_label:
        lines.append(f"📌 You're at: {cur_label}")

    # Prefer phone GPS as primary when available
    if gps_loc:
        acc = round(gps_loc.get("accuracy", 0))
        lines.append(
            f"📍 Phone GPS: {gps_loc['lat']:.6f}, {gps_loc['lon']:.6f} (±{acc}m)"
        )
    elif loc and loc.is_live:
        lines.append(f"📍 {loc.display}")
        if loc.lat and loc.lon:
            lines.append(f"   Coordinates: {loc.lat:.6f}, {loc.lon:.6f}")
        lines.append(f"   Source: {loc.source}")
    else:
        lines.append(
            "I can't pinpoint where you are right now — "
            "I need your phone GPS to figure that out."
        )
        try:
            from bantz.core.gps_server import gps_server
            lines.append(
                f"Open {gps_server.url} on your phone and "
                f"hit 'Share Location' so I can see where you are."
            )
        except Exception:
            pass

    return "\n".join(lines)


async def handle_save_place(name: str) -> str:
    """Save current GPS position as a named place."""
    from bantz.core.places import places as _places

    result = _places.save_here(name)
    if result:
        lat = result.get("lat", 0.0)
        lon = result.get("lon", 0.0)
        return (
            f"📌 Saved '{name}' as a place!\n"
            f"   Coordinates: {lat:.6f}, {lon:.6f}\n"
            f"   Radius: {result.get('radius', 100)}m"
        )
    return "❌ No GPS data — couldn't save location. Is the phone GPS on?"


async def handle_list_places() -> str:
    """List all saved places."""
    from bantz.core.places import places as _places

    all_p = _places.all_places()
    if not all_p:
        return "No saved places yet. Say 'save here as X' to save one."
    lines = ["📌 Saved places:"]
    for key, p in all_p.items():
        label = p.get("label", key)
        lat = p.get("lat", 0.0)
        lon = p.get("lon", 0.0)
        radius = p.get("radius", 100)
        marker = " ⬅ you are here" if key == _places._current_place_key else ""
        lines.append(f"  • {label} ({lat:.4f}, {lon:.4f}, r={radius}m){marker}")
    return "\n".join(lines)


async def handle_delete_place(name: str) -> str:
    """Delete a saved place."""
    from bantz.core.places import places as _places

    if _places.delete_place(name):
        return f"📌 '{name}' deleted."
    return f"❌ No saved place named '{name}' found."
