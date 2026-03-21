"""
Bantz v3 — Unified Navigation Pipeline (#123)

Orchestrates AT-SPI → Spatial Cache → Remote VLM → fallback chain into a
single pipeline that ``Brain`` calls with natural language like
"click the search bar in Firefox".

Pipeline order:
    1. Spatial Cache — instant (< 1 ms), if we've seen this element before
    2. AT-SPI       — reliable on GTK/Qt apps, ~50-200 ms
    3. Remote VLM   — screenshot + vision model, ~2-5 s
    4. Give up      — return None, ask user

Why cache first?  It's O(1) with high confidence after the first lookup.
AT-SPI tree walks cost 50-200 ms and can fail on Electron/custom-drawn UIs.
VLM is the most expensive but works on *any* visible UI.

Analytics are stored per-app so Bantz learns which method works best for
each application over time.

Usage:
    from bantz.vision.navigator import navigator

    result = await navigator.navigate_to("firefox", "search bar")
    if result:
        print(f"Found at ({result.x}, {result.y}) via {result.method}")
        await navigator.execute_action("click", "firefox", "search bar")
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bantz.data.connection_pool import get_pool

log = logging.getLogger("bantz.vision.navigator")


# ━━ Data types ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class NavResult:
    """Result of a navigation attempt."""
    found: bool
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    method: str = ""          # "cache", "atspi", "vlm", "none"
    confidence: float = 0.0
    latency_ms: float = 0.0
    app_name: str = ""
    element_label: str = ""
    role: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        if self.width and self.height:
            return (self.x + self.width // 2, self.y + self.height // 2)
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": self.center[0],
            "center_y": self.center[1],
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "latency_ms": round(self.latency_ms, 1),
            "app": self.app_name,
            "element": self.element_label,
            "role": self.role,
        }


@dataclass
class ActionResult:
    """Result of a full GUI action (navigate + interact)."""
    success: bool
    nav: NavResult
    action: str = ""
    message: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = self.nav.to_dict()
        d.update({
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "error": self.error,
        })
        return d


# ━━ Analytics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NavigationAnalytics:
    """Track per-app navigation success rates and method preferences.

    Schema:
        nav_analytics(
            id          INTEGER PRIMARY KEY,
            app_name    TEXT NOT NULL,
            element     TEXT NOT NULL,
            method      TEXT NOT NULL,     -- cache, atspi, vlm, none
            success     INTEGER NOT NULL,  -- 0 or 1
            latency_ms  REAL,
            confidence  REAL,
            created_at  TEXT NOT NULL
        )
    """

    def __init__(self) -> None:
        self._initialized = False

    def init(self, db_path: Path) -> None:
        """Initialize analytics table — reuses main DB."""
        get_pool(db_path)
        with get_pool().connection(write=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nav_analytics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name    TEXT NOT NULL,
                    element     TEXT NOT NULL,
                    method      TEXT NOT NULL,
                    success     INTEGER NOT NULL DEFAULT 0,
                    latency_ms  REAL,
                    confidence  REAL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nav_app
                    ON nav_analytics(app_name)
            """)
        self._initialized = True
        log.debug("Navigation analytics table ready")

    def record(
        self,
        app_name: str,
        element: str,
        method: str,
        success: bool,
        latency_ms: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        """Record a single navigation attempt."""
        if not self._initialized:
            return
        now = datetime.now().isoformat(timespec="seconds")
        with get_pool().connection(write=True) as conn:
            conn.execute(
                """INSERT INTO nav_analytics
                   (app_name, element, method, success, latency_ms, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (app_name.lower(), element.lower(), method, int(success),
                 latency_ms, confidence, now),
            )

    def best_method_for_app(self, app_name: str) -> Optional[str]:
        """Return the method that most often succeeds for this app.

        Returns None if no data or too few samples.
        """
        if not self._initialized:
            return None
        with get_pool().connection() as conn:
            rows = conn.execute(
            """SELECT method, COUNT(*) as cnt, AVG(success) as rate
               FROM nav_analytics
               WHERE app_name = ? AND success = 1
               GROUP BY method
               ORDER BY cnt DESC, rate DESC
               LIMIT 1""",
            (app_name.lower(),),
        ).fetchall()
        if rows and rows[0]["cnt"] >= 3:
            return rows[0]["method"]
        return None

    def app_stats(self, app_name: Optional[str] = None) -> dict:
        """Aggregated stats per method, optionally filtered by app."""
        if not self._initialized:
            return {"enabled": False}
        where = ""
        params: tuple = ()
        if app_name:
            where = "WHERE app_name = ?"
            params = (app_name.lower(),)
        with get_pool().connection() as conn:
            rows = conn.execute(
            f"""SELECT method,
                       COUNT(*) as attempts,
                       SUM(success) as successes,
                       AVG(latency_ms) as avg_latency_ms,
                       AVG(confidence) as avg_confidence
                FROM nav_analytics {where}
                GROUP BY method
                ORDER BY successes DESC""",
            params,
        ).fetchall()
        return {
            "methods": [dict(r) for r in rows],
            "total_attempts": sum(r["attempts"] for r in rows) if rows else 0,
        }

    def close(self) -> None:
        self._initialized = False


# ━━ Navigator Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Navigator:
    """Unified UI element navigation pipeline.

    Tries methods in order of speed:
      1. Spatial Cache  — < 1 ms, persistent across sessions
      2. AT-SPI         — 50-200 ms, native accessibility tree
      3. Remote VLM     — 2-5 s, screenshot + vision model
      4. Fail           — return NavResult(found=False)

    Successful navigations are auto-stored in the Spatial Cache for
    future instant lookups.

    If analytics has enough data, the pipeline reorders methods to
    try the most successful one first for each app.
    """

    def __init__(self) -> None:
        self.analytics = NavigationAnalytics()
        self._initialized = False

    def init(self, db_path: Path) -> None:
        """Initialize — call once at startup (DataLayer handles this)."""
        if self._initialized:
            return
        self.analytics.init(db_path)
        self._initialized = True
        log.debug("Navigator initialized")

    # ── Core Pipeline ─────────────────────────────────────────────────────

    async def navigate_to(
        self,
        app_name: str,
        element_label: str,
        *,
        role_filter: Optional[str] = None,
        preferred_method: Optional[str] = None,
    ) -> NavResult:
        """Find a UI element using the multi-method pipeline.

        Args:
            app_name: Application name (e.g., "firefox", "vscode")
            element_label: Human description (e.g., "search bar", "save button")
            role_filter: Optional AT-SPI role (e.g., "push button")
            preferred_method: Force a specific method ("cache", "atspi", "vlm")

        Returns:
            NavResult with coordinates if found, or found=False
        """
        start = time.monotonic()

        # Determine method order — analytics-optimized or default
        if preferred_method:
            methods = [preferred_method]
        else:
            methods = self._method_order(app_name)

        result = NavResult(
            found=False,
            app_name=app_name,
            element_label=element_label,
            method="none",
        )

        for method in methods:
            try:
                if method == "cache":
                    r = self._try_cache(app_name, element_label)
                elif method == "atspi":
                    r = self._try_atspi(app_name, element_label, role_filter)
                elif method == "vlm":
                    r = await self._try_vlm(app_name, element_label)
                else:
                    continue

                elapsed = (time.monotonic() - start) * 1000

                if r and r.found:
                    r.latency_ms = elapsed
                    r.app_name = app_name
                    r.element_label = element_label

                    # Auto-store in cache if method wasn't cache
                    if method != "cache":
                        self._store_in_cache(r)

                    # Record analytics
                    self.analytics.record(
                        app_name, element_label, method,
                        success=True, latency_ms=elapsed,
                        confidence=r.confidence,
                    )
                    log.info(
                        "nav ✓  %s/%s via %s  (%.0f ms, conf=%.2f)",
                        app_name, element_label, method, elapsed, r.confidence,
                    )
                    return r

                # Record failed attempt for this method
                self.analytics.record(
                    app_name, element_label, method,
                    success=False, latency_ms=(time.monotonic() - start) * 1000,
                )

            except Exception as exc:
                log.debug("nav %s failed for %s/%s: %s", method, app_name, element_label, exc)
                self.analytics.record(
                    app_name, element_label, method,
                    success=False,
                )

        # All methods failed
        elapsed = (time.monotonic() - start) * 1000
        result.latency_ms = elapsed
        log.warning("nav ✗  %s/%s — all methods failed (%.0f ms)", app_name, element_label, elapsed)
        return result

    # ── GUI Actions ───────────────────────────────────────────────────────

    async def execute_action(
        self,
        action: str,
        app_name: str,
        element_label: str,
        *,
        text: str = "",
        role_filter: Optional[str] = None,
    ) -> ActionResult:
        """Navigate to element then perform an action.

        Supported actions:
            click, double_click, right_click, type, focus

        For "type": navigates → clicks → types text.
        For "focus": just focuses the window (no element needed).
        """
        # Focus-only doesn't need navigation
        if action == "focus":
            return await self._action_focus(app_name)

        # Navigate to the element
        nav = await self.navigate_to(app_name, element_label, role_filter=role_filter)

        if not nav.found:
            return ActionResult(
                success=False,
                nav=nav,
                action=action,
                message="",
                error=f"Could not find '{element_label}' in {app_name}. "
                       f"Tried: {', '.join(self._method_order(app_name))}.",
            )

        cx, cy = nav.center

        # Perform the action at the found coordinates
        try:
            if action == "click":
                await self._do_click(cx, cy)
            elif action == "double_click":
                await self._do_double_click(cx, cy)
            elif action == "right_click":
                await self._do_right_click(cx, cy)
            elif action == "type":
                await self._do_click(cx, cy)
                if text:
                    await self._do_type(text)
            else:
                return ActionResult(
                    success=False, nav=nav, action=action,
                    error=f"Unknown action: {action}",
                )

            return ActionResult(
                success=True,
                nav=nav,
                action=action,
                message=(
                    f"{action} on '{element_label}' in {app_name} "
                    f"at ({cx}, {cy}) via {nav.method}"
                ),
            )
        except Exception as exc:
            log.error("Action %s failed at (%d,%d): %s", action, cx, cy, exc)
            return ActionResult(
                success=False, nav=nav, action=action,
                error=f"Action failed: {exc}",
            )

    # ── Method Implementations ────────────────────────────────────────────

    def _method_order(self, app_name: str) -> list[str]:
        """Determine method order — default or analytics-optimized."""
        default = ["cache", "atspi", "vlm"]
        best = self.analytics.best_method_for_app(app_name)
        if best and best in default:
            # Promote best method to first, keep others
            order = [best] + [m for m in default if m != best]
            return order
        return default

    def _try_cache(self, app_name: str, label: str) -> Optional[NavResult]:
        """Step 1: Spatial Cache lookup."""
        try:
            from bantz.vision.spatial_cache import spatial_db
            entry = spatial_db.lookup(app_name, label)
            if entry and not entry.is_expired:
                return NavResult(
                    found=True,
                    x=entry.x, y=entry.y,
                    width=entry.width, height=entry.height,
                    method="cache",
                    confidence=entry.effective_confidence,
                    role=entry.role,
                )
        except Exception as exc:
            log.debug("Cache lookup failed: %s", exc)
        return None

    def _try_atspi(
        self, app_name: str, label: str, role_filter: Optional[str] = None,
    ) -> Optional[NavResult]:
        """Step 2: AT-SPI accessibility tree search."""
        try:
            from bantz.tools.accessibility import find_element
            elem = find_element(app_name, label, role_filter=role_filter)
            if elem and elem.get("bounds"):
                b = elem["bounds"]
                cx, cy = elem.get("center", (b["x"] + b["width"] // 2, b["y"] + b["height"] // 2))
                return NavResult(
                    found=True,
                    x=b["x"], y=b["y"],
                    width=b["width"], height=b["height"],
                    method="atspi",
                    confidence=elem.get("score", 1.0),
                    role=elem.get("role", ""),
                )
        except Exception as exc:
            log.debug("AT-SPI lookup failed: %s", exc)
        return None

    async def _try_vlm(self, app_name: str, label: str) -> Optional[NavResult]:
        """Step 3: Remote VLM screenshot analysis."""
        try:
            from bantz.config import config
            if not config.vlm_enabled:
                return None

            from bantz.vision.screenshot import capture_window_base64, capture_base64
            from bantz.vision.remote_vlm import analyze_screenshot

            # Try capturing the specific app window, fall back to full screen
            img_b64 = None
            try:
                img_b64 = await capture_window_base64(app_name)
            except Exception:
                pass
            if not img_b64:
                img_b64 = await capture_base64()
            if not img_b64:
                return None

            vlm_result = await analyze_screenshot(img_b64, label=label)
            if not vlm_result.success or not vlm_result.elements:
                return None

            # Try to find the specific element by label
            match = vlm_result.find(label)
            if not match:
                match = vlm_result.best
            if not match:
                return None

            # Store all detected elements in cache (background enrichment)
            self._store_vlm_elements(app_name, vlm_result.elements)

            return NavResult(
                found=True,
                x=match.x, y=match.y,
                width=match.width, height=match.height,
                method="vlm",
                confidence=match.confidence * 0.7,  # Discount VLM confidence
                role=match.role,
            )
        except Exception as exc:
            log.debug("VLM lookup failed: %s", exc)
        return None

    # ── Cache Storage ─────────────────────────────────────────────────────

    def _store_in_cache(self, nav: NavResult) -> None:
        """Auto-store a successful navigation result in Spatial Cache."""
        try:
            from bantz.vision.spatial_cache import spatial_db
            source_map = {"atspi": "atspi", "vlm": "vlm"}
            source = source_map.get(nav.method, "manual")
            spatial_db.store(
                nav.app_name, nav.element_label,
                x=nav.x, y=nav.y,
                width=nav.width, height=nav.height,
                role=nav.role, source=source,
                confidence=nav.confidence,
            )
        except Exception as exc:
            log.debug("Cache store failed: %s", exc)

    def _store_vlm_elements(self, app_name: str, elements: list) -> None:
        """Store all VLM-detected elements for future lookups."""
        try:
            from bantz.vision.spatial_cache import spatial_db
            for elem in elements:
                spatial_db.store(
                    app_name, elem.label,
                    x=elem.x, y=elem.y,
                    width=elem.width, height=elem.height,
                    role=elem.role, source="vlm",
                    confidence=elem.confidence * 0.7,
                )
        except Exception as exc:
            log.debug("VLM elements cache store failed: %s", exc)

    # ── Input Actions ─────────────────────────────────────────────────────

    async def _do_click(self, x: int, y: int) -> None:
        """Click at coordinates using input_control backend."""
        from bantz.tools.input_control import click
        await click(x, y)

    async def _do_double_click(self, x: int, y: int) -> None:
        from bantz.tools.input_control import double_click
        await double_click(x, y)

    async def _do_right_click(self, x: int, y: int) -> None:
        from bantz.tools.input_control import right_click
        await right_click(x, y)

    async def _do_type(self, text: str) -> None:
        from bantz.tools.input_control import type_text
        from bantz.config import config
        interval = config.input_type_interval_ms / 1000.0
        await type_text(text, interval=interval)

    async def _action_focus(self, app_name: str) -> ActionResult:
        """Focus a window without element navigation."""
        try:
            from bantz.tools.accessibility import focus_window
            ok = focus_window(app_name)
            nav = NavResult(found=ok, app_name=app_name, element_label="", method="atspi")
            return ActionResult(
                success=ok,
                nav=nav,
                action="focus",
                message=f"Focused {app_name}" if ok else "",
                error="" if ok else f"Could not focus {app_name}",
            )
        except Exception as exc:
            nav = NavResult(found=False, app_name=app_name, element_label="", method="none")
            return ActionResult(
                success=False, nav=nav, action="focus",
                error=f"Focus failed: {exc}",
            )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Navigation analytics summary."""
        base = self.analytics.app_stats()
        base["initialized"] = self._initialized
        return base

    def close(self) -> None:
        self.analytics.close()


# ── Module singleton ──────────────────────────────────────────────────────────

navigator = Navigator()
