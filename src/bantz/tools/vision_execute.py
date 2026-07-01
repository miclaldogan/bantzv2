"""
Bantz v2 — vision_execute tool

Hands a long multi-step desktop goal to the vision-guided executor
(:class:`bantz.vision_executor.VisionExecutor`): screenshot → decide →
act → repeat. Pre-fetches personal preferences from the user profile so
goals carry resolved values ("Thunderstruck", "firefox") instead of
placeholders.

Typical planner step for "play me some AC/DC" ("bana biraz AC/DC aç"):
    {"tool": "vision_execute",
     "params": {"goal": "play AC/DC on the preferred music service",
                "intent": "play_music", "artist": "AC/DC"}}
"""
from __future__ import annotations

import logging
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

log = logging.getLogger("bantz.tool.vision_execute")

_executor = None  # lazy singleton


def _get_executor():
    global _executor
    if _executor is None:
        from bantz.vision_executor import VisionExecutor
        _executor = VisionExecutor()
    return _executor


async def _resolve_video_id(query: str, music: bool = True) -> str | None:
    """Resolve a song's YouTube videoId server-side, without the browser.

    The search results page embeds ytInitialData with videoIds; the first
    one is the top result. This removes coordinate clicking — a 4B VLM's
    clicks measured 0/6 hits on YT Music's ~30px play affordance (and the
    misses switched the user's browser tabs).
    """
    import re as _re
    from urllib.parse import quote_plus
    import httpx
    url = (f"https://music.youtube.com/search?q={quote_plus(query)}" if music
           else f"https://www.youtube.com/results?search_query={quote_plus(query)}")
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) "
                       "Gecko/20100101 Firefox/130.0"),
        "Accept-Language": "en-US,en;q=0.8",
    }
    def _best_id(html: str, want: str) -> str | None:
        """Pick the videoId whose position is closest to an occurrence of
        the wanted title — the FIRST videoId in the page is often a shelf
        item, not the top result (measured: returned a different AC/DC
        track on a repeat run)."""
        ids = [(m.start(), m.group(1)) for m in
               _re.finditer(r'"videoId"\s*:\s*"([\w-]{11})"', html)]
        if not ids:
            return None
        want_l = want.lower().split()[0] if want else ""
        if want_l:
            title_positions = [m.start() for m in
                               _re.finditer(_re.escape(want_l), html.lower())]
            if title_positions:
                def dist(item):
                    pos, _ = item
                    return min(abs(pos - tp) for tp in title_positions)
                return min(ids, key=dist)[1]
        return ids[0][1]

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers, timeout=10.0)
            vid = _best_id(resp.text, query)
            if vid:
                return vid
    except Exception as exc:
        log.warning("videoId resolution failed: %s", exc)
    # Fallback: plain YouTube search embeds videoIds the same way
    try:
        url2 = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url2, headers=headers, timeout=10.0)
            m = _re.search(r'"videoId"\s*:\s*"([\w-]{11})"', resp.text)
            if m:
                return m.group(1)
    except Exception as exc:
        log.warning("videoId fallback resolution failed: %s", exc)
    return None


def _video_verifier():
    """Deterministic "is a video actually playing?" check.

    YouTube page titles look the same paused or playing, so the only
    trustworthy signal is an active audio stream.
    """
    async def _check() -> bool | str:
        import asyncio as _aio
        import shutil as _shutil
        pactl = _shutil.which("pactl")
        if pactl:
            try:
                proc = await _aio.create_subprocess_exec(
                    pactl, "list", "sink-inputs",
                    stdout=_aio.subprocess.PIPE,
                    stderr=_aio.subprocess.DEVNULL,
                )
                out, _ = await _aio.wait_for(proc.communicate(), 3)
                # Require the BROWSER's stream — any-audio would false-pass
                # on an unrelated app (a call, a local player).
                if b"irefox" in (out or b"") or b"hromium" in (out or b""):
                    return True
            except Exception:
                pass
        return ("the video is not playing — no active audio stream. Click "
                "the BIG play button in the CENTER of the video area (not "
                "the sidebar, not a thumbnail)")
    return _check


def _music_verifier():
    """Deterministic "is music actually playing?" check.

    Two independent signals, either passes:
      1. an active audio stream exists (pactl sink-inputs)
      2. the browser window title carries a track name — YT Music sets
         "<Song> - YouTube Music" while playing; a bare "YouTube Music"
         title means nothing is playing.
    """
    async def _check() -> bool | str:
        import asyncio as _aio
        import json as _json
        import shutil as _shutil

        # Signal 1: live audio stream
        pactl = _shutil.which("pactl")
        if pactl:
            try:
                proc = await _aio.create_subprocess_exec(
                    pactl, "list", "sink-inputs",
                    stdout=_aio.subprocess.PIPE,
                    stderr=_aio.subprocess.DEVNULL,
                )
                out, _ = await _aio.wait_for(proc.communicate(), 3)
                # Require the BROWSER's stream — any-audio would false-pass
                # on an unrelated app (a call, a local player).
                if b"irefox" in (out or b"") or b"hromium" in (out or b""):
                    return True
            except Exception:
                pass

        # Signal 2: browser title carries a track name
        hyprctl = _shutil.which("hyprctl") or "/usr/bin/hyprctl"
        try:
            proc = await _aio.create_subprocess_exec(
                hyprctl, "clients", "-j",
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.DEVNULL,
            )
            out, _ = await _aio.wait_for(proc.communicate(), 3)
            for c in _json.loads(out or b"[]"):
                title = c.get("title", "")
                if "youtube music" in title.lower() and title.lower().strip() not in (
                    "youtube music — original profile — mozilla firefox",
                    "youtube music - mozilla firefox",
                    "youtube music",
                ) and " - youtube music" in title.lower():
                    return True
        except Exception:
            pass

        return ("no song is playing — there is no active audio stream and "
                "the browser title shows no track name. Click the FIRST "
                "search result's thumbnail PLAY button (the round overlay "
                "that appears over the artwork), or click the song row and "
                "then press the space key")
    return _check


class VisionExecuteTool(BaseTool):
    name = "vision_execute"
    description = (
        "Execute a long multi-step desktop task by LOOKING at the screen "
        "each step (screenshot → decide → act). goal=<natural language "
        "task>. Optional: intent=play_music artist=<name> to pull the "
        "user's favorite song/service/browser from their profile. "
        "max_steps=<int> (default 15)."
    )
    risk_level = "moderate"

    async def execute(self, **kwargs: Any) -> ToolResult:
        goal = str(kwargs.get("goal") or kwargs.get("task") or "").strip()
        if not goal:
            return ToolResult(False, "", error="vision_execute needs a goal")

        context: dict[str, Any] = {}
        intent = str(kwargs.get("intent") or "")
        # Fallback: routers sometimes omit intent — detect media goals from
        # the text so the deterministic paths still engage. "video"/"videos"
        # ("open quantum videos", "star videos") is a watch request.
        import re as _re0
        if not intent and ("watch" in goal.lower()
                           or _re0.search(r"\bvideos?\b", goal.lower())):
            intent = "watch_video"
        if not intent and any(
            w in goal.lower() for w in ("music", "listen", "play", "song")
        ):
            intent = "play_music"
        if intent == "play_music" and not kwargs.get("artist"):
            # artist guess: words after listen/play, stopping at service
            # phrases ("on music service", "in youtube", …)
            import re as _re
            m = _re.search(
                r"(?:listen to|listen|play|hear)\s+(?:to\s+)?(?:some\s+)?"
                r"([\w/&' -]{2,30}?)(?=\s+(?:on|in|at|from|via)\b|\s*$|[.,!?])",
                goal, _re.I,
            )
            if m:
                cand = m.group(1).strip()
                # "i want to listen music" / "play some music" carry no real
                # artist — the captured word is a generic noun. Treat as empty
                # so the profile-favorite fallback engages instead of
                # searching YouTube for the literal word "music".
                _GENERIC = {"music", "some music", "a song", "song", "songs",
                            "tunes", "something", "anything", "stuff"}
                if cand.lower() not in _GENERIC:
                    kwargs = {**kwargs, "artist": cand}
        if intent:
            from bantz.core.profile import profile
            context = profile.get_relevant(
                intent, artist=str(kwargs.get("artist") or ""),
            )
            # Make the goal concrete when memory resolved a favorite.
            # Navigate by SEARCH URL (typed in the address bar) instead of
            # clicking the page's search box — small-VLM click coordinates
            # are unreliable; the address bar via ctrl+l is deterministic.
            if intent == "play_music":
                from urllib.parse import quote_plus
                base = context.get("service_url", "https://music.youtube.com")
                browser = context.get("browser", "firefox")
                if context.get("song"):
                    query = f"{context['song']} {context['artist']}"
                elif context.get("artist"):
                    query = str(context["artist"])
                else:
                    query = ""
                if not query:
                    # Generic "play music" and no favorite configured → just
                    # open YT Music's home; don't search the literal word.
                    context["open_home"] = True
                    goal = (
                        f"In {browser}, press ctrl+l, type {base} and press "
                        f"Enter. Done when YouTube Music's home page is open."
                    )
                    video_id = None
                else:
                    video_id = await _resolve_video_id(query)
                if video_id:
                    # Fully deterministic path — navigation needs no vision
                    # model at all (measured: the VLM typed the URL then
                    # chose "wait" forever instead of pressing Enter).
                    # Script it; the verifier is the only judge. The vision
                    # loop below only runs if autoplay didn't start.
                    context["video_id"] = video_id
                    goal = (
                        f"A song page {base}/watch?v={video_id} is already "
                        f"open in {browser} but playback has not started. "
                        f"Click the play button in the player bar at the "
                        f"bottom of the page. Done ONLY when the player "
                        f"shows the song playing."
                    )
                else:
                    goal = (
                        f"In {browser}, press ctrl+l, type "
                        f"{base}/search?q={quote_plus(query)} and press Enter. "
                        f"When the search results load, click the FIRST song "
                        f"result's title (or its play button). Done ONLY when "
                        f"the player bar at the bottom shows a song playing."
                    )
            elif intent == "watch_video":
                from urllib.parse import quote_plus
                import re as _re
                browser = context.get("browser") or "firefox"
                context.setdefault("browser", browser)
                context["service_url"] = "https://www.youtube.com"
                # Extract the topic: drop the leading verb ("open/show/find/
                # watch …") and the trailing "video(s)" noun, so both
                # "watch quantum videos on youtube" and "open quantum videos"
                # resolve the query "quantum".
                q = _re.sub(
                    r"(?i)\b(?:i (?:want|wanna|would like) to|let'?s|let us|"
                    r"can we|wanna)\s+watch\b", " ", goal)
                q = _re.sub(
                    r"(?i)^\s*(?:open|show me|show|find|watch|pull up|put on|"
                    r"play)\b", " ", q)
                q = _re.sub(r"(?i)\bon youtube\b|\btogether\b", " ", q)
                q = _re.sub(r"(?i)\bvideos?\b", " ", q)
                query = _re.sub(r"\s+", " ", q).strip(" .,!?") or goal
                context["query"] = query
                video_id = await _resolve_video_id(query, music=False)
                if video_id:
                    context["video_id"] = video_id
                    goal = (
                        f"A YouTube video page for '{query}' is already open "
                        f"in {browser} but playback has not started. Click "
                        f"the big play button in the CENTER of the video "
                        f"area. Done ONLY when the video is visibly playing "
                        f"(progress bar moving, pause icon shown)."
                    )
                else:
                    goal = (
                        f"In {browser}, press ctrl+l, type "
                        f"https://www.youtube.com/results?search_query="
                        f"{quote_plus(query)} and press Enter. When results "
                        f"load, click the FIRST video's thumbnail. Done ONLY "
                        f"when the video is visibly playing."
                    )
        log.info("vision_execute goal: %.160s | context=%s", goal,
                 {k: v for k, v in context.items() if k != "service_url"})

        try:
            max_steps = int(kwargs.get("max_steps") or 15)
        except (TypeError, ValueError):
            max_steps = 15

        if intent == "play_music":
            verifier = _music_verifier()
        elif intent == "watch_video":
            verifier = _video_verifier()
        else:
            verifier = None
        executor = _get_executor()

        # Deterministic pre-step: when the goal targets the browser, focus
        # (don't ask the vision model to do window management — measured:
        # it pressed bare "super" on Hyprland in a loop until stuck-abort).
        browser = context.get("browser", "")
        if browser and browser.lower() in goal.lower():
            focused = await executor.screen.focus_window(browser)
            log.info("pre-focus %s: %s", browser, "ok" if focused else
                     "not found — vision loop must open it")

        # Scripted "just open YT Music" — generic request with no favorite.
        # Launch the home page directly and return; no vision loop needed.
        if context.get("open_home"):
            import asyncio as _aio
            import shutil as _shutil
            base = context.get("service_url", "https://music.youtube.com")
            browser_bin = _shutil.which(browser or "firefox") or "firefox"
            await _aio.create_subprocess_exec(
                browser_bin, base,
                stdout=_aio.subprocess.DEVNULL, stderr=_aio.subprocess.DEVNULL,
            )
            log.info("scripted open-home via %s %s", browser_bin, base)
            await _aio.sleep(1.0)
            await executor.screen.focus_window(browser or "firefox")
            return ToolResult(
                True, "Opened YouTube Music. Tip: set a favorite artist with "
                "`bantz --setup profile` so I can start a song straight away.",
                data={"mode": "open_home", "url": base},
            )

        # Scripted navigation for resolved songs: hand the URL to the
        # browser binary directly — no synthesized keystrokes (measured:
        # 8ms ydotool typing dropped characters and the garbled URL
        # autocompleted to random history pages). Zero vision calls on the
        # happy path.
        if context.get("video_id") and verifier is not None:
            import asyncio as _aio
            import shutil as _shutil
            sc = executor.screen
            url = (f"{context.get('service_url', 'https://music.youtube.com')}"
                   f"/watch?v={context['video_id']}")
            browser_bin = _shutil.which(browser or "firefox") or "firefox"
            # Fire-and-forget: the browser owns its lifetime from here.
            await _aio.create_subprocess_exec(
                browser_bin, url,
                stdout=_aio.subprocess.DEVNULL,
                stderr=_aio.subprocess.DEVNULL,
            )
            log.info("scripted navigation via %s %s", browser_bin, url)
            await _aio.sleep(1.5)
            await sc.focus_window(browser or "firefox")
            await sc.wait_for_stable(timeout=8.0)
            for attempt in range(10):  # up to ~25s incl. play toggles
                if await verifier() is True:
                    if intent == "watch_video":
                        label = context.get("query") or "the video"
                        msg = f"{label} is playing on YouTube."
                    else:
                        song = context.get("song") or "the song"
                        artist = context.get("artist") or ""
                        label = f"{song}{' by ' + artist if artist else ''}"
                        msg = f"{label} is playing on YouTube Music."
                    return ToolResult(
                        True, msg,
                        data={"video_id": context["video_id"],
                              "mode": "scripted"},
                    )
                if attempt in (2, 5):
                    # Autoplay blocked? space toggles play/pause on both
                    # YouTube and YT Music once the page has focus.
                    await sc.focus_window(browser or "firefox")
                    await sc.key("space")
                    log.info("scripted play toggle (space), attempt %d",
                             attempt)
                await _aio.sleep(2.5)
            log.info("scripted playback unverified — vision loop takes over")

        try:
            result = await executor.execute(
                goal, context, max_steps=max_steps, verifier=verifier,
            )
        except Exception as exc:
            log.error("vision_execute failed: %s", exc)
            return ToolResult(False, "", error=f"vision_execute error: {exc}")

        data = {
            "model": result.model_used,
            "steps": [
                {"n": s.step, "action": s.action, "target": s.target,
                 "ok": s.ok, "observation": s.observation}
                for s in result.steps
            ],
        }
        if result.success:
            return ToolResult(True, result.summary(), data=data)
        return ToolResult(False, result.summary(), data=data, error=result.error)


registry.register(VisionExecuteTool())
