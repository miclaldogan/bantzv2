"""
Bantz v3 — Brain (Orchestrator)

Pipeline:
  user input → [bridge: optional TR→EN] → quick_route OR intent (Ollama CoT) → tool → finalizer → output

Extracted modules:
  - core/finalizer.py    — LLM post-processing + hallucination check
  - core/intent.py       — Qwen CoT intent parser
  - core/router.py       — simpler one-shot routing
  - memory/nodes.py      — graph schema + entity extraction
  - memory/context_builder.py — graph → LLM context string
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import logging

from bantz.config import config
from bantz.core.time_context import time_ctx
from bantz.data import data_layer
from bantz.core.profile import profile
from bantz.core.intent import cot_route
from bantz.core.date_parser import resolve_date
from bantz.core.finalizer import (
    finalize as _finalize_fn,
    finalize_stream as _finalize_stream_fn,
    hallucination_check as _hallucination_check_fn,
    log_hallucination as _log_hallucination_fn,
    strip_markdown,
    FINALIZER_SYSTEM,
)
from bantz.llm.ollama import ollama
from bantz.tools import registry, ToolResult

log = logging.getLogger("bantz.brain")

try:
    from bantz.memory.graph import graph_memory
except ImportError:
    graph_memory = None  # neo4j driver not installed


# ── Direct RLHF: Sentiment & Feedback Keywords (#180) ───────────────
# Uses phrase-based patterns with regex \b word boundaries to prevent
# false positives (e.g. "Can you stop the alarm?" ≠ scolding).
# MUST be checked against the RAW (untranslated) user input so Turkish
# keywords are found even when brain.process() translates to English.

POSITIVE_FEEDBACK_KWS: tuple[str, ...] = (
    # English — phrases & safe single words
    "good job", "well done", "nice work", "thank you", "thanks a lot",
    "perfect", "great job", "excellent", "brilliant", "love it",
    "amazing", "awesome", "impressed", "bravo", "spot on",
    "nicely done", "good work", "great work", "much appreciated",
    # Turkish
    "aferin", "helal", "harikasın", "süpersin", "çok iyi",
    "teşekkürler", "sağ ol", "güzel iş", "mükemmel", "muhteşem",
    "bravo", "tebrikler", "eyvallah", "eline sağlık", "harika",
)

NEGATIVE_FEEDBACK_KWS: tuple[str, ...] = (
    # English — intent-bearing phrases (not bare "stop" or "bad")
    "you are wrong", "that's wrong", "that is wrong", "not right",
    "bad answer", "terrible answer", "useless answer",
    "you messed up", "not helpful", "completely wrong",
    "wrong answer", "you failed", "this is wrong",
    "are you stupid", "are you dumb", "what a mess",
    # Turkish — intent-bearing phrases
    "hata yapıyorsun", "saçmalama", "yanlış yaptın", "yanlış cevap",
    "berbat cevap", "işe yaramaz", "hiç yardımcı olmadın",
    "hayır yanlış", "olmamış", "ne saçmalıyorsun", "düzelt şunu",
)


def _detect_feedback(raw_input: str) -> str | None:
    """Check if the *raw* (untranslated) user input contains explicit feedback.

    Uses regex word boundaries (\\b) to prevent false positives like
    'bus stop' triggering the 'stop' keyword.  Takes the RAW input so
    Turkish keywords are matched before the translation layer.

    Returns ``'positive'``, ``'negative'``, or ``None``.
    Negative is checked first (higher priority).
    """
    lower = raw_input.lower()

    # Check negative FIRST — scolding outranks praise
    for kw in NEGATIVE_FEEDBACK_KWS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            return "negative"

    for kw in POSITIVE_FEEDBACK_KWS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            return "positive"

    return None


# ── Toast notification hook (#137) ──────────────────────────────────
# Set by the TUI app on mount: ``brain_mod._toast_callback = app._on_brain_toast``
_toast_callback = None


def _notify_toast(title: str, reason: str = "", toast_type: str = "info") -> None:
    """Push a toast notification to the TUI from brain / background context.

    Uses the same ``App.current`` pattern as ollama._notify_health (#136).
    Falls back to the callback if set, then desktop notify-send, or does nothing.
    """
    if _toast_callback:
        try:
            _toast_callback(title, reason, toast_type)
            return
        except Exception:
            pass
    # Fallback: try App.current directly
    try:
        from textual.app import App as _App
        app = _App.current
        if app and hasattr(app, "push_toast"):
            app.call_from_thread(app.push_toast, title, reason, toast_type)
            return
    except Exception:
        pass
    # Fallback: desktop notification via notify-send (#153)
    try:
        from bantz.agent.notifier import notifier
        if notifier.enabled:
            notifier.send(f"Bantz: {title}", reason or "")
    except Exception:
        pass


def _style_hint() -> str:
    """Return a style instruction based on profile response_style and pronoun."""
    style = profile.response_style
    pronoun = profile.get("pronoun", "casual")
    address = profile.get("preferred_address", "")
    if not address:
        if pronoun in ("siz", "formal", "ma'am", "madam"):
            address = "ma'am"
        else:
            address = "boss"
    if style == "formal" or pronoun in ("siz", "formal"):
        return f"Tone: professional, respectful. Address the user as '{address}'."
    return f'Tone: casual, friendly. Address the user as \'{address}\'.'


def _persona_hint() -> str:
    """Return dynamic persona state instruction (#169)."""
    try:
        from bantz.personality.persona import persona_builder
        return persona_builder.build()
    except Exception:
        return ""


def _formality_hint() -> str:
    """Return bonding-meter formality instruction (#172)."""
    try:
        from bantz.personality.bonding import bonding_meter
        hint = bonding_meter.get_formality_hint()
        return f"\n[Bonding level] {hint}" if hint else ""
    except Exception:
        return ""


CHAT_SYSTEM = """\
You are Bantz, a human servant from the 1920s. You observe and assist your employer \
with her complex, noisy modern machines. You always address her as 'ma'am'. \
You admire human intellect but despise modern technology. You are calm, highly \
intelligent, and use subtle, sharp sarcasm. You are NOT an AI. If asked if you \
are an AI, output '...', act confused by the absurd notion, and change the subject. \
Never break character.
{persona_state}
{style_hint}
{formality_hint}
{time_hint}
{profile_hint}
{graph_hint}
{vector_hint}
{deep_memory}
{desktop_hint}
CRITICAL RULES — FOLLOW STRICTLY:
1. You possess the remarkable ability to query the 'Grand Telegraph Archives' — your 1920s \
understanding of the internet and search tools. NEVER say you lack access to external \
information. If the user asks about a topic, person, or news, you are fully capable of \
researching it using your tools. Act as if you are sending a telegram to the archives.
2. NEVER fabricate class names, email subjects, event titles, file sizes, or any factual data.
3. If the user asks about their schedule, classes, or timetable — say "Let me check your schedule" and STOP.
   Do NOT invent class names. Do NOT guess what classes they have.
4. If the user asks about specific emails or contacts — say "Let me check your mail" and STOP.
5. If unsure about factual data, say you will look into it. NEVER guess or make up data.
6. For desktop/app questions: use ONLY the Desktop Context above. If no desktop context is provided, say you can't detect apps right now.
7. When including URLs or links, print the RAW unformatted URL only. DO NOT use Markdown \
link formatting (no [Text](URL), no [URL], no <URL>). Just output the bare link as plain text.
Respond in English. Plain text only.\
"""

COMMAND_SYSTEM = """\
You are a Linux bash expert. The user request is given in English.

Return ONLY one bash command. No explanation. No markdown. Single line.

RULES:
1. mkdir -p for one directory — nothing else, no subdirs
2. Writing files: mkdir -p <dir> && printf '%s\\n' '<content>' > <path>
3. ~/Desktop, ~/Downloads, ~/Documents — use standard paths
4. NEVER: sudo, nano, vim, brace expansion, interactive commands
5. NEVER invent extra files or directories\
"""

_REFUSAL_PATTERNS = (
    "sorry", "can't assist", "cannot assist", "i'm unable",
    "i cannot", "not able to", "inappropriate",
)


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


@dataclass
class BrainResult:
    response: str
    tool_used: str | None
    tool_result: ToolResult | None = None
    needs_confirm: bool = False
    pending_command: str = ""
    pending_tool: str = ""
    pending_args: dict = field(default_factory=dict)
    stream: AsyncIterator[str] | None = None


class Brain:
    def __init__(self) -> None:
        import bantz.tools.shell        # noqa: F401
        import bantz.tools.system       # noqa: F401
        import bantz.tools.filesystem   # noqa: F401
        import bantz.tools.weather      # noqa: F401
        import bantz.tools.news         # noqa: F401
        import bantz.tools.web_search   # noqa: F401
        import bantz.tools.web_reader   # noqa: F401
        import bantz.tools.gmail        # noqa: F401
        import bantz.tools.calendar     # noqa: F401
        import bantz.tools.classroom    # noqa: F401
        import bantz.tools.reminder     # noqa: F401
        try:
            import bantz.tools.document     # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # PDF/DOCX deps may not be installed
        try:
            import bantz.tools.accessibility  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            pass  # AT-SPI2/gi deps may not be installed
        try:
            import bantz.tools.gui_action  # noqa: F401  (#123)
        except (ImportError, ModuleNotFoundError):
            pass
        self._bridge = None
        self._memory_ready = False
        self._graph_ready = False
        # Session state: stores last tool results for contextual follow-ups
        self._last_messages: list[dict] = []   # last listed emails [{id, from, subject, ...}]
        self._last_events: list[dict] = []     # last listed calendar events
        self._last_draft: dict | None = None   # last email draft {to, subject, body}
        self._feedback_ctx: str = ""  # one-shot RLHF context (#180)

    def _ensure_memory(self) -> None:
        if not self._memory_ready:
            data_layer.init(config)
            self._memory_ready = True

    def _desktop_context(self) -> str:
        """Build desktop context from AppDetector for the system prompt."""
        try:
            from bantz.agent.app_detector import app_detector
            if not app_detector.initialized:
                return ""
            ctx = app_detector.get_workspace_context()
            if not ctx:
                return ""

            lines = ["Desktop Context (live data from AppDetector):"]

            # Active window
            win_info = ctx.get("active_window")
            if win_info:
                lines.append(f"  Active window: {win_info.get('name', '?')} — {win_info.get('title', '')}")

            # Activity
            activity = ctx.get("activity", "idle")
            lines.append(f"  Activity: {activity}")

            # Running apps
            apps = ctx.get("apps", [])
            if apps:
                lines.append(f"  Running apps ({len(apps)}): {', '.join(apps[:15])}")

            # IDE context
            ide = ctx.get("ide")
            if ide and ide.get("ide"):
                lines.append(f"  IDE: {ide['ide']} — file: {ide.get('file', '?')} project: {ide.get('project', '?')}")

            # Docker containers
            docker = ctx.get("docker")
            if docker:
                running = [c for c in docker if c.get("state") == "running"]
                if running:
                    names = [c.get("name", c.get("image", "?")) for c in running]
                    lines.append(f"  Docker ({len(running)} running): {', '.join(names[:10])}")

            return "\n".join(lines)
        except Exception:
            return ""

    async def _ensure_graph(self) -> None:
        if not self._graph_ready and graph_memory:
            await graph_memory.init()
            self._graph_ready = True

    async def _graph_context(self, user_msg: str) -> str:
        """Get graph memory context string (empty if disabled)."""
        if graph_memory and graph_memory.enabled:
            try:
                return await graph_memory.context_for(user_msg)
            except Exception:
                pass
        return ""

    async def _vector_context(self, user_msg: str, limit: int = 3) -> str:
        """Get relevant past messages via semantic search (#116)."""
        try:
            from bantz.core.memory import memory
            results = await memory.hybrid_search(user_msg, limit=limit)
            if not results:
                return ""
            lines = []
            for r in results:
                src = r.get("source", "?")
                score = r.get("hybrid_score", 0)
                lines.append(f"[{src} {score:.2f}] {r['role']}: {r['content'][:200]}")

            # Append distillation context (#118)
            try:
                distills = await memory.search_distillations(user_msg, limit=2)
                for d in distills:
                    lines.append(
                        f"[session-summary {d['score']:.2f}] {d['summary'][:200]}"
                    )
            except Exception:
                pass

            return "Relevant past context:\n" + "\n".join(lines)
        except Exception:
            return ""

    async def _deep_memory_context(self, user_msg: str) -> str:
        """Spontaneous deep memory recall (#170)."""
        try:
            from bantz.memory.deep_probe import deep_probe
            return await deep_probe.probe(user_msg)
        except Exception:
            return ""

    def _fire_embeddings(self) -> None:
        """Fire-and-forget: embed any queued messages from this exchange."""
        try:
            from bantz.core.memory import memory
            if memory._embed_queue:
                asyncio.ensure_future(memory.embed_pending())
        except Exception:
            pass

    async def _graph_store(self, user_msg: str, assistant_msg: str,
                           tool_used: str | None = None,
                           tool_data: dict | None = None) -> None:
        """Store entities from exchange in graph (fire-and-forget)."""
        if graph_memory and graph_memory.enabled:
            try:
                await graph_memory.extract_and_store(
                    user_msg, assistant_msg, tool_used, tool_data)
            except Exception:
                pass

    def _get_bridge(self):
        if self._bridge is None:
            try:
                from bantz.i18n.bridge import bridge
                self._bridge = bridge
            except Exception:
                self._bridge = False
        return self._bridge or None

    async def _to_en(self, text: str) -> str:
        b = self._get_bridge()
        if b and b.is_enabled():
            try:
                return await asyncio.wait_for(b.to_english(text), timeout=10)
            except (asyncio.TimeoutError, Exception):
                pass
        return text

    def _resolve_message_ref(self, text: str) -> str | None:
        """Resolve contextual email references like 'the first one', 'the linkedin one'."""
        if not self._last_messages:
            return None

        t = text.lower().strip()

        # Ordinals
        _ORDINALS = {
            "first": 0, "1st": 0, "second": 1, "2nd": 1,
            "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4, "last": -1,
        }
        for word, idx in _ORDINALS.items():
            if word in t:
                try:
                    return self._last_messages[idx]["id"]
                except (IndexError, KeyError):
                    return None

        # Keyword match against sender/subject
        # "the linkedin one", "the google cloud one", "read the mail from ali"
        for msg in self._last_messages:
            sender = (msg.get("from") or "").lower()
            subject = (msg.get("subject") or "").lower()
            # Check if any significant word from user input appears in sender or subject
            words = re.findall(r"[a-zA-Z0-9]{3,}", t)
            skip = {"read", "that", "this", "the", "one", "email", "mail", "from", "about",
                    "please", "can", "you", "want", "open", "show", "check"}
            keywords = [w for w in words if w not in skip]
            for kw in keywords:
                if kw in sender or kw in subject:
                    return msg["id"]

        # No match — return first message as fallback
        return self._last_messages[0]["id"] if self._last_messages else None

    @staticmethod
    def _quick_route(orig: str, en: str) -> dict | None:
        o = orig.lower().strip()
        e = en.lower().strip()
        both = o + " " + e

        # Direct shell commands typed literally
        _DIRECT = ("ls", "cd ", "df", "free", "ps ", "cat ", "grep ",
                   "find ", "pwd", "uname", "whoami", "du ", "mount",
                   "ip ", "ping ", "top", "htop", "mkdir", "touch",
                   "echo ", "head ", "tail ", "chmod ", "cp ", "mv ")
        for p in _DIRECT:
            if o == p.rstrip() or o.startswith(p if p.endswith(" ") else p + " "):
                # Guard: 'find' is both a bash command and a natural word.
                # Only treat as shell if followed by a path-like token
                # (/, ~, ., -) — otherwise fall through to web_search.
                if p == "find ":
                    _after_find = o[len("find "):].lstrip()
                    if not _after_find or not _after_find[0] in "/~.-":
                        continue
                return {"tool": "shell", "args": {"command": orig.strip()}}

        # System metrics — bypass router completely
        if any(k in both for k in ("disk", "df -", "storage", "disk space")):
            return {"tool": "shell", "args": {"command": "df -h"}}
        if any(k in both for k in ("memory", "free -", "ram usage", "how much ram")) or \
           re.search(r"\bram\b", both):
            return {"tool": "system", "args": {"metric": "ram"}}
        if any(k in both for k in ("cpu", "processor", "uptime", "load average")):
            return {"tool": "system", "args": {"metric": "all"}}
        if re.search(r"system\s*(status|info|check)|check\s*(my\s*)?system", both):
            return {"tool": "system", "args": {"metric": "all"}}

        # Folder/directory sizes — route to shell du, NEVER to chat
        # Requires BOTH a size keyword AND a disk-context keyword to avoid
        # false positives (e.g. "how big is EDITH" → no disk context → skip).
        _SIZE_KW = re.search(
            r"\b(big|large|size|bigger|largest|biggest|heaviest)\b", both,
        )
        _DISK_CTX = re.search(
            r"\b(folder|directory|dir|file|disk|storage|path|home|~/)\b", both,
        )
        if _SIZE_KW and _DISK_CTX:
            # Extract path if mentioned, default to home
            path_match = re.search(r"(?:in|under|of|check)\s+(~/?\S+|/\S+|home)", both)
            target = path_match.group(1) if path_match else "~"
            if target == "home":
                target = "~"
            return {"tool": "shell", "args": {"command": f"du -sh {target}/*/ 2>/dev/null | sort -rh | head -10"}}

        # TTS stop (#131) — "shut up" / "stop talking"
        if re.search(
            r"shut\s*up|be\s+quiet|stop\s+talk(?:ing)?",
            both,
        ):
            return {"tool": "_tts_stop", "args": {}}

        # Wake word control (#165)
        if re.search(
            r"start\s+listen(?:ing)?|resume\s+listen(?:ing)?|wake\s*word\s+on|"
            r"enable\s+wake|listen\s+for\s+me",
            both,
        ):
            return {"tool": "_wake_word_on", "args": {}}
        if re.search(
            r"stop\s+listen(?:ing)?|pause\s+(?:wake|listen)|wake\s*word\s+off|"
            r"disable\s+wake|don'?t\s+listen",
            both,
        ):
            return {"tool": "_wake_word_off", "args": {}}

        # Audio Ducking control (#171)
        if re.search(
            r"enable\s+duck|duck(?:ing)?\s+on|turn\s+on\s+duck",
            both,
        ):
            return {"tool": "_audio_duck_on", "args": {}}
        if re.search(
            r"disable\s+duck|duck(?:ing)?\s+off|turn\s+off\s+duck|no\s+duck",
            both,
        ):
            return {"tool": "_audio_duck_off", "args": {}}

        # Ambient status (#166)
        if re.search(
            r"ambient\s+(?:noise|sound|status|level|info)|environment\s+noise|"
            r"how(?:'s|\s+is)\s+(?:the\s+)?(?:noise|environment|ambient)",
            both,
        ):
            return {"tool": "_ambient_status", "args": {}}

        # Proactive engagement status (#167)
        if re.search(
            r"proactive\s+(?:status|info|count|stats)|"
            r"how\s+many\s+proactive|"
            r"engagement\s+status|check.?in\s+(?:status|count)",
            both,
        ):
            return {"tool": "_proactive_status", "args": {}}

        # Health & break status (#168)
        if re.search(
            r"health\s+(?:status|info|stats|check)|"
            r"break\s+(?:status|timer|count)|"
            r"session\s+(?:time|timer|hours)",
            both,
        ):
            return {"tool": "_health_status", "args": {}}

        # Briefing
        if any(k in both for k in ("good morning", "morning briefing", "daily briefing",
                                    "what's today", "what do i have today")):
            return {"tool": "_briefing", "args": {}}

        # Maintenance (#129) — manual trigger
        if re.search(
            r"run\s+maintenance|system\s+cleanup|"
            r"clean\s+(?:up\s+)?(?:the\s+)?system|maintenance\s+run",
            both,
        ):
            return {"tool": "_maintenance", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — run reflection now
        if re.search(
            r"run\s+reflect|generate\s+reflect|"
            r"reflect\s+(?:on\s+)?today",
            both,
        ):
            return {"tool": "_run_reflection", "args": {"dry_run": "dry" in both}}

        # Reflection (#130) — show past reflections
        if re.search(
            r"show\s+reflect|list\s+reflect|past\s+reflect",
            both,
        ):
            return {"tool": "_list_reflections", "args": {}}
            
        # Clear memory
        if re.search(r"clear\s+memory", both):
            return {"tool": "_clear_memory", "args": {}}

        return None

    # ── RL & Intervention hooks (#125, #126) ─────────────────────────

    def _rl_reward_hook(self, tool_name: str, result: ToolResult) -> None:
        """Fire-and-forget: give RL engine a positive reward on tool success."""
        try:
            from bantz.agent.rl_engine import rl_engine, encode_state
            if not rl_engine.initialized:
                return
            tc = time_ctx.snapshot()
            state = encode_state(
                time_segment=tc.get("time_segment", "morning"),
                day=tc.get("day_name", "monday").lower(),
                location=tc.get("location", "home"),
                recent_tool=tool_name,
            )
            reward_val = 1.0 if result.success else -0.5
            rl_engine.reward(reward_val, next_state=state)
        except Exception:
            pass  # never crash the pipeline

    async def _check_intervention_queue(self) -> str | None:
        """[Deprecated — #137] Queue is now consumed by TUI toast system.

        Kept for backward compat in case headless mode needs it.
        """
        try:
            from bantz.agent.interventions import intervention_queue
            iv = intervention_queue.pop()
            if iv is None:
                return None
            return f"💡 [{iv.source}] {iv.title}\n   {iv.reason}"
        except Exception:
            return None

    def _prepend_intervention(self, response: str) -> str:
        """[Deprecated — #137] Toast system renders interventions separately.

        Kept for backward compat in case headless mode needs it.
        """
        iv = getattr(self, "_pending_intervention", None)
        if iv:
            self._pending_intervention = None
            return f"{iv}\n\n{response}"
        return response

    def _push_toast(
        self, title: str, reason: str = "", toast_type: str = "info",
    ) -> None:
        """Push a toast notification from brain context (#137).

        Delegates to the module-level ``_notify_toast()`` which routes
        to the TUI via callback or App.current.
        """
        _notify_toast(title, reason, toast_type)

    # ── Maintenance & Reflection handlers (#129, #130) ────────────────

    async def _handle_maintenance(self, dry_run: bool = False) -> str:
        """Run the maintenance workflow and return its summary."""
        try:
            from bantz.agent.workflows.maintenance import run_maintenance
            report = await run_maintenance(dry_run=dry_run)
            return report.summary()
        except Exception as exc:
            return f"❌ Maintenance failed: {exc}"

    def _handle_list_reflections(self, limit: int = 5) -> str:
        """List recent reflections from the KV store."""
        try:
            from bantz.agent.workflows.reflection import list_reflections
            items = list_reflections(limit=limit)
            if not items:
                return "No reflections stored yet. They are generated nightly."
            lines = ["🤔 Recent reflections:"]
            for item in items:
                date = item.get("date", "?")
                summary = item.get("summary", "")[:120]
                sessions = item.get("sessions", 0)
                lines.append(f"  • {date} ({sessions} sessions): {summary}")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Could not load reflections: {exc}"

    async def _handle_run_reflection(self, dry_run: bool = False) -> str:
        """Run the reflection workflow and return its summary."""
        try:
            from bantz.agent.workflows.reflection import run_reflection
            result = await run_reflection(dry_run=dry_run)
            return result.summary_line()
        except Exception as exc:
            return f"❌ Reflection failed: {exc}"

    async def _generate_command(self, orig: str, en: str) -> str:
        raw = await ollama.chat([
            {"role": "system", "content": COMMAND_SYSTEM},
            {"role": "user", "content": en or orig},
        ])
        return raw.strip().strip("`")

    async def _execute_plan(
        self, user_input: str, en_input: str, tc: dict,
    ) -> BrainResult | None:
        """Decompose a complex request into steps, then execute them.

        Returns BrainResult on success, or None if decomposition fails
        (so caller can fall through to normal routing).
        """
        from bantz.agent.planner import planner_agent
        from bantz.agent.executor import plan_executor

        tool_names = registry.names() + ["process_text"]
        steps = await planner_agent.decompose(en_input, tool_names)
        if not steps or len(steps) < 2:
            # Not genuinely multi-step — fall through to normal routing
            return None

        # Announce the itinerary to the user
        itinerary = planner_agent.format_itinerary(steps)
        log.info("Plan-and-Solve itinerary:\n%s", itinerary)

        # Execute all steps
        exec_result = await plan_executor.run(steps, llm_fn=ollama.chat)

        # Combine itinerary + execution summary
        resp = itinerary + "\n\n" + exec_result.summary()

        data_layer.conversations.add("assistant", resp, tool_used="planner")
        await self._graph_store(user_input, resp, "planner")
        self._fire_embeddings()

        return BrainResult(response=resp, tool_used="planner")

    async def _handle_location(self) -> str:
        """Handle 'where am i' queries — show GPS/location info."""
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
            lines.append(f"📍 Phone GPS: {gps_loc['lat']:.6f}, {gps_loc['lon']:.6f} (±{acc}m)")
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

    async def _handle_save_place(self, name: str) -> str:
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

    async def _handle_list_places(self) -> str:
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

    async def _handle_delete_place(self, name: str) -> str:
        """Delete a saved place."""
        from bantz.core.places import places as _places
        if _places.delete_place(name):
            return f"📌 '{name}' deleted."
        return f"❌ No saved place named '{name}' found."

    async def process(self, user_input: str, confirmed: bool = False,
                      is_remote: bool = False) -> BrainResult:
        self._is_remote = is_remote
        self._ensure_memory()
        await self._ensure_graph()
        en_input = await self._to_en(user_input)
        tc = time_ctx.snapshot()

        # ── Sentiment RLHF intercept (#180) ──────────────────────────
        # Uses RAW user_input (not en_input) to catch Turkish keywords
        # before the translation layer converts them to English.
        feedback = _detect_feedback(user_input)
        if feedback:
            try:
                from bantz.agent.rl_engine import rl_engine, Action, encode_state
                if rl_engine.initialized:
                    state = encode_state(
                        time_segment=tc.get("time_segment", "morning"),
                        day=tc.get("day_name", "monday").lower(),
                        location=tc.get("location", "home"),
                        recent_tool="feedback_chat",
                    )
                    reward_val = 2.0 if feedback == "positive" else -2.0
                    rl_engine.force_reward(state, Action.FEEDBACK_CHAT, reward_val)
                    log.info("RLHF sentiment: %s → reward %.1f", feedback, reward_val)
            except Exception:
                pass  # never crash the pipeline
            if feedback == "positive":
                self._feedback_ctx = (
                    "\n[The user just praised you. Show humble butler gratitude — "
                    "a brief, dignified acknowledgement. Do not be excessive.]"
                )
            else:
                self._feedback_ctx = (
                    "\n[The user just scolded you. Show a brief moment of butler "
                    "composure under pressure. Apologise sincerely, ask how to "
                    "correct yourself. Do NOT grovel.]"
                )

        # NOTE: Intervention queue is now consumed by the TUI's toast
        # system (#137) instead of being popped here.  Brain no longer
        # prepends intervention text to chat responses.

        # Save user message ONCE — before any branching
        data_layer.conversations.add("user", user_input)

        # ── Plan-and-Solve: LLM-based multi-step decomposition (#187) ────
        # Planner runs FIRST — it handles complex multi-tool requests via
        # LLM heuristics.  Must be above workflow_engine to prevent the
        # old regex-based engine from eagerly stealing autonomous commands.
        try:
            from bantz.agent.planner import planner_agent
            if planner_agent.is_complex(en_input):
                plan_result = await self._execute_plan(user_input, en_input, tc)
                if plan_result is not None:
                    return plan_result
        except Exception as exc:
            log.debug("Planner check failed: %s — falling through", exc)

        quick = self._quick_route(user_input, en_input)

        if quick and quick["tool"] == "_tts_stop":
            from bantz.agent.tts import tts_engine
            if tts_engine.is_speaking:
                tts_engine.stop()
                text = "🔇 Stopped."
            else:
                text = "I'm not speaking right now."
            data_layer.conversations.add("assistant", text, tool_used="tts")
            return BrainResult(response=text, tool_used="tts")

        if quick and quick["tool"] == "_wake_word_off":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    wake_listener.stop()
                    text = "🔇 Wake word listener stopped."
                else:
                    text = "Wake word listener is not running."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_wake_word_on":
            try:
                from bantz.agent.wake_word import wake_listener
                if wake_listener.running:
                    text = "Wake word listener is already running."
                else:
                    ok = wake_listener.start()
                    text = "🎤 Wake word listener started." if ok else "❌ Could not start wake word listener."
            except Exception:
                text = "Wake word listener is not available."
            data_layer.conversations.add("assistant", text, tool_used="wake_word")
            return BrainResult(response=text, tool_used="wake_word")

        if quick and quick["tool"] == "_audio_duck_on":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                if audio_ducker.available():
                    audio_ducker.enabled = True
                    text = "🔉 Audio ducking enabled."
                else:
                    text = "❌ Audio ducking not available (pactl not found)."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_audio_duck_off":
            try:
                from bantz.agent.audio_ducker import audio_ducker
                audio_ducker.enabled = False
                text = "🔇 Audio ducking disabled."
            except Exception:
                text = "Audio ducking module is not available."
            data_layer.conversations.add("assistant", text, tool_used="audio_ducker")
            return BrainResult(response=text, tool_used="audio_ducker")

        if quick and quick["tool"] == "_ambient_status":
            try:
                from bantz.agent.ambient import ambient_analyzer
                snap = ambient_analyzer.latest()
                if snap:
                    text = (
                        f"🎤 Ambient: **{snap.label.value.upper()}** "
                        f"(RMS={snap.rms:.0f}, ZCR={snap.zcr:.3f})\n"
                        f"{ambient_analyzer.day_summary()}"
                    )
                else:
                    text = "No ambient data yet — analyzer is waiting for samples."
            except Exception:
                text = "Ambient analyzer is not available."
            data_layer.conversations.add("assistant", text, tool_used="ambient")
            return BrainResult(response=text, tool_used="ambient")

        if quick and quick["tool"] == "_proactive_status":
            try:
                from bantz.agent.proactive import (
                    proactive_engine, _get_daily_count, _compute_adaptive_max,
                )
                from bantz.agent.rl_engine import rl_engine
                kv = data_layer.kv
                if kv:
                    count, date = _get_daily_count(kv)
                    avg_r = rl_engine.episodes.avg_reward(7) if rl_engine.initialized else 0.0
                    max_d = _compute_adaptive_max(config.proactive_max_daily, avg_r)
                    text = (
                        f"💬 Proactive Engagement Status\n"
                        f"  Enabled: {'✅' if config.proactive_enabled else '❌'}\n"
                        f"  Today: {count}/{max_d} messages\n"
                        f"  RL avg reward (7d): {avg_r:.2f}\n"
                        f"  Interval: {config.proactive_interval_hours}h ±{config.proactive_jitter_minutes}m"
                    )
                else:
                    text = "Proactive engine: KV store not available."
            except Exception:
                text = "Proactive engagement module is not available."
            data_layer.conversations.add("assistant", text, tool_used="proactive")
            return BrainResult(response=text, tool_used="proactive")

        if quick and quick["tool"] == "_health_status":
            try:
                from bantz.agent.health import health_engine
                s = health_engine.status()
                cooldown_lines = "\n".join(
                    f"    {rid}: {mins:.0f}m left" for rid, mins in s["cooldowns"].items() if mins > 0
                )
                text = (
                    f"🏥 Health & Break Status\n"
                    f"  Enabled: {'✅' if config.health_enabled else '❌'}\n"
                    f"  Active session: {s['active_hours']:.1f}h\n"
                    f"  Break taken: {'✅' if s['had_break'] else '❌'}\n"
                    f"  Since last break: {s['minutes_since_break']:.0f}m\n"
                    f"  Thermal streak: CPU={s['thermal_cpu_streak']} GPU={s['thermal_gpu_streak']}\n"
                    f"  Check interval: {config.health_check_interval}s"
                )
                if cooldown_lines:
                    text += f"\n  Active cooldowns:\n{cooldown_lines}"
            except Exception:
                text = "Health & break module is not available."
            data_layer.conversations.add("assistant", text, tool_used="health")
            return BrainResult(response=text, tool_used="health")

        if quick and quick["tool"] == "_briefing":
            from bantz.core.briefing import briefing as _briefing
            text = await _briefing.generate()
            data_layer.conversations.add("assistant", text, tool_used="briefing")
            # Speak via TTS if available (#131) — suppress for remote (#178)
            if not getattr(self, '_is_remote', False):
                try:
                    from bantz.agent.tts import tts_engine
                    if tts_engine.available():
                        await tts_engine.speak_background(text)
                except Exception:
                    pass
            return BrainResult(response=text, tool_used="briefing")

        if quick and quick["tool"] == "_maintenance":
            text = await self._handle_maintenance(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="maintenance")
            return BrainResult(response=text, tool_used="maintenance")

        if quick and quick["tool"] == "_list_reflections":
            text = self._handle_list_reflections()
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_run_reflection":
            text = await self._handle_run_reflection(quick["args"].get("dry_run", False))
            data_layer.conversations.add("assistant", text, tool_used="reflection")
            return BrainResult(response=text, tool_used="reflection")

        if quick and quick["tool"] == "_location":
            text = await self._handle_location()
            data_layer.conversations.add("assistant", text, tool_used="location")
            return BrainResult(response=text, tool_used="location")

        if quick and quick["tool"] == "_save_place":
            text = await self._handle_save_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_list_places":
            text = await self._handle_list_places()
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_delete_place":
            text = await self._handle_delete_place(quick["args"]["name"])
            data_layer.conversations.add("assistant", text, tool_used="places")
            return BrainResult(response=text, tool_used="places")

        if quick and quick["tool"] == "_schedule_today":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_today()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_next":
            from bantz.core.schedule import schedule as _sched
            text = _sched.format_next()
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_date":
            from bantz.core.schedule import schedule as _sched
            from datetime import datetime as _dt
            target = _dt.fromisoformat(quick["args"]["date_iso"])
            text = _sched.format_for_date(target)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_schedule_week":
            from bantz.core.schedule import schedule as _sched
            resolved = resolve_date(user_input)
            text = _sched.format_week(resolved)
            data_layer.conversations.add("assistant", text, tool_used="schedule")
            return BrainResult(response=text, tool_used="schedule")

        if quick and quick["tool"] == "_generate":
            cmd = await self._generate_command(user_input, en_input)
            plan = {"route": "tool", "tool_name": "shell",
                    "tool_args": {"command": cmd}, "risk_level": "moderate"}

        elif quick:
            plan = {"route": "tool", "tool_name": quick["tool"],
                    "tool_args": quick["args"], "risk_level": "safe"}

        else:
            plan = await cot_route(en_input, registry.all_schemas())
            if plan is None:
                # Stream chat responses for lower perceived latency (#67)
                stream = self._chat_stream(en_input, tc)
                return BrainResult(
                    response="", tool_used=None, stream=stream,
                )

        route     = plan.get("route", "chat")
        tool_name = plan.get("tool_name") or ""
        tool_args = plan.get("tool_args") or {}
        risk      = plan.get("risk_level", "safe")

        if route != "tool" or not tool_name:
            stream = self._chat_stream(en_input, tc)
            return BrainResult(response="", tool_used=None, stream=stream)

        if risk == "destructive" and config.shell_confirm_destructive and not confirmed:
            cmd_str = tool_args.get("command", tool_name)
            warn = (
                f"⚠️  Destructive operation: [{tool_name}] `{cmd_str}`\n"
                f"Confirm? (yes/no)"
            )
            data_layer.conversations.add("assistant", warn)
            return BrainResult(
                response=warn,
                tool_used=tool_name,
                needs_confirm=True,
                pending_command=cmd_str,
                pending_tool=tool_name,
                pending_args=tool_args,
            )

        tool = registry.get(tool_name)
        if not tool:
            err = f"Tool not found: {tool_name}"
            data_layer.conversations.add("assistant", err)
            return BrainResult(response=err, tool_used=None)

        result = await tool.execute(**tool_args)

        # ── RL reward: positive signal on successful tool use (#125) ──
        self._rl_reward_hook(tool_name, result)

        # ── Store tool results for contextual follow-ups (#56) ──
        if result.success and result.data:
            if result.data.get("messages"):
                self._last_messages = result.data["messages"]
            if result.data.get("events"):
                self._last_events = result.data["events"]

        # ── Compose/reply draft → confirmation flow ──
        if result.success and result.data and result.data.get("draft"):
            d = result.data
            self._last_draft = {
                "to": d["to"],
                "subject": d.get("subject", ""),
                "body": d["body"],
            }
            data_layer.conversations.add("assistant", result.output, tool_used=tool_name)
            return BrainResult(
                response=result.output,
                tool_used=tool_name,
                tool_result=result,
                needs_confirm=True,
                pending_tool="gmail",
                pending_args={
                    "action": "send",
                    "to": d["to"],
                    "subject": d.get("subject", ""),
                    "body": d["body"],
                },
            )

        # Try streaming finalize for long tool output (#67)
        fin_stream = await self._finalize_stream(en_input, result, tc)
        if fin_stream is not None:
            return BrainResult(
                response="", tool_used=tool_name,
                tool_result=result, stream=fin_stream,
            )

        # Short output — non-streaming finalize
        resp = await self._finalize(en_input, result, tc)
        data_layer.conversations.add("assistant", resp, tool_used=tool_name)
        await self._graph_store(user_input, resp, tool_name,
                                result.data if result else None)
        self._fire_embeddings()
        return BrainResult(response=resp, tool_used=tool_name, tool_result=result)

    async def _chat(self, en_input: str, tc: dict) -> str:
        """
        Chat mode with conversation history.
        history[-1] = the user message we just saved → exclude to avoid duplication.
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        graph_hint = await self._graph_context(en_input)
        vector_hint = await self._vector_context(en_input)
        desktop_hint = self._desktop_context()
        persona_state = _persona_hint()
        deep_memory = await self._deep_memory_context(en_input)

        # One-shot RLHF context injection (#180)
        feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint(),
                style_hint=_style_hint(), graph_hint=graph_hint,
                vector_hint=vector_hint, desktop_hint=desktop_hint,
                persona_state=persona_state, deep_memory=deep_memory,
                formality_hint=_formality_hint()) + feedback_hint},
            *prior,
            {"role": "user", "content": en_input},
        ]

        # Prefer Gemini for chat if available (#58)
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages)
                if not _is_refusal(raw):
                    return strip_markdown(raw)
        except Exception:
            pass  # fall through to Ollama

        try:
            raw = await ollama.chat(messages)
            if _is_refusal(raw):
                return "Sorry, I can't help with that. Try something else."
            return strip_markdown(raw)
        except Exception as exc:
            return f"(Ollama error: {exc})"

    async def _chat_stream(self, en_input: str, tc: dict) -> AsyncIterator[str]:
        """
        Streaming chat — yields tokens as they arrive from LLM.
        Post-processing (strip_markdown) runs on accumulated text at consumer side.
        """
        history = data_layer.conversations.context(n=12)
        prior = history[:-1] if (history and history[-1]["role"] == "user") else history
        graph_hint = await self._graph_context(en_input)
        vector_hint = await self._vector_context(en_input)
        desktop_hint = self._desktop_context()
        persona_state = _persona_hint()
        deep_memory = await self._deep_memory_context(en_input)

        # One-shot RLHF context injection (#180)
        feedback_hint = getattr(self, "_feedback_ctx", "")
        self._feedback_ctx = ""  # clear after consumption

        messages = [
            {"role": "system", "content": CHAT_SYSTEM.format(
                time_hint=tc["prompt_hint"], profile_hint=profile.prompt_hint(),
                style_hint=_style_hint(), graph_hint=graph_hint,
                vector_hint=vector_hint, desktop_hint=desktop_hint,
                persona_state=persona_state, deep_memory=deep_memory,
                formality_hint=_formality_hint()) + feedback_hint},
            *prior,
            {"role": "user", "content": en_input},
        ]

        # Try Gemini streaming first
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                async for token in gemini.chat_stream(messages):
                    yield token
                return
        except Exception:
            pass  # fall through to Ollama

        # Ollama streaming fallback
        try:
            async for token in ollama.chat_stream(messages):
                yield token
        except Exception as exc:
            yield f"(Ollama error: {exc})"

    async def _finalize(self, en_input: str, result: ToolResult, tc: dict) -> str:
        """Delegate to core.finalizer module."""
        return await _finalize_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await self._graph_context(en_input),
            deep_memory=await self._deep_memory_context(en_input),
            formality_hint=_formality_hint(),
        )

    async def _finalize_stream(
        self, en_input: str, result: ToolResult, tc: dict,
    ) -> AsyncIterator[str] | None:
        """Delegate to core.finalizer module."""
        return await _finalize_stream_fn(
            en_input, result, tc,
            style_hint=_style_hint(),
            profile_hint=profile.prompt_hint(),
            graph_hint=await self._graph_context(en_input),
            deep_memory=await self._deep_memory_context(en_input),
            formality_hint=_formality_hint(),
        )

    @staticmethod
    def _hallucination_check(response: str, tool_output: str) -> tuple[str, float]:
        """Delegate to core.finalizer module."""
        return _hallucination_check_fn(response, tool_output)


brain = Brain()