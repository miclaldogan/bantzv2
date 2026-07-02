"""Fixture tool backends with injectable failures (issue #506).

Real gmail/calendar hit live Google accounts and are non-deterministic;
unattended experiments need deterministic, side-effect-free tools with
controllable failures. Tools resolve through the global registry
(``bantz.tools.registry``), so fixtures are INJECTED at harness runtime —
no brain.py edits.

Components:

- :class:`CallLog` — machine-readable record of every tool call
  (name, args, result, exception) feeding ``success_check`` evaluation.
- :class:`FailureSpec` — the four injectable mechanisms from the contract
  (#500): ``fail_transient`` (fail N times then succeed), ``fail_always``,
  ``require_exact_args`` (arg error unless args match), ``raise_exception``
  (exercises the Brain's tool-exception interception path).
- Seven fixture tools mirroring the REAL tools' names and argument
  conventions (so Chain-of-Thought routing needs no special-casing):
  gmail, calendar, filesystem, shell (canned), weather, reminder,
  web_search (canned).
- :func:`install_fixtures` — swaps the registry to fixtures-only AFTER
  sandbox bootstrap and returns the :class:`FixtureWorld` handle.

Deterministic on purpose: same state + same args -> same ToolResult, no
network, no real filesystem, no clock reads.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Excerpt cap from the frozen contract (SCHEMA.md §2.1).
EXCERPT_MAX = 500

#: The contract's failure classes (task spec) → fixture mechanisms (here).
CLASS_TO_MECHANISM = {
    "none": "none",
    "transient_error": "fail_transient",
    "bad_args": "require_exact_args",
    "unrecoverable": "fail_always",
    # wrong_tool_first is a routing-level condition: the corpus crafts a
    # misleading prompt; nothing is injected at the fixture layer.
    "wrong_tool_first": "none",
}
MECHANISMS = {
    "none", "fail_transient", "fail_always", "require_exact_args",
    "raise_exception",
}


# ── call log ─────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    tool: str
    args: dict[str, Any]
    success: bool
    error: str | None
    output_excerpt: str
    exception: str | None = None  # non-null when the tool RAISED

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "result": {
                "success": self.success,
                "error": self.error,
                "output_excerpt": self.output_excerpt,
            },
            "exception": self.exception,
        }


class CallLog:
    """Records every fixture call — the evidence for ``success_check``."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record(self, tool: str, args: dict[str, Any], *, success: bool,
               error: str | None, output: str,
               exception: str | None = None) -> None:
        self.records.append(CallRecord(
            tool=tool,
            args=copy.deepcopy(args),
            success=success,
            error=error,
            output_excerpt=(output or "")[:EXCERPT_MAX],
            exception=exception,
        ))

    def calls(self, tool: str | None = None) -> list[CallRecord]:
        if tool is None:
            return list(self.records)
        return [r for r in self.records if r.tool == tool]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.records]


# ── failure injection ────────────────────────────────────────────────────────

@dataclass
class FailureSpec:
    """One injectable failure, targeted at a single tool.

    ``mechanism`` ∈ MECHANISMS. Built either directly or from a task spec's
    ``failure_injection`` object via :meth:`from_contract`.
    """
    mechanism: str = "none"
    tool: str = ""                      # which fixture this applies to
    fail_times: int = 1                 # fail_transient: fail N then succeed
    error: str = "injected failure"
    exact_args: dict[str, Any] = field(default_factory=dict)
    _failures_done: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown failure mechanism: {self.mechanism!r}")

    @classmethod
    def from_contract(cls, failure_injection: dict[str, Any] | None,
                      expected_tool: str = "") -> "FailureSpec":
        """Build from the task spec's ``failure_injection`` field (#500).

        ``params.mechanism`` may override the default class→mechanism map
        (e.g. an ``unrecoverable`` variant that RAISES instead of erroring).
        ``params.tool`` may retarget the injection (default: expected_tool).
        """
        if not failure_injection:
            return cls(mechanism="none")
        klass = failure_injection.get("class", "none")
        params = failure_injection.get("params", {}) or {}
        mechanism = params.get("mechanism", CLASS_TO_MECHANISM.get(klass))
        if mechanism is None:
            raise ValueError(f"unknown failure class: {klass!r}")
        return cls(
            mechanism=mechanism,
            tool=params.get("tool", expected_tool),
            fail_times=int(params.get("fail_times", 1)),
            error=params.get("error", f"injected {klass}"),
            exact_args=params.get("require_exact_args",
                                  params.get("exact_args", {})),
        )

    # ── gate ──────────────────────────────────────────────────────────────

    def intercept(self, args: dict[str, Any]):
        """Return a failing ToolResult, raise, or return None (= proceed)."""
        from bantz.tools import ToolResult

        if self.mechanism == "none":
            return None
        if self.mechanism == "fail_always":
            return ToolResult(success=False, output="", error=self.error)
        if self.mechanism == "fail_transient":
            if self._failures_done < self.fail_times:
                self._failures_done += 1
                return ToolResult(success=False, output="", error=self.error)
            return None
        if self.mechanism == "require_exact_args":
            for key, want in self.exact_args.items():
                if args.get(key) != want:
                    return ToolResult(
                        success=False, output="",
                        error=(f"invalid argument '{key}': "
                               f"got {args.get(key)!r} — {self.error}"),
                    )
            return None
        if self.mechanism == "raise_exception":
            raise RuntimeError(self.error)
        return None  # pragma: no cover — MECHANISMS is closed above


# ── fixture base ─────────────────────────────────────────────────────────────

class FixtureTool:
    """Duck-types ``bantz.tools.BaseTool``: name/description/risk_level,
    ``async execute(**kwargs) -> ToolResult``, ``schema()``."""

    name = "fixture"
    description = "fixture tool"
    risk_level = "safe"

    def __init__(self, state: dict[str, Any] | None, call_log: CallLog,
                 failure: FailureSpec | None = None) -> None:
        self.state: dict[str, Any] = copy.deepcopy(state or {})
        self._log = call_log
        self._failure = failure

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "risk_level": self.risk_level}

    async def execute(self, **kwargs: Any):
        failure = self._failure
        if failure is not None and failure.tool == self.name:
            try:
                intercepted = failure.intercept(kwargs)
            except Exception as exc:
                # Record BEFORE propagating — the call log must show the
                # raise even though the brain intercepts it upstream.
                self._log.record(self.name, kwargs, success=False,
                                 error=str(exc), output="",
                                 exception=str(exc))
                raise
            if intercepted is not None:
                self._log.record(self.name, kwargs,
                                 success=intercepted.success,
                                 error=intercepted.error or None,
                                 output=intercepted.output)
                return intercepted

        result = self._run(**kwargs)
        self._log.record(self.name, kwargs, success=result.success,
                         error=result.error or None, output=result.output)
        return result

    def _run(self, **kwargs: Any):
        raise NotImplementedError

    # helper
    def _ok(self, output: str, **data: Any):
        from bantz.tools import ToolResult
        return ToolResult(success=True, output=output, data=data)

    def _err(self, error: str):
        from bantz.tools import ToolResult
        return ToolResult(success=False, output="", error=error)


# ── fixtures ─────────────────────────────────────────────────────────────────

class GmailFixture(FixtureTool):
    """State: {"inbox": [{id, from, subject, body, unread}], "sent": [...]}"""

    name = "gmail"
    description = "Read, search and send email (fixture)."

    def _run(self, action: str = "summary", message_id: str = "",
             from_sender: str = "", subject_filter: str = "",
             to: str = "", subject: str = "", body: str = "",
             **kwargs: Any):
        inbox = self.state.setdefault("inbox", [])
        sent = self.state.setdefault("sent", [])

        if action in ("summary", "count", "filter"):
            unread = [m for m in inbox if m.get("unread", True)]
            lines = [f"- {m['from']}: {m['subject']}" for m in unread]
            return self._ok(
                f"You have {len(unread)} unread email(s).\n" + "\n".join(lines),
                count=len(unread), messages=unread,
            )
        if action == "search":
            needle = (subject_filter or from_sender or kwargs.get("query", "")).lower()
            hits = [m for m in inbox
                    if needle in m.get("subject", "").lower()
                    or needle in m.get("from", "").lower()
                    or needle in m.get("body", "").lower()]
            lines = [f"- {m['from']}: {m['subject']}" for m in hits]
            return self._ok(
                f"Found {len(hits)} message(s) matching '{needle}'.\n"
                + "\n".join(lines),
                count=len(hits), messages=hits,
            )
        if action == "read":
            for m in inbox:
                if m.get("id") == message_id:
                    m["unread"] = False
                    return self._ok(
                        f"From: {m['from']}\nSubject: {m['subject']}\n\n"
                        f"{m.get('body', '')}",
                        message=m,
                    )
            return self._err(f"message not found: {message_id!r}")
        if action in ("send", "compose", "reply"):
            if not to or not (subject or body):
                return self._err("compose requires 'to' and 'subject'/'body'")
            msg = {"to": to, "subject": subject, "body": body}
            sent.append(msg)
            return self._ok(f"Email sent to {to}: {subject}", sent=msg)
        return self._err(f"unsupported gmail action: {action!r}")


class CalendarFixture(FixtureTool):
    """State: {"events": [{id, title, date, time, duration}]}"""

    name = "calendar"
    description = "Query and create calendar events (fixture)."

    def _run(self, action: str = "today", title: str = "", date: str = "",
             time: str = "", duration: int = 60, event_id: str = "",
             **kwargs: Any):
        events = self.state.setdefault("events", [])

        if action in ("today", "week", "date", "upcoming"):
            subset = ([e for e in events if e.get("date") == date]
                      if action == "date" and date else events)
            lines = [f"- {e.get('time', '?')} {e['title']} ({e.get('date', '')})"
                     for e in subset]
            return self._ok(
                f"{len(subset)} event(s) on the calendar.\n" + "\n".join(lines),
                events=subset,
            )
        if action == "create":
            if not title or not date:
                return self._err("create requires 'title' and 'date'")
            event = {"id": f"evt_{len(events) + 1}", "title": title,
                     "date": date, "time": time, "duration": duration}
            events.append(event)
            return self._ok(
                f"Created event '{title}' on {date}"
                + (f" at {time}" if time else ""),
                event=event,
            )
        if action == "delete":
            for e in list(events):
                if e.get("id") == event_id or e.get("title") == title:
                    events.remove(e)
                    return self._ok(f"Deleted event '{e['title']}'.", event=e)
            return self._err(f"event not found: {event_id or title!r}")
        if action == "conflicts":
            seen: dict[tuple, list] = {}
            for e in events:
                seen.setdefault((e.get("date"), e.get("time")), []).append(e)
            clashes = [v for v in seen.values() if len(v) > 1]
            return self._ok(f"{len(clashes)} conflict(s) found.",
                            conflicts=clashes)
        return self._err(f"unsupported calendar action: {action!r}")


class FilesystemFixture(FixtureTool):
    """Virtual FS. State: {"files": {path: content}, "dirs": [path, ...]}"""

    name = "filesystem"
    description = "List, read and write files (fixture, virtual)."

    def _run(self, action: str = "ls", path: str = "~", content: str = "",
             folder_path: str = "", file_name: str = "", **kwargs: Any):
        files = self.state.setdefault("files", {})
        dirs = self.state.setdefault("dirs", [])

        if action == "ls":
            names = sorted(
                [p for p in files if p.startswith(path.rstrip("/"))]
                + [d for d in dirs if d.startswith(path.rstrip("/"))]
            ) if path not in ("~", "", "/") else sorted(files) + sorted(dirs)
            return self._ok(
                f"{len(names)} entr(y/ies) in {path}:\n"
                + "\n".join(f"- {n}" for n in names),
                entries=names,
            )
        if action == "read":
            if path in files:
                return self._ok(files[path], path=path)
            return self._err(f"no such file: {path}")
        if action == "write":
            if not path or path in ("~", "/"):
                return self._err("write requires a file path")
            files[path] = content
            return self._ok(f"Wrote {len(content)} chars to {path}",
                            path=path)
        if action == "create_folder_and_file":
            if not folder_path or not file_name:
                return self._err(
                    "create_folder_and_file requires 'folder_path' and "
                    "'file_name'")
            if folder_path not in dirs:
                dirs.append(folder_path)
            full = f"{folder_path.rstrip('/')}/{file_name}"
            files[full] = content
            return self._ok(f"Created {full}", path=full)
        return self._err(f"unsupported filesystem action: {action!r}")


class ShellFixture(FixtureTool):
    """Canned shell. State: {"commands": {cmd: {stdout, stderr, returncode}},
    "default": {stdout, stderr, returncode}}  — deterministic, nothing runs."""

    name = "shell"
    description = "Run a shell command (fixture, canned output)."

    def _run(self, command: str = "", **kwargs: Any):
        if not command.strip():
            return self._err("empty command")
        canned = self.state.setdefault("commands", {})
        spec = canned.get(command.strip(),
                          self.state.get("default",
                                         {"stdout": "", "returncode": 0}))
        rc = int(spec.get("returncode", 0))
        stdout = spec.get("stdout", "")
        stderr = spec.get("stderr", "")
        self.state.setdefault("executed", []).append(command.strip())
        if rc == 0:
            return self._ok(stdout or "(no output)", returncode=rc,
                            stdout=stdout, stderr=stderr)
        return self._err(stderr or f"command failed with exit code {rc}")


class WeatherFixture(FixtureTool):
    """State: {"reports": {city_lower: text}, "default": text}"""

    name = "weather"
    description = "Current weather for a city (fixture, canned)."

    def _run(self, city: str = "", **kwargs: Any):
        reports = self.state.get("reports", {})
        key = city.strip().lower()
        if key and key in reports:
            return self._ok(reports[key], city=city)
        default = self.state.get("default", "")
        if default:
            return self._ok(default, city=city or "default")
        return self._err(f"no weather data for {city!r}")


class ReminderFixture(FixtureTool):
    """State: {"reminders": [{id, text, time}]}"""

    name = "reminder"
    description = "Add, list and cancel reminders (fixture)."

    def _run(self, action: str = "add", **kwargs: Any):
        reminders = self.state.setdefault("reminders", [])
        if action == "list":
            lines = [f"- [{r['id']}] {r['text']} at {r.get('time', '?')}"
                     for r in reminders]
            return self._ok(
                f"{len(reminders)} reminder(s).\n" + "\n".join(lines),
                reminders=reminders,
            )
        if action == "cancel":
            rid = kwargs.get("reminder_id", kwargs.get("id", ""))
            for r in list(reminders):
                if str(r.get("id")) == str(rid):
                    reminders.remove(r)
                    return self._ok(f"Cancelled reminder {rid}.", reminder=r)
            return self._err(f"reminder not found: {rid!r}")
        # add (default)
        text = kwargs.get("text", kwargs.get("intent", kwargs.get("title", "")))
        when = kwargs.get("time", kwargs.get("when", ""))
        if not text:
            return self._err("reminder requires 'text'")
        item = {"id": f"rem_{len(reminders) + 1}", "text": text, "time": when}
        reminders.append(item)
        return self._ok(
            f"Reminder set: {text}" + (f" at {when}" if when else ""),
            reminder=item,
        )


class WebSearchFixture(FixtureTool):
    """Canned search. State:
    {"results": [{"query_contains": str, "items": [{title,url,snippet}]}],
     "default": [items]}"""

    name = "web_search"
    description = "Web search (fixture, canned results)."

    def _run(self, query: str = "", **kwargs: Any):
        if not query.strip():
            return self._err("empty query")
        q = query.lower()
        items = None
        for entry in self.state.get("results", []):
            if entry.get("query_contains", "").lower() in q:
                items = entry.get("items", [])
                break
        if items is None:
            items = self.state.get("default", [])
        lines = [f"- {it['title']} — {it.get('snippet', '')} ({it.get('url', '')})"
                 for it in items]
        return self._ok(
            f"{len(items)} result(s) for '{query}'.\n" + "\n".join(lines),
            results=items, query=query,
        )


FIXTURE_CLASSES: dict[str, type[FixtureTool]] = {
    cls.name: cls
    for cls in (GmailFixture, CalendarFixture, FilesystemFixture,
                ShellFixture, WeatherFixture, ReminderFixture,
                WebSearchFixture)
}


# ── world / registry swap ────────────────────────────────────────────────────

class FixtureWorld:
    """The per-task fixture universe: tools + shared call log + state."""

    def __init__(self, fixture_setup: dict[str, Any] | None = None,
                 failure: FailureSpec | None = None) -> None:
        setup = fixture_setup or {}
        unknown = set(setup) - set(FIXTURE_CLASSES)
        if unknown:
            raise ValueError(f"fixture_setup names unknown fixtures: {unknown}")
        self.call_log = CallLog()
        self.failure = failure
        self.tools: dict[str, FixtureTool] = {
            name: cls(setup.get(name), self.call_log, failure)
            for name, cls in FIXTURE_CLASSES.items()
        }
        self._saved_registry: dict[str, Any] | None = None

    # registry swap ─────────────────────────────────────────────────────────

    def install(self) -> "FixtureWorld":
        """Swap the global registry to fixtures ONLY (complete replacement:
        a fixture task cannot reach a real gmail/calendar/shell)."""
        from bantz.tools import registry
        self._saved_registry = dict(registry._tools)
        registry._tools.clear()
        for tool in self.tools.values():
            registry.register(tool)  # type: ignore[arg-type]
        self.verify_swap()
        return self

    def restore(self) -> None:
        from bantz.tools import registry
        if self._saved_registry is not None:
            registry._tools.clear()
            registry._tools.update(self._saved_registry)
            self._saved_registry = None

    def verify_swap(self) -> None:
        """Assert the registry serves exactly this world's fixtures."""
        from bantz.tools import registry
        for name, tool in self.tools.items():
            if registry.get(name) is not tool:
                raise AssertionError(
                    f"registry swap incomplete: {name!r} does not resolve "
                    "to the fixture instance")
        foreign = [n for n, t in registry._tools.items()
                   if t not in self.tools.values()]
        if foreign:
            raise AssertionError(
                f"registry still serves non-fixture tools: {foreign}")

    # evidence ──────────────────────────────────────────────────────────────

    def state(self, fixture: str) -> dict[str, Any]:
        return self.tools[fixture].state

    def transcript(self) -> dict[str, Any]:
        """Machine-readable evidence for success_check + taxonomy labeling."""
        return {
            "calls": self.call_log.as_dicts(),
            "final_state": {name: t.state for name, t in self.tools.items()},
        }


def install_fixtures(fixture_setup: dict[str, Any] | None = None,
                     failure_injection: dict[str, Any] | None = None,
                     expected_tool: str = "",
                     *, require_sandbox: bool = True) -> FixtureWorld:
    """Build a FixtureWorld from a task spec and swap it into the registry.

    Call AFTER ``sandbox.bootstrap()`` (enforced unless ``require_sandbox``
    is False — only unit tests may disable it).
    """
    if require_sandbox:
        import sandbox
        sandbox.sandbox_root()  # raises SandboxViolation if not bootstrapped

    failure = FailureSpec.from_contract(failure_injection, expected_tool)
    return FixtureWorld(fixture_setup, failure).install()
