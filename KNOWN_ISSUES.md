# Known Issues

Last updated: 2026-05-28  
Test environment: Python 3.12.7, pytest 9.0.3

All failures below are **pre-existing** — they existed before the v2.1.0 feature work
(PRs #443–#453) and are not regressions introduced by those changes.

---

## Summary

| Category | Files | Failures |
|---|---|---|
| TUI implementation gap | 6 test files | 226 |
| Test ordering / state pollution | 6 test files | 63 |
| Missing `apscheduler` dependency | `test_job_scheduler.py` | 21 |
| Python 3.12 asyncio deprecation | `test_bridge_smoke.py` | 6 |
| Piper CLI flag rename | `test_tts.py` | 1 |
| Missing `apscheduler` (via TTS) | `test_tts.py` | 1 |
| Collection error | `test_feed_tool.py` | 1 error |
| **Total** | | **318 failures, 1 error** |

---

## 1 — TUI implementation gap (226 failures)

**Affected files:**
- `tests/tui/test_toast.py` — 69 failures
- `tests/tui/test_telemetry.py` — 55 failures
- `tests/tui/test_mood.py` — 45 failures + 8 collection errors
- `tests/tui/test_event_bridge.py` — 31 failures
- `tests/tui/test_streaming.py` — 25 failures
- `tests/tui/test_input_control.py` — 1 failure

**Root causes:**
- `FileNotFoundError`: `src/bantz/interface/tui/styles.tcss` does not exist.
- `AttributeError`: `BantzApp` is missing methods/attributes that the test suite
  expects: `_bus_handler`, `_on_bus_wake_word`, `_relay_bus_event`,
  `_on_bus_ambient_change`, `_on_bus_health_alert`, `action_focus_input`,
  `_handle_intervention_response`, `_process_interventions`,
  `_start_wake_word_listener`, `on_bantz_event_message`.
- `ImportError`: `from bantz.interface.tui.app import run` — `run` not exported.
- `AssertionError`: Tests assert app title `'BANTZ v3'`; actual is `'BantzApp'`.
- Tests assert event bus wiring, keyboard bindings, and toast/mood widgets that are
  not yet implemented in `src/bantz/interface/tui/app.py`.

**Status:** These tests are aspirational specs for TUI features not yet implemented.
They predate v2.1.0 and will be addressed in a future TUI overhaul milestone.

---

## 2 — Test ordering / state pollution (63 failures in full-suite run)

**Affected files (all pass when run in isolation):**
- `tests/core/test_brain_integrations.py` — 21 failures in full suite, 0 in isolation
- `tests/core/test_routing_engine.py` — 15 failures in full suite, 0 in isolation
- `tests/tools/test_gmail_autochain.py` — 11 failures in full suite, 0 in isolation
- `tests/tools/test_browser_tool.py` — 8 failures in full suite, 0 in isolation
- `tests/agent/test_proactive.py` — 7 failures in full suite, 0 in isolation
- `tests/interface/test_telegram_screenshot.py` — 1 failure in full suite, 0 in isolation

**Root cause:** Some earlier test file in the full suite mutates global/module-level
state (mock objects, singletons, or `sys.modules` entries) without cleaning up,
causing downstream tests to see stale mocks or missing attributes.

**Workaround:** Run these files individually with `pytest <file>` — all pass.  
**Fix:** Add `autouse` fixtures to reset shared singletons between test sessions.

---

## 3 — Missing `apscheduler` dependency (21 failures)

**Affected file:** `tests/agent/test_job_scheduler.py`

**Root cause:** `src/bantz/agent/job_scheduler.py` imports
`apscheduler.schedulers.asyncio.AsyncIOScheduler` and
`apscheduler.triggers.interval.IntervalTrigger` at runtime. The `apscheduler`
package is listed in `pyproject.toml` but is not installed in this environment.

**Fix:**
```bash
pip install "apscheduler>=3.10"
```

---

## 4 — Python 3.12 asyncio deprecation (6 failures)

**Affected file:** `tests/memory/test_bridge_smoke.py`

**Root cause:** Tests call `asyncio.get_event_loop().run_until_complete(coro)`.  
In Python 3.12, `asyncio.get_event_loop()` raises `RuntimeError: There is no current
event loop in thread 'MainThread'` when called outside an async context and no loop
has been created.

**Fix:** Migrate the 6 affected test methods to use `pytest-asyncio` (`async def` +
`asyncio_mode = auto`) or `asyncio.run()`.

---

## 5 — Piper CLI flag rename (1 failure)

**Affected file:** `tests/agent/test_tts.py::TestTTSAsync::test_synthesize_with_rate`

**Root cause:** The test asserts `'--length-scale'` appears in the Piper subprocess
argv, but the installed Piper binary uses `--length_scale` (underscore, not hyphen).

**Fix:** Update the test assertion to match the actual Piper flag, or normalise the
flag in `src/bantz/agent/tts.py` when building the command.

---

## 6 — Missing `apscheduler` (via TTS briefing watcher, 1 failure)

**Affected file:** `tests/agent/test_tts.py::TestBriefingWatcherRegistration::test_register_when_tts_enabled`

Same root cause as item 3 — `_register_briefing_watcher()` imports
`apscheduler.triggers.interval.IntervalTrigger`. Resolved by installing `apscheduler`.

---

## 7 — test_feed_tool.py collection error (1 error)

**Affected file:** `tests/tools/test_feed_tool.py`

**Root cause:** The file fails to collect (import-time error). Not counted in the
318-failure total above.

**Workaround:** `pytest --ignore=tests/tools/test_feed_tool.py`
