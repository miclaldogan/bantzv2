# Bantz — Copilot Context

## Project Identity

Bantz is a local-first, offline-capable AI assistant written in Python that runs entirely on the user's machine using Ollama as its LLM backend (default: `llama3.1:8b`). It presents itself as a 1920s English butler — always polite, subtly sarcastic, and firmly convinced he is a real person — handling Turkish-to-English translation, multi-step tool planning (web search, Gmail, Calendar, shell, filesystem, AT-SPI desktop control), voice I/O via faster-whisper + Piper TTS, and persistent memory via MemPalace (ChromaDB + SQLite knowledge graph). The primary user speaks Turkish on a Linux desktop; the entire pipeline — except optional Google/Telegram integrations — runs on-device with no cloud dependency.

---

## Stack

- **Language**: Python 3.11+, async-first (`asyncio` everywhere)
- **LLM backend**: Ollama HTTP API at `http://localhost:11434` — `llm/ollama.py`
- **Translation**: Helsinki-NLP MarianMT (`opus-mt-tr-en`, `opus-mt-tc-big-en-tr`) via `transformers` — `i18n/bridge.py`
- **TTS**: Piper binary (`~/.local/bin/piper` or miniforge3) + `tr_TR-dfki-medium.onnx` — audio via `pw-play` → `paplay` → `aplay`
- **STT**: `faster-whisper` (NOT installed) — `agent/stt.py`
- **Voice capture**: `pyaudio` + `webrtcvad` (NOT installed) — `agent/voice_capture.py`
- **Wake word**: Porcupine / Picovoice (key unset) — `agent/wake_word.py`
- **Memory**: `mempalace>=3.0.0` — ChromaDB L3 vector + SQLite knowledge graph — `memory/bridge.py`
- **Scheduler**: APScheduler (`AsyncIOScheduler`) — `agent/job_scheduler.py`
- **TUI**: Textual — `interface/live_ui.py`
- **Telegram**: `python-telegram-bot` — `interface/telegram_bot.py`
- **Config**: Pydantic-settings `Config` class reading `.env` — `config.py`
- **Entry points**:
  - `bantz` → TUI (`live_ui.run()`)
  - `bantz --daemon` → headless daemon (WS server + APScheduler)
  - `bantz --once "..."` → single-shot query
  - `bantz --doctor` → system health check
  - `bantz --setup <target>` → guided setup wizard

---

## Architecture Map

```
src/bantz/
├── __main__.py              Entry point — argparse dispatch to TUI / daemon / --once / --doctor / --setup
├── config.py                Pydantic-settings Config — all env vars; _voice_master_switch cascade validator
│
├── core/
│   ├── brain.py             Orchestrator — translation → routing → tool execution → finalisation
│   ├── intent.py            cot_route() — CoT streaming router via Ollama; returns (plan_dict, thinking_str)
│   ├── routing_engine.py    quick_route() (regex hardware-only) + execute_plan() (multi-step dispatch)
│   ├── translation_layer.py to_en() — async TR→EN wrapper over i18n/bridge.py
│   ├── finalizer.py         finalize() — strip markdown, hallucination check, EN→TR back-translation
│   ├── context.py           BantzContext — typed dataclass carrier for one full request→response cycle
│   ├── memory_injector.py   inject(ctx) — parallel gather of graph/vector/deep/persona context into ctx
│   ├── prompt_builder.py    build_chat_system(ctx, tc) — renders CHAT_SYSTEM template
│   ├── memory.py            Memory (ConversationStore) — SQLite conversation log, FTS5 search
│   ├── habits.py            HabitEngine — mines tool usage patterns from SQLite by time segment
│   └── rl_hooks.py          rl_reward_hook() / rl_feedback_reward() — affinity rewards after tool runs
│
├── agent/
│   ├── tts.py               TTSEngine — Piper subprocess + pw-play/paplay/aplay, sentence streaming ✅
│   ├── stt.py               STTEngine — faster-whisper lazy load, PCM→transcript 🔴 (not installed)
│   ├── voice_capture.py     VoiceCapture — PyAudio + WebRTC VAD, records until silence 🔴 (not installed)
│   ├── wake_word.py         WakeWordListener — Porcupine wake-word, feeds AmbientAnalyzer 🔴 (no key)
│   ├── ghost_loop.py        Continuous listen→STT→brain loop; requires all voice deps 🔴
│   ├── ambient.py           AmbientAnalyzer — RMS+ZCR sound classifier, fed by wake word stream ⚠️
│   ├── job_scheduler.py     JobScheduler — APScheduler cron night jobs + persistent reminders ✅
│   ├── planner.py           PlannerAgent — LLM plan decomposition into JSON steps (PlanStep[])
│   ├── executor.py          PlanExecutor — sequential step execution with $REF_STEP_N binding
│   ├── affinity_engine.py   AffinityEngine — cumulative reward score, drives BONDED persona state ⚠️
│   ├── proactive.py         ProactiveEngine — time+context-triggered unprompted messages
│   └── workflows/
│       ├── maintenance.py   03:00 maintenance job
│       ├── reflection.py    23:00 reflection job
│       └── overnight_poll.py Overnight polling job
│
├── tools/
│   ├── __init__.py          BaseTool, ToolResult, ToolRegistry — every tool self-registers here
│   ├── system.py            SystemTool — psutil CPU/RAM/disk/uptime ✅
│   ├── shell.py             ShellTool — bash execution with destructive confirmation gate
│   ├── filesystem.py        FilesystemTool — read/write/list with home-dir sandbox
│   ├── desktop.py           DesktopTool — AT-SPI2 app/window enumeration ⚠️
│   ├── accessibility.py     AT-SPI2 accessibility tree walker
│   ├── screenshot.py        ScreenshotTool — delegates to vision/screenshot.py ✅
│   ├── browser.py           BrowserTool — headless browser via playwright
│   ├── gmail.py             GmailTool — Gmail API read/send ⚠️ (needs OAuth token)
│   ├── calendar.py          CalendarTool — Google Calendar read/write ⚠️ (needs OAuth token)
│   ├── search.py            SearchTool — web search backend
│   ├── visual_click.py      VisualClickTool — screenshot + VLM + AT-SPI click
│   └── [18 more tools]      weather, maps, translator, code_runner, reminders, etc.
│
├── memory/
│   ├── bridge.py            MemPalaceBridge — adapter over MemPalace (ChromaDB L3 + SQLite KG)
│   └── omni_memory.py       OmniMemoryManager — parallel KG+vector+deep recall with token budget
│
├── i18n/
│   └── bridge.py            LanguageBridge — MarianMT TR↔EN, lazy-loaded, thread-pool offloaded
│
├── interface/
│   ├── live_ui.py           Textual TUI — chat pane, status bar, service dots
│   ├── telegram_bot.py      run_bot() — python-telegram-bot dual-path (commands + LLM) ⚠️
│   └── ws_server.py         WebSocket broadcast server for bantz-ui React frontend
│
├── personality/
│   └── persona.py           PersonaStateBuilder — 6 states from CPU/time/app/affinity signals ✅
│
├── vision/
│   ├── screenshot.py        capture() — Wayland/X11 screenshot, JPEG compression ✅
│   └── remote_vlm.py        describe_screen() — POSTs base64 JPEG to VLM endpoint ❌ (disabled)
│
├── llm/
│   ├── ollama.py            OllamaClient — async HTTP, streaming, chat, embed
│   └── gemini.py            GeminiClient — optional Gemini fallback (disabled by default)
│
├── data/
│   ├── layer.py             DataLayer — composes Memory, Scheduler, KV store behind one init point
│   ├── sqlite_store.py      SQLiteKVStore — generic key-value and set storage on bantz.db
│   └── connection_pool.py   Thread-safe SQLite connection pool
│
├── auth/
│   └── [token files]        Google OAuth token store and refresh
│
├── cli/
│   └── setup.py             _doctor(), _handle_setup() — all --setup X and --doctor logic
│
└── workflows/
    └── [YAML runner]        YAML workflow runner + registry for user-defined multi-step pipelines
```

---

## Data Flow

Every user message travels this pipeline:

```
1. INPUT
   User types in TUI / Telegram / --once arg
   └── brain.py: handle_message(user_input)

2. TRANSLATION (if BANTZ_LANGUAGE=tr)
   core/translation_layer.py: to_en(user_input)
   └── i18n/bridge.py: LanguageBridge.to_english()
       └── Helsinki-NLP opus-mt-tr-en via transformers (thread executor)
   Result: en_input (English string)

3. CONTEXT LOADING
   core/memory_injector.py: inject(ctx)
   └── asyncio.gather(
         memory/omni_memory.py: OmniMemoryManager.recall()  ← KG + ChromaDB + deep
         personality/persona.py: PersonaStateBuilder.build() ← CPU/time/app/affinity
         core/habits.py: HabitEngine.top_tools_for_segment()
       )
   Result: ctx.memory_combined, ctx.persona_state, ctx.style_hint populated

4. QUICK ROUTE (hardware-only regex, no LLM)
   core/routing_engine.py: quick_route(original, en_input)
   └── Matches: TTS stop, wake word on/off, audio duck on/off, clear memory
   └── Returns: ToolResult immediately OR None to continue

5. COT ROUTE (LLM-based intent classification)
   core/intent.py: cot_route(en_input, tool_schemas)
   └── Streams <thinking>…</thinking> then JSON plan via ollama.chat_stream()
   └── Returns: (plan_dict, thinking_str)  ← TUPLE, not dict
   plan_dict keys: "route" (tool name or "chat"), "args" (dict), "steps" (list)

6. TOOL DISPATCH
   core/routing_engine.py: execute_plan(plan, ctx)
   └── Single tool: tool_registry.get(route).execute(**args)
   └── Multi-step: agent/executor.py: PlanExecutor.run(steps)
       └── $REF_STEP_N resolved at Python dict level before each step
   Result: tool output string injected into ctx

7. LLM RESPONSE GENERATION
   llm/ollama.py: OllamaClient.chat_stream()
   └── System prompt: core/prompt_builder.py: build_chat_system(ctx, time_ctx)
   └── Streams response tokens to TUI / Telegram

8. FINALISATION
   core/finalizer.py: finalize(response, ctx)
   └── Strip markdown → hallucination check → EN→TR back-translation
   └── i18n/bridge.py: LanguageBridge.to_turkish() (thread executor)

9. OUTPUT
   └── TUI: rendered in Textual chat pane
   └── Telegram: streamed via edit_message_text()
   └── TTS: agent/tts.py: TTSEngine.speak() sentence-by-sentence via Piper + pw-play
```

---

## Critical Files

| File | Description |
|------|-------------|
| `src/bantz/core/brain.py` | Central orchestrator — all request handling flows through here |
| `src/bantz/core/intent.py` | `cot_route()` — the LLM routing brain; returns `(plan, thinking)` tuple |
| `src/bantz/config.py` | All configuration; `_voice_master_switch` cascades voice flags |
| `src/bantz/memory/bridge.py` | MemPalace adapter; `.enabled` is always False until `await init()` called |
| `src/bantz/agent/tts.py` | TTS engine — confirmed working; pw-play priority, sentence streaming |
| `src/bantz/agent/stt.py` | STT engine — broken; `faster-whisper` not installed |
| `src/bantz/interface/live_ui.py` | Textual TUI — primary user interface |
| `src/bantz/interface/telegram_bot.py` | Telegram integration — complete code, blocked on missing token |
| `src/bantz/agent/job_scheduler.py` | APScheduler — 6 cron jobs running; persistent reminders via SQLAlchemy |
| `src/bantz/core/memory_injector.py` | Parallel async memory injection into BantzContext before every prompt |

---

## Active Issues — Priority Order

### #460 — `finalizer.py` Hardcodes Ollama — Bypasses Configured LLM Provider (CRITICAL)
**Affected files**: `core/finalizer.py`
**What needs to change**:
- `finalize()` (line 123), the streaming path (line 190), and `synthesize_plan_response()` (line 300) all do `from bantz.llm.ollama import ollama` directly — they always use Ollama regardless of `config.llm_provider`
- When the user runs `bantz --setup claude` or `bantz --setup openai`, the configured provider is used for the main brain chat but finalizer always falls back to Ollama — if Ollama is not running, finalizer crashes
- Fix: replace the three hardcoded imports with a `get_llm()` factory (same pattern brain.py should use) that returns the active provider's client based on `config.llm_provider`
- The factory should live in `llm/__init__.py` or a new `llm/router.py` so all callsites share one lookup

---

### #461 — `summarizer.py` Hardcodes Ollama/Gemini — Bypasses Configured LLM Provider (CRITICAL)
**Affected files**: `tools/summarizer.py`
**What needs to change**:
- `SummarizerTool.execute()` checks `gemini.is_enabled()` and falls back to `ollama` (lines 79–88) — Claude and OpenAI are never considered
- When `BANTZ_LLM_PROVIDER=claude` or `BANTZ_LLM_PROVIDER=openai`, summaries silently fall back to Ollama if Gemini is not enabled
- Fix: use the same `get_llm()` factory from #460 so the summarizer respects the configured provider
- Affects every report or document-summary the planner produces

---

### ~~#462 — `_WSLogHandler.emit()` Uses `self._log_q` (AttributeError) — Desktop UI Logs Silently Dropped~~ ✅ FIXED (PR #467, merged 2026-06-04)
**Affected files**: `interface/ws_server.py`
**What was changed**:
- Line 734: `self._log_q.put_nowait` → `self._q.put_nowait` — one-character rename to match the attribute set in `__init__` (line 719)
- The bare `except Exception: pass` was swallowing the `AttributeError` silently; desktop UI now receives live log messages

---

### #465 — TUI Service Dots Always Show Ollama/Gemini — No Claude or OpenAI Indicator (HIGH)
**Affected files**: `interface/live_ui.py`
**What needs to change**:
- `_run_health_checks()` only checks Ollama (line 587) and Gemini (line 698) — when `BANTZ_LLM_PROVIDER=claude` or `openai`, the Ollama dot shows red even though Bantz is working correctly with the configured provider
- The `dots` renderer (line 292) needs to show the active provider dot instead of always showing Ollama
- Fix: add `check_claude()` and `check_openai()` coroutines alongside the existing Ollama/Gemini checks; only run the check that matches `config.llm_provider`; update the dot label to the active provider name
- `_get_llm_label()` (line 102) already returns the correct short label — use it for the dot text too

---

### #463 — TUI: Rich `Live` Full-Screen Layout Causes Terminal Paint UX — No Scrollback, Can't Copy Text (HIGH)
**Affected files**: `interface/live_ui.py`
**What needs to change**:
- `Live(screen=True, ...)` at line 1178 owns the entire terminal in an alternate buffer — on exit, all output vanishes; there is no scrollback and the user cannot select/copy text from previous responses
- Redraws cause visible flicker and overwrite terminal history
- Fix: replace full-screen `Live` with a panel-based approach that renders inline (no alternate screen), or switch to Textual's `App` which handles this correctly via its own compositor — Textual is already a dependency
- At minimum, remove `screen=True` and test whether the layout still works at reduced refresh rate

---

### #464 — TUI: `_erase_prompt_line()` Causes Double-Render Artifact — Whole TUI Block Duplicates on Screen (MEDIUM)
**Affected files**: `interface/live_ui.py:_erase_prompt_line`
**What needs to change**:
- `_erase_prompt_line()` (line 484) writes escape sequences to `sys.stdout` while `Rich Live` is running — this races with the Live render loop and causes the entire TUI layout to print a second copy on screen
- Fix: instead of writing escape sequences directly, update the prompt panel's `Renderable` to clear itself (set prompt text to empty string) and let the next Live refresh handle the erase — no raw stdout writes while Live is active
- Alternatively, call `self._live.stop()` → erase → `self._live.start()` but that causes flicker; prefer the renderable approach

---

### #422 — Async Streaming Translation to Reduce Turkish Response Latency from 18s to <10s (HIGH)
**Affected files**: `core/finalizer.py`, `i18n/bridge.py`, `interface/live_ui.py`, `interface/ws_server.py`
**What needs to change**:
- EN→TR back-translation in `finalizer.py` runs as a single blocking call on the full accumulated response — the user waits for LLM inference *plus* full-text translation before seeing any output (measured: 12–18s per turn)
- Fix path A (quick): switch EN→TR model from `opus-mt-tc-big-en-tr` to `opus-mt-en-tr` — smaller model, slightly lower quality, significantly faster
- Fix path B (correct): split response on sentence boundaries in `finalize()`, translate each sentence in the thread executor, stream translated sentences to the TUI/WS as they complete — matches how TTS already works sentence-by-sentence
- Fix path C (cache): add an in-memory LRU cache for translated butler stock phrases (greetings, error messages, capability descriptions) — these are short, high-frequency, and translate identically every time
- Combining A + C gives measurable improvement without structural changes

---

### #431 — Ghost Loop / STT Broken (CRITICAL)
**Affected files**: `agent/stt.py`, `agent/voice_capture.py`, `agent/ghost_loop.py`, `agent/wake_word.py`
**What needs to change**:
- Install `faster-whisper`, `pyaudio`, `webrtcvad` (add to `pyproject.toml` as optional `[voice]` extras)
- `STTEngine._ensure_model()` currently silently returns `False` on ImportError — surface this as a visible TUI warning
- `VoiceCapture.capture()` silently fails when PyAudio is missing — add check in `__init__`
- `WakeWordListener` needs `BANTZ_PICOVOICE_ACCESS_KEY` populated in `.env`
- All three failures are invisible to the user; ghost loop appears "running" but never hears anything

---

### #440 — First-Run Onboarding Missing (HIGH)
**Affected files**: `interface/live_ui.py`, `__main__.py`, `cli/setup.py`
**What needs to change**:
- On fresh TUI launch, detect if conversation history is empty → show welcome banner with capability list
- Add `bantz --setup onboarding` or inline first-run wizard before the chat cursor appears
- Banner must confirm Ollama is running and list what works (text chat, Turkish, TTS) vs what needs setup (voice)
- `bantz --once` must emit a "Loading models…" progress line before the 15–30s silence window

---

### #442 — Raw Tracebacks Break Persona (HIGH)
**Affected files**: `core/brain.py`, `core/finalizer.py`, `interface/live_ui.py`, `interface/telegram_bot.py`
**What needs to change**:
- Wrap all tool execution and LLM calls in `try/except` at the brain level
- Map exception types to butler-voice error messages (e.g., "I'm afraid I encountered a slight mechanical difficulty, ma'am")
- `interface/live_ui.py` must catch widget render errors and display styled error cells, not raw tracebacks
- `interface/telegram_bot.py` must never send a Python traceback to the user

---

### #432 — `--doctor` MemPalace False Negative (HIGH)
**Affected files**: `cli/setup.py:_doctor()`
**What needs to change**:
- `_doctor()` checks `palace_bridge.enabled` which is always `False` at import time
- Fix: add `await palace_bridge.init()` before the enabled check inside `_doctor()`
- Also fix: Ollama tool count shows 0 in doctor output; needs `await` on tool schema fetch
- Group voice failures together with actionable fix commands (pip install commands) instead of flat list

---

### #433 — MarianMT Wrong Routing (MEDIUM)
**Affected files**: `core/intent.py:COT_SYSTEM`, `core/routing_engine.py:quick_route()`
**What needs to change**:
- Turkish hardware/system queries must be caught by `quick_route()` regex patterns before translation
- Add Turkish keyword patterns: `cpu kullanımı`, `ram`, `bellek`, `disk`, `işlemci`, `sistem` → route to `system` tool
- Add Turkish examples to `COT_SYSTEM` prompt in `intent.py:_ROUTING_HINTS` for the system tool
- Post-translation normalization: "What is the use of X" → "What is X usage" for metric queries

---

### #437 — TUI Status Bar Missing (MEDIUM)
**Affected files**: `interface/live_ui.py`
**What needs to change**:
- Add a persistent footer/status bar to the Textual TUI showing service health indicators
- Required dots: Ollama (green/red), MemPalace (green/red), TTS (green/red), STT (green/red), Voice (green/red)
- Health checks must be non-blocking (async, polled every 30s in background)
- Status bar should also show: model name, active persona state, memory drawer count

---

### #434 — Guided Voice Setup Wizard Missing (MEDIUM)
**Affected files**: `cli/setup.py:_handle_setup()`
**What needs to change**:
- Implement `bantz --setup voice` wizard that:
  1. Checks which of `faster-whisper`, `pyaudio`, `webrtcvad`, `pvporcupine` are installed
  2. Offers to run `pip install` for missing packages
  3. Prompts for `BANTZ_PICOVOICE_ACCESS_KEY` and writes it to `.env`
  4. Sets `BANTZ_VOICE_ENABLED=true`, `BANTZ_STT_ENABLED=true`, `BANTZ_GHOST_LOOP_ENABLED=true` in `.env`
  5. Runs a mic test + STT test + TTS test and reports pass/fail
- Add `bantz --setup voice` to argparse in `__main__.py`

---

### #435 — `bantz --once` Silent Hang (MEDIUM)
**Affected files**: `__main__.py:_once()`, `core/brain.py`, `i18n/bridge.py`
**What needs to change**:
- `_once()` must emit progress to stderr before any async operation:
  - "Loading translation model…" before `bridge.to_english()`
  - "Thinking…" before Ollama call
  - "Synthesizing voice…" before TTS
- MarianMT load time (~3s) and Ollama inference (~8–15s) happen in complete silence currently
- Consider pre-warming MarianMT in a background thread during startup

---

### #439 — VLM Vision Never Called (LOW-MEDIUM)
**Affected files**: `vision/remote_vlm.py`, `tools/visual_click.py`, `config.py`
**What needs to change**:
- `BANTZ_VLM_ENABLED=false` disables `describe_screen()` entirely
- Add a local VLM option (e.g., `llava` via Ollama) so vision works without a remote server
- When user asks "ekran görüntüsü al ve anlat" (take screenshot and describe), route to `screenshot` + local VLM
- Update `remote_vlm.py` to support `BANTZ_VLM_BACKEND=ollama` with `BANTZ_VLM_MODEL=llava`

---

### #438 — AffinityEngine Never Fires (LOW)
**Affected files**: `core/rl_hooks.py`, `agent/affinity_engine.py`, `config.py`
**What needs to change**:
- `BANTZ_RL_ENABLED=false` causes `rl_reward_hook()` to always return early
- 96 interactions collected in `messages` table are never used for personalization
- Fix: set `BANTZ_RL_ENABLED=true` in default `.env.example` (AffinityEngine is safe to enable)
- Ensure `AffinityEngine` is initialized in `DataLayer.__init__()` when `rl_enabled=True`
- Wire `HabitEngine.top_tools_for_segment()` output into `memory_injector.inject(ctx)`

---

### #436 — Redis Dead Reference (LOW)
**Affected files**: `memory/bridge.py:178`, `interface/telegram_bot.py`
**What needs to change**:
- Remove or update the Redis architecture comment at `bridge.py:178` — Redis was never implemented
- `telegram_bot.py:_active_chats` is a plain Python set lost on restart; replace with SQLite KV store using `data/sqlite_store.py:SQLiteKVStore`
- `SQLiteKVStore` already exists and supports set operations — this is a 10-line fix

---

### #441 — AmbientEngine Blocked by Picovoice (LOW)
**Affected files**: `agent/ambient.py`, `agent/wake_word.py`
**What needs to change**:
- `ambient_analyzer.feed_frames()` is only called from `WakeWordListener._listen_loop()`
- When Picovoice is unavailable, AmbientAnalyzer is completely unreachable
- Add `StandaloneAmbientSampler` class in `agent/ambient.py` that opens a raw PyAudio stream independently of Porcupine
- Guard with `if config.ambient_enabled and not config.wake_word_enabled: start standalone sampler`

---

## Conventions

### Tool Registration
Every tool must:
1. Subclass `BaseTool` from `tools/__init__.py`
2. Implement `name: str`, `description: str`, `risk_level: str`, and `execute(**kwargs) -> ToolResult`
3. Call `registry.register(MyTool())` at the **bottom** of its module file (not inside any class or function)
4. Be imported in `tools/__init__.py` so the registry populates on Brain init

```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    risk_level = "low"

    async def execute(self, param: str) -> ToolResult:
        return ToolResult(success=True, output=f"Done: {param}")

registry.register(MyTool())
```

### Routing
- `quick_route(original_text, en_text)` — hardware-only regex; returns `ToolResult` or `None`
- If `quick_route()` returns `None`, call `cot_route(en_text, tool_schemas)`
- `cot_route()` **always returns a tuple**: `(plan_dict, thinking_str)` — never call `.get()` directly on the return value
- `plan_dict["route"]` is the tool name or `"chat"` for conversational responses
- `plan_dict["args"]` is a dict of kwargs to pass to the tool's `execute()` method

### Async Patterns
- All I/O is `async`/`await` — never use `time.sleep()`, always `asyncio.sleep()`
- CPU-bound work (translation, image processing) runs in `asyncio.get_event_loop().run_in_executor(None, ...)`
- Database access uses the connection pool in `data/connection_pool.py` — never open raw `sqlite3.connect()` outside it
- `memory_injector.inject(ctx)` uses `asyncio.gather()` — all memory sources queried in parallel

### Required Environment Variables
These must be set in `.env` for core features:
```
BANTZ_LANGUAGE=tr                        # enables MarianMT translation
BANTZ_TTS_ENABLED=true                   # enables Piper TTS output
BANTZ_MEMPALACE_ENABLED=true             # enables ChromaDB+KG memory
BANTZ_PERSONA_ENABLED=true               # enables 6-state persona system
OLLAMA_HOST=http://localhost:11434       # Ollama server URL
BANTZ_LLM_MODEL=llama3.1:8b             # primary LLM model

# Voice (currently broken — all three packages missing):
BANTZ_VOICE_ENABLED=true                 # master switch (cascades all below)
BANTZ_PICOVOICE_ACCESS_KEY=<key>         # required for wake word
```

---

## What NOT to Touch

These modules are stable, well-tested, and should not be modified unless a specific issue directly requires it:

| Module | Reason |
|--------|--------|
| `data/connection_pool.py` | Thread-safe SQLite pool — subtle concurrency logic; changes cause race conditions |
| `data/sqlite_store.py` | Generic KV store used by scheduler, memory, and reminders — stable API |
| `llm/ollama.py` | Async HTTP client with streaming — battle-tested; changes break all LLM calls |
| `i18n/bridge.py` | MarianMT wrapper with thread executor and chunking — works correctly |
| `core/context.py` | `BantzContext` dataclass — adding fields is OK, renaming or removing breaks the entire pipeline |
| `agent/workflows/` | Night job implementations — only touch for scheduler-related issues |
| `personality/persona.py` | 6-state persona system — stable; only touch for issue #442 (error message persona) |
| `memory/omni_memory.py` | Parallel gather with token budget — correct logic; don't touch the budget math |
| `agent/tts.py` | TTS confirmed working end-to-end — only touch if a TTS-specific issue arises |
| `core/prompt_builder.py` | `CHAT_SYSTEM` template is carefully tuned — anti-hallucination rules are load-bearing |
