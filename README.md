<div align="center">

<pre>
                      \//        \\/
                      //          \\
                     //\          /\\
                    ///\          /\\\
    ____     _    _   _ \________/  ____
   | __ )   / \  | \ | | |__   __||__  /
   |  _ \  / _ \ |  \| |    | |     / /
   | |_) |/ ___ \| |\  |    | |    / /_
   |____//_/   \_\_| \_|    |_|   /____|
</pre>

**Local-First AI Agent — Terminal Interface, System Observer, Autonomous Operator**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: 2991](https://img.shields.io/badge/tests-2991-brightgreen.svg)](#test-suite)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)
[![Textual](https://img.shields.io/badge/TUI-Textual-purple.svg)](https://textual.textualize.io)

</div>

---

## What is Bantz?

Bantz is a **local-first, autonomous AI agent** that lives in your terminal. It handles tasks through natural language, observes your system in real time, manages your calendar and email, controls your desktop when needed, and operates proactively in the background — all without sending your data to the cloud.

It is not a wrapper around a hosted API. The entire reasoning, memory, and tool execution stack runs on your machine.

**Design principles:**

- **Local-first.** Powered by Ollama (`qwen3:8b` / `llama3`), SQLite, and on-device graph memory. No data leaves the machine by default.
- **Chain-of-Thought routing.** Every request goes through a structured CoT classifier that extracts intent, selects a tool, and validates parameters before executing anything.
- **Autonomous background agent.** Runs as a systemd service. Performs nightly reflection, cache maintenance, wake-word listening, and proactive suggestions — independently.
- **Butler persona.** Polite, discreet, and adaptive. A reinforcement learning layer adjusts tone and formality based on your implicit feedback over time.

---

## Chain-of-Thought Reasoning

Before any tool fires, Bantz reasons through the request in a quarantined scratchpad. The `<thinking>` block is streamed live to the TUI's dedicated thinking panel and stripped before the final JSON output is parsed.

```
User request: "Check the PDF and email the summary."
                           │
                           ▼
              ┌────────────────────────┐
              │      <thinking>        │
              │  1. What file?         │
              │     What address?      │
              │  2. Tools needed:      │
              │     filesystem +       │
              │     document + gmail   │
              │  3. Do not invent      │
              │     content. Read      │
              │     the file first.    │
              │  </thinking>           │
              └────────────────────────┘
                           │
                           ▼
              { "route": "planner",
                "tool_name": null,
                "reasoning": "Needs document
                              read then gmail" }
```

Routing decisions carry a confidence score. Below the configured threshold (`BANTZ_COT_CONFIDENCE_THRESHOLD`), the agent asks for clarification rather than guessing.

---

## Architecture

Bantz is built in five decoupled layers. Each layer communicates through typed contracts — no layer reaches into another's internals.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INTERFACE LAYER                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐    │
│  │ TUI (Textual)   │  │ Telegram Bot     │  │ CLI             │    │
│  │ Chat + Thinking │  │ async, progress  │  │ --once          │    │
│  │ Panel + Mood    │  │ indicators       │  │ --daemon        │    │
│  │ + System Stats  │  │                  │  │ --doctor        │    │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘    │
├───────────┴─────────────────────┴───────────────────── ┴────────────┤
│                            CORE LAYER                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Brain    │  │ Intent   │  │ Planner  │  │ Executor         │   │
│  │ (orch.)  │  │ (CoT     │  │ (Plan-   │  │ (step runner,    │   │
│  │          │  │  router) │  │  Solve)  │  │  $REF binding)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Finalizer│  │ Session  │  │ Notif.   │  │ EventBus         │   │
│  │ (output  │  │ Tracker  │  │ Manager  │  │ (async pub/sub)  │   │
│  │  format) │  │          │  │          │  │                  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                           AGENT LAYER                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Ghost    │  │ RL       │  │ Interventions│  │ Job Scheduler │  │
│  │ Loop     │  │ Engine   │  │ (priority Q  │  │ (APScheduler) │  │
│  │ (bg obs.)│  │ (Q-learn)│  │  + rate lim.)│  │               │  │
│  └──────────┘  └──────────┘  └──────────────┘  └───────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Observer │  │ App      │  │ Proactive    │  │ Affinity      │  │
│  │ (stderr) │  │ Detector │  │ (idle-aware  │  │ Engine (RLHF) │  │
│  │          │  │          │  │  engagement) │  │               │  │
│  └──────────┘  └──────────┘  └──────────────┘  └───────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Wake     │  │ STT      │  │ TTS          │  │ Audio Ducker  │  │
│  │ Word     │  │ (Whisper)│  │ (Piper+aplay)│  │ (vol. ctrl)   │  │
│  │(Porcupn.)│  │          │  │              │  │               │  │
│  └──────────┘  └──────────┘  └──────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                            DATA LAYER                               │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │ SQLite + FTS5  │  │ Vector Store   │  │ Graph Memory (Neo4j)  │ │
│  │ (conversations,│  │ (cosine sim.,  │  │ (entity graph,        │ │
│  │  Q-table, mood)│  │  embeddings)   │  │  Cypher queries)      │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │ Connection Pool│  │ Distiller      │  │ Spatial Cache         │ │
│  │ (SQLite WAL,   │  │ (LLM summary   │  │ (UI element coords,   │ │
│  │  thread-safe)  │  │  → vectors)    │  │  24h TTL)             │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      TOOLS LAYER  (21 registered tools)             │
│  shell · gmail · calendar · classroom · filesystem · document       │
│  weather · news · web_search · web_reader · browser_control         │
│  visual_click · input_control · accessibility · system              │
│  reminder · contacts · gui_action · summarizer · read_url           │
├─────────────────────────────────────────────────────────────────────┤
│                          VISION LAYER                               │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │ AT-SPI         │  │ Remote VLM     │  │ Navigator             │ │
│  │ (<10ms,        │  │ (Jetson/Colab  │  │ (unified fallback     │ │
│  │  no screenshot)│  │  REST client)  │  │  chain)               │ │
│  └────────────────┘  └────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

125 modules, ~39,500 lines of Python.

```
src/bantz/
├── __main__.py                   # CLI entry point (--once, --daemon, --doctor, --setup)
├── config.py                     # ~70 settings via .env (Pydantic Settings)
│
├── core/                         # Orchestration & Routing
│   ├── brain.py                  # Main request orchestrator
│   ├── intent.py                 # CoT routing — streaming <thinking> + JSON output
│   ├── planner.py                # Plan-and-Solve for multi-tool sequences
│   ├── executor.py               # Sequential plan runner with $REF variable binding
│   ├── finalizer.py              # Output formatting + hallucination guard
│   ├── event_bus.py              # Async pub/sub (thinking events, TUI bridge)
│   ├── notification_manager.py   # Cross-surface notification dispatch
│   └── session.py                # Conversation session tracking
│
├── data/                         # Unified Data Access Layer
│   ├── store.py                  # Abstract base classes (7 store contracts)
│   ├── layer.py                  # Singleton DataLayer — composes all stores
│   ├── connection_pool.py        # Thread-safe SQLite WAL connection pool
│   └── migration.py              # Versioned schema migration utility
│
├── memory/                       # Multi-tier Memory
│   ├── vector_store.py           # Pure SQLite vector store (cosine similarity)
│   ├── graph.py                  # Neo4j knowledge graph (entities + relations)
│   ├── omni_memory.py            # OmniMemoryManager — context bloat control
│   └── context_builder.py        # Graph → LLM context injection
│
├── agent/                        # Autonomous Background Subsystems
│   ├── ghost_loop.py             # Background observer loop
│   ├── observer.py               # Stderr monitoring + error classification
│   ├── planner.py                # Planning agent (tools: SummarizerTool, $REF)
│   ├── executor.py               # Plan step executor
│   ├── interventions.py          # Priority queue + rate limiting + focus mode
│   ├── rl_engine.py              # Q-learning (1,680 states, SQLite Q-table)
│   ├── affinity_engine.py        # RLHF — sentiment-driven reward shaping
│   ├── proactive.py              # Idle detection → proactive suggestions
│   ├── job_scheduler.py          # APScheduler integration
│   ├── notifier.py               # Desktop notification dispatch
│   ├── app_detector.py           # Active window / activity category detection
│   ├── health.py                 # System health monitor
│   ├── stt.py                    # Whisper STT (faster-whisper, VAD-gated)
│   ├── tts.py                    # Piper TTS + aplay streaming
│   ├── wake_word.py              # Porcupine offline wake-word detection
│   ├── voice_capture.py          # Microphone capture pipeline
│   ├── audio_ducker.py           # Volume control during TTS playback
│   └── workflows/
│       ├── maintenance.py        # Nightly cache + DB maintenance (03:00)
│       ├── reflection.py         # Evening self-reflection summary (23:00)
│       └── overnight_poll.py     # Background task polling
│
├── personality/
│   ├── system_prompt.py          # Dynamic system prompt construction
│   ├── persona.py                # Mood-scaled persona adaptation
│   └── bonding.py                # RL bonding meter — formality progression
│
├── vision/                       # Desktop Perception
│   ├── navigator.py              # AT-SPI → screenshot → VLM fallback chain
│   ├── browser_vision.py         # Browser-specific visual interaction
│   └── remote_vlm.py             # REST client for remote VLM (Jetson/Colab)
│
├── tools/                        # Tool Implementations
│   ├── shell.py                  # Bash with risk classification + audit log
│   ├── gmail.py                  # Full Gmail CRUD + auto-chain compose/send
│   ├── calendar.py               # Google Calendar CRUD
│   ├── filesystem.py             # File ops + LLM auto-chain create-on-miss
│   ├── browser_control.py        # Firefox/browser subprocess automation
│   ├── visual_click.py           # AT-SPI + screenshot-based UI element clicks
│   ├── input_control.py          # Keyboard/mouse input via xdotool
│   ├── accessibility.py          # AT-SPI tree traversal + app inspection
│   ├── gui_action.py             # High-level GUI action sequencer
│   ├── web_search.py             # DuckDuckGo search (no API key)
│   ├── web_reader.py             # URL fetch → clean text extraction
│   ├── document.py               # PDF/TXT/MD/DOCX summarize + Q&A
│   ├── summarizer.py             # General-purpose text summarizer tool
│   ├── system.py                 # CPU/RAM/disk/uptime metrics via psutil
│   ├── weather.py                # Weather lookup
│   ├── news.py                   # News + HackerNews headlines
│   ├── reminder.py               # Reminder CRUD (SQLite-backed)
│   ├── classroom.py              # Google Classroom assignments
│   └── contacts.py               # Contact resolution + lookup
│
├── interface/
│   ├── telegram_bot.py           # Telegram integration ("Hold the Line" UX)
│   └── tui/
│       ├── app.py                # Textual App — main TUI entry point
│       ├── styles.tcss           # TUI stylesheet
│       ├── mood.py               # MoodStateMachine + SQLite mood history
│       └── panels/
│           ├── chat.py           # Chat log + thinking panel
│           ├── header.py         # Operations header + service health indicators
│           └── system.py         # Live system telemetry panel
│
└── llm/
    ├── ollama.py                 # Ollama streaming client (primary)
    └── gemini.py                 # Gemini REST client (fallback)
```

---

## Tool Inventory

| Tool | Capability | Risk Level |
|------|-----------|-----------|
| `shell` | Execute arbitrary bash commands with denylist enforcement | destructive |
| `gmail` | Read, compose, send, search, filter email | moderate |
| `calendar` | Create, read, update, delete calendar events | moderate |
| `filesystem` | Read, write, create files and directories | moderate |
| `document` | Summarize or query PDF, TXT, MD, DOCX files | safe |
| `web_search` | DuckDuckGo search — no API key required | safe |
| `web_reader` | Fetch and extract clean text from any URL | safe |
| `browser_control` | Subprocess-based browser automation (Firefox) | moderate |
| `visual_click` | Click any visible UI element via AT-SPI or screenshot | moderate |
| `input_control` | Keyboard and mouse control via xdotool | moderate |
| `accessibility` | AT-SPI tree traversal, window inspection, focus | safe |
| `system` | CPU, RAM, disk, uptime via psutil | safe |
| `weather` | Current weather and forecast | safe |
| `news` | Headlines from multiple sources, HackerNews | safe |
| `reminder` | SQLite-backed reminder CRUD | safe |
| `classroom` | Google Classroom assignment listing | safe |
| `contacts` | Contact lookup and resolution | safe |
| `summarizer` | LLM-powered text summarization | safe |

The intent router (`cot_route`) classifies every request into one of: `tool`, `planner`, `chat`. The planner activates when a request requires two or more tools in sequence and coordinates the full execution chain via `$REF` variable binding between steps.

---

## Voice Pipeline

Bantz supports fully offline, hands-free voice interaction. All components run locally — no cloud STT or TTS services.

Enable with a single flag:

```bash
# .env
BANTZ_VOICE_ENABLED=true
```

**Pipeline:**

```
Microphone → VAD (WebRTC) → Wake Word (Porcupine) → STT (Whisper tiny)
                                                           │
                                                           ▼
                                                     Brain → Tool
                                                           │
                                                           ▼
                                              TTS (Piper) → aplay
                                         (audio ducked during playback)
```

**Prerequisites (Linux):**

```bash
sudo apt install portaudio19-dev
pip install pyaudio faster-whisper webrtcvad pvporcupine piper-tts
```

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BANTZ_VOICE_ENABLED` | `false` | Master switch for all voice features |
| `BANTZ_WAKE_WORD_SENSITIVITY` | `0.5` | Porcupine detection sensitivity (0.0–1.0) |
| `BANTZ_STT_MODEL` | `tiny` | Whisper model size (`tiny`, `base`, `small`) |
| `BANTZ_TTS_VOICE` | `en_US-lessac-medium` | Piper voice model |

Run `bantz --doctor` to verify all voice dependencies are satisfied before first use.

---

## Configuration

All settings are loaded from `.env` in the project root via Pydantic Settings. Every variable is prefixed with `BANTZ_`.

```bash
# Core LLM
BANTZ_OLLAMA_MODEL=qwen3:8b
BANTZ_OLLAMA_BASE_URL=http://localhost:11434
BANTZ_GEMINI_API_KEY=          # optional cloud fallback

# Memory
BANTZ_NEO4J_URI=bolt://localhost:7687
BANTZ_NEO4J_USER=neo4j
BANTZ_NEO4J_PASSWORD=

# Voice (all disabled by default)
BANTZ_VOICE_ENABLED=false
BANTZ_WAKE_WORD_SENSITIVITY=0.5

# Notifications
BANTZ_DESKTOP_NOTIFICATIONS=true
BANTZ_NOTIFICATION_SOUND=false

# Agent behavior
BANTZ_COT_CONFIDENCE_THRESHOLD=0.4
BANTZ_PERSONA_ENABLED=true
BANTZ_DEEP_MEMORY_ENABLED=true
```

---

## Installation

```bash
git clone https://github.com/miclaldogan/bantzv2
cd bantzv2
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in credentials
bantz --doctor         # verify environment
bantz                  # launch TUI
```

**Requirements:** Python 3.11+, Ollama running locally, SQLite (stdlib).

**Optional:** Neo4j (graph memory), Redis (session store), PortAudio (voice), `xdotool` (desktop automation), `chafa` (image rendering).

---

## Running Modes

```bash
bantz                        # Launch interactive TUI
bantz --once "ls -la"        # Single query, print response, exit
bantz --daemon               # Background service mode (no TUI)
bantz --doctor               # Verify all dependencies
bantz --mood-history         # Print last 24h mood transitions
```

For persistent background operation, a systemd unit is provided at `deploy/bantz.service`.

---

## Test Suite

```bash
pytest                        # Run full suite
pytest tests/core/            # Core routing and brain tests
pytest tests/tools/           # Tool-level integration tests
pytest -q                     # Compact output
```

2991 tests, 0 failures. Coverage spans intent routing, tool execution, agent loop, memory, TUI event bridge, and voice pipeline components.

---

## Roadmap

Active development priorities for v3:

| # | Feature | Layer | Status |
|---|---------|-------|--------|
| **#1** | `BrowserTool` — curl + pup + readability pipeline (zero API dependency) | web | In progress |
| **#2** | `FeedTool` — RSS/Atom feed parser via xmllint, YAML feed registry | web | Planned |
| **#3** | `ImageTool` — terminal image rendering via chafa, 24h cache | web | Planned |
| **#4** | `SystemTool` — unified subprocess interface, audit log, safe-mode denylist | system | In progress |
| **#5** | `GUITool` — pyautogui + xdotool bridge, DRY_RUN mode | system | In progress |
| **#6** | Neo4j memory — entity graph, NER extraction, Cypher context retrieval | memory | In progress |
| **#7** | Redis session store — in-flight state, task queue, TUI↔Telegram pub/sub | memory | Planned |
| **#8** | APScheduler — cron + one-shot tasks, Redis job store, natural language scheduling | scheduler | In progress |
| **#9** | TUI migration: Textual → Rich Live (diff rendering, native asyncio, mouse scroll) | interface | Planned |

**Build order:** `#4 → #1 → #7 → #6 → #2 → #3 → #5 → #8 → #9`

Issues #1–#8 are tracked with full acceptance criteria, implementation notes, and dependency graphs in `bantz_v3_issues.md`. Issue #9 (TUI migration) is detailed with a complete migration checklist in `bantz_v3_issue_tui_migration.md`.

---

## Recent Completed Work

| # | Feature |
|---|---------|
| **#287** | Voice feedback loop fix + full TUI integration audit |
| **#286** | Planner: `$REF` variable binding, `SummarizerTool`, Butler lore toasts |
| **#285** | `OmniMemoryManager` — context bloat control + GraphRAG integration |
| **#284** | `BANTZ_VOICE_ENABLED` master switch |
| **#282** | `_is_refusal` thinking-aware detection — prevents false refusals on `<thinking>` content |
| **#277** | Voice pipeline: Porcupine wake word + Whisper STT + Piper TTS |
| **#273** | Streaming `<thinking>` events to TUI ThinkingPanel in real time |
| **#253** | People-Pleaser guard — malformed JSON returns error, no silent chat fallback |
| **#220** | EventBus → Textual thread bridge (no more `call_from_thread` footguns) |
| **#183** | Telegram async progress indicators ("Hold the Line" UX) |

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

<div align="center">

Built by [@miclaldogan](https://github.com/miclaldogan)

</div>
