<div align="center">

<img src="bantz.png" alt="Bantz" width="700"/>

# BANTZ

**Your AI-Powered Personal Host — Terminal Assistant, System Observer, Autonomous Agent**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: 2261](https://img.shields.io/badge/tests-2261-brightgreen.svg)](#test-suite)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)
[![Textual](https://img.shields.io/badge/TUI-Textual-purple.svg)](https://textual.textualize.io)

*Bantz isn't just an assistant — it's the host of your machine.*

</div>

---

## What is Bantz?

Bantz is a **local-first, privacy-respecting AI assistant** that lives in your terminal. It combines conversational AI with tool execution, persistent memory, background autonomy, and hardware telemetry — all orchestrated from a rich TUI.

**Key principles:**
- **100% local by default** — Ollama LLMs, SQLite storage, no cloud dependency
- **Your data stays on your machine** — optional cloud services (Gemini, Gmail, Calendar) are opt-in
- **Autonomous agent** — observes errors, learns routines, runs maintenance, briefs you in the morning
- **Terminal-native** — Textual TUI with sparklines, live telemetry, streaming chat

---

## Features

### Conversational AI
| Feature | Description |
|---------|-------------|
| **Ollama (local)** | Primary LLM — `qwen3:8b` or any GGUF model. Zero cloud cost |
| **Gemini (fallback)** | Optional cloud fallback via `gemini-2.0-flash` |
| **Chain-of-Thought** | CoT intent classification → tool selection → structured output |
| **Streaming** | Token-by-token streamed responses in the TUI |
| **TR ↔ EN** | Automatic translation bridge — speak Turkish, tools run in English |

### Tool Execution (18 tools)
| Tool | Description |
|------|-------------|
| **Shell** | Bash commands with security controls — blocked commands, destructive ops require confirmation |
| **Gmail** | Read, search, filter, compose, send, reply. NL queries: *"unread emails from professor"* |
| **Google Calendar** | View today/week, create/update/delete events |
| **Google Classroom** | Assignments, due dates, course announcements |
| **Weather** | Current conditions + 3-day forecast via wttr.in (no API key) |
| **News** | Hacker News + Google News headlines with LLM summarization |
| **File ops** | Read, write, list — sandboxed to home directory |
| **Documents** | Read and summarize PDF, DOCX, TXT, MD files |
| **Reminders** | Time and place-based reminders with snooze, repeat, geofence triggers |
| **Web search** | DuckDuckGo search with LLM synthesis |
| **Contacts** | Local address book |
| **System info** | CPU, RAM, disk, uptime |
| **GUI action** | Click, type, scroll bridged to input control |
| **AT-SPI reader** | Instant UI element detection via accessibility tree (<10ms) |
| **Input control** | Mouse/keyboard simulation with safety model (safe → moderate → destructive) |

**Auto-chaining:** Tools detect follow-up actions and offer them automatically:
- **Gmail** — composing an email → *"Shall I send it, sir?"* (one-click confirmation)
- **Filesystem** — referencing a non-existent file/folder → LLM extracts path & content → auto-creates it
- **Multi-step** — complex requests decomposed via Plan-and-Solve (see below)

### Multi-Tool Workflows & Plan-and-Solve
**Workflow engine** — detects chained commands, orchestrates sequential tool calls:  
*"Send email to prof, add meeting to calendar, remind me tomorrow at 9"* → 3 tools, 1 command.

**Plan-and-Solve** — when a request is too complex for a single chain, the Planner decomposes it into numbered steps, announces an *itinerary*, then the Executor runs each step sequentially with inter-step context passing:  
*"Research the best flight to London, draft an email to my boss asking for time off, then create a calendar event for the trip"*  
→ Planner produces 3-step JSON plan → Executor runs each with `{step_N_output}` substitution → butler-style summary.

### Memory System
| Layer | Technology | Purpose |
|-------|------------|---------|
| **Conversations** | SQLite + FTS5 | Full-text searchable chat history |
| **Vector memory** | SQLite BLOBs + cosine similarity | Semantic cross-session recall via `nomic-embed-text` (768-dim) |
| **Session distillation** | LLM summarization → vector embed | Previous sessions compressed to searchable summaries |
| **Knowledge graph** | Neo4j (optional) | Entities: Person, Topic, Decision, Task, Event, Location, Document, Commitment |
| **Context builder** | Graph query → LLM prompt | Injects relevant entities/relationships into conversation |

### Personality & Bonding
| Feature | Description |
|---------|-------------|
| **Bonding meter** | RL-driven formality score (0–100). Starts formal, gradually relaxes as trust builds through positive interactions |
| **Progressive formality** | System prompt dynamically adjusts tone: *"Good day, sir"* → *"Hey, what's up?"* based on bonding level |
| **Dynamic persona** | LLM persona adapts based on system telemetry — stressed system → calming tone, idle → playful |
| **RLHF feedback** | Detects sentiment & explicit keywords (*"great answer"*, *"that's wrong"*) → reward/penalty signals to RL engine |
| **Spontaneous recall** | Vector memory retrieval surfaces relevant past conversations unprompted during chat |

### Senses & Awareness
| Feature | Description |
|---------|-------------|
| **Wake word** | Offline *"Hey Bantz"* detection via Porcupine PPN — always-on listening without cloud |
| **Ambient audio** | Periodic microphone sampling to detect environmental context (quiet, noisy, music) |
| **Audio ducking** | System volume automatically lowers during TTS playback and wake word listening |
| **Proactive engagement** | Idle detection → butler initiates conversation based on time-of-day and recent activity |
| **Health interventions** | Telemetry-driven break reminders — posture alerts, hydration, screen-time limits |

### Vision & OS Control (Hybrid Pipeline)
```
AT-SPI (< 10ms) → Spatial Cache (< 1ms) → Remote VLM (2-5s) → Give up
```
| Component | What it does |
|-----------|-------------|
| **AT-SPI reader** | Reads accessibility tree — bounding boxes for every UI element (GTK, Qt, Chromium/Electron) |
| **Spatial cache** | SQLite cache for element coordinates, 24h TTL, confidence decay, LRU eviction at 1000 entries |
| **Remote VLM** | Screenshot → base64 JPEG → VLM endpoint (Jetson Nano, Colab, or local Ollama VLM) |
| **Navigator** | Unified fallback chain with per-app analytics to learn which method works |
| **Input control** | PyAutoGUI/pynput with safety tiers: safe (click), moderate (type), destructive (hotkeys) |

### Background Agent System
| Component | Issue | What it does |
|-----------|-------|-------------|
| **Stderr Observer** | #124 | Monitors terminal error streams. Classifies: ignore → log → toast → full LLM analysis popup |
| **RL Engine** | #125 | Q-learning over ~1680 states (time × day × location × recent_tool). Learns which proactive suggestions you accept |
| **Interventions** | #126 | Priority queue bridging RL/Observer → user. Rate limiting, focus mode, explainability labels |
| **App Detector** | #127 | Active window → activity category (Coding, Browsing, Entertainment, Idle). X11/Wayland/AT-SPI/proc |
| **Job Scheduler** | #128 | APScheduler with SQLAlchemy job store. Misfire grace = 86400s (laptop sleep recovery) |
| **Maintenance** | #129 | 3 AM: Docker cleanup, temp purge, disk health, service checks, log rotation. Dry-run mode |
| **Reflection** | #130 | 11 PM: Hierarchical summarization of the day. Vector orphan cleanup. Entity resolution (deduplicated) |
| **TTS Briefing** | #131 | Piper + aplay streaming. Sentence-by-sentence pipeline (synth N+1 while playing N). Instant stop via SIGTERM |
| **Overnight Poll** | #132 | Gmail/Calendar/Classroom every 2h overnight. KV store with dedup. Urgent keyword detection |
| **Desktop Notifier** | — | `notify-send` integration. Smart dispatch: skips if TUI active, priority → urgency mapping |
| **Wake Word** | #165 | Offline Porcupine PPN — *"Hey Bantz"* triggers the assistant without keyboard input |
| **Ambient Audio** | #166 | Periodic microphone sampling — environment classification (quiet / noisy / music) for context |
| **Proactive Engine** | #167 | Idle → butler initiates conversation; combines time-of-day, last interaction, and app context |
| **Health Monitor** | #168 | Posture / hydration / screen-time reminders based on elapsed active time and telemetry |
| **Audio Ducker** | #171 | PulseAudio/PipeWire volume reduction during TTS playback and wake word listening |
| **RLHF Feedback** | #180 | Sentiment + keyword detection → direct reward/penalty to Q-table (no explicit thumbs up) |
| **Planner** | #187 | LLM-powered Plan-and-Solve — decomposes complex requests into numbered JSON step arrays |
| **Executor** | #187 | Sequential step runner with `{step_N_output}` context substitution and graceful failure |

**Night Schedule:**
```
11 PM → Memory Reflection (summarize day's conversations, entity extraction)
 3 AM → System Maintenance (Docker prune, disk cleanup, service health)
 2h   → Email/Calendar/Classroom polls (store results for morning)
 7 AM → Audio Morning Briefing via TTS on first unlock
```

### Hardware Telemetry & TUI
| Metric | Source | Display |
|--------|--------|---------|
| CPU % | psutil | Bar + Sparkline (60-reading, 2-min window) |
| RAM % | psutil | Bar + Sparkline |
| Disk % | psutil | Bar + Sparkline |
| Net TX/RX | psutil delta math | MB/s rate + Sparkline |
| CPU Temp | psutil sensors | Colored indicator, thermal throttle alert >90°C |
| GPU Temp | pynvml (NVML C bindings) | Colored indicator (hidden if no NVIDIA) |
| VRAM | pynvml | Used/Total MB bar |

- **2-second refresh interval**, all collection in `@work(thread=True)` — never blocks the event loop  
- **pynvml** instead of nvidia-smi subprocess — zero CPU overhead, Jetson ARM compatible  
- **Network I/O delta math** — cumulative `net_io_counters()` → `(new - old) / dt` = MB/s  
- **GPU graceful** — `nvmlInit()` wrapped in try/except, panel auto-hides on non-NVIDIA

### Other Features
| Feature | Description |
|---------|-------------|
| **Morning briefing** | Parallel: calendar + classroom + gmail + weather + schedule + overnight cache |
| **Daily/weekly digest** | Gemini Flash synthesizes raw usage data into natural language summaries |
| **Phone GPS** | Real-time location via LAN HTTP (:9777) or ntfy.sh relay for cross-network |
| **Named places** | Save locations with geofence detection and stationary alerts |
| **Proactive butler** | Context-aware greeting — knows how long you've been away, adjusts tone |
| **Habit engine** | Mines usage patterns by time segment (morning/afternoon/evening/night) |
| **Telegram bot** | Access from phone — `/briefing`, `/mail`, `/weather`, `/reminders`. Async progress indicators, MarkdownV2 rendering, message editing for live updates, terminal-parity quality |
| **Bonding meter** | RL-driven progressive formality — the butler warms up to you over time |
| **Plan-and-Solve** | Complex requests auto-decomposed into numbered steps with inter-step context passing |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                              │
│   ┌──────────────┐  ┌───────────┐  ┌───────────────────────────┐    │
│   │ TUI App      │  │ Telegram  │  │ CLI (--once, --daemon,    │    │
│   │ (Textual)    │  │ Bot (async│  │  --doctor, --setup)       │    │
│   │ + Telemetry  │  │ progress) │  │                           │    │
│   └──────┬───────┘  └─────┬─────┘  └─────────────┬─────────────┘    │
├──────────┴─────────────────┴──────────────────────┴──────────────────┤
│                          CORE LAYER                                  │
│   ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
│   │ Brain  │  │ Router │  │ Butler  │  │ Planner  │  │ Workflow │  │
│   │ (LLM   │  │ (Tool  │  │ (Greet  │  │ (Plan &  │  │ (multi-  │  │
│   │ orch.) │  │ select)│  │ & mood) │  │  Solve)  │  │  tool)   │  │
│   └────┬───┘  └────┬───┘  └─────────┘  └──────────┘  └──────────┘  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│   │ Executor │  │ Session  │  │ Bonding  │                         │
│   │ (step    │  │ Tracker  │  │ Meter    │                         │
│   │  runner) │  │          │  │ (RL)     │                         │
│   └──────────┘  └──────────┘  └──────────┘                         │
├──────────────────────────────────────────────────────────────────────┤
│                        AGENT LAYER                                   │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │ Observer │  │ RL      │  │ Interventions│  │ Job Scheduler  │   │
│   │ (stderr) │  │ Engine  │  │ (queue +     │  │ (APScheduler)  │   │
│   │          │  │(Q-learn)│  │  rate limit) │  │                │   │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘   │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │ App      │  │ TTS     │  │ Maintenance  │  │ Reflection     │   │
│   │ Detector │  │ (Piper) │  │ (3 AM)       │  │ (11 PM)        │   │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘   │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │ Wake     │  │ Ambient │  │ Health       │  │ Audio Ducker   │   │
│   │ Word     │  │ Audio   │  │ Monitor      │  │ (volume ctrl)  │   │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                         DATA LAYER                                   │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│   │ Memory       │  │ Vector Store  │  │ Data Access Layer       │  │
│   │ (SQLite+FTS5)│  │ (embeddings)  │  │ (store.py — unified     │  │
│   │              │  │               │  │  ABCs for all storage)   │  │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│   │ Graph Memory │  │ Distiller     │  │ Spatial Cache           │  │
│   │ (Neo4j)      │  │ (LLM summary  │  │ (UI element coords,    │  │
│   │              │  │  → vectors)   │  │  24h TTL, SQLite)       │  │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                        TOOLS LAYER (18 tools)                        │
│  ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────────┐ ┌───────────────┐  │
│  │ Shell │ │ Gmail │ │ Calendar │ │ Classroom │ │ Accessibility │  │
│  └───────┘ └───────┘ └──────────┘ └───────────┘ └───────────────┘  │
│  ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────────┐ ┌───────────────┐  │
│  │ Files │ │Weather│ │Web Search│ │ Reminder  │ │ Input Control │  │
│  └───────┘ └───────┘ └──────────┘ └───────────┘ └───────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                        VISION LAYER                                  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│   │ AT-SPI       │  │ Remote VLM    │  │ Navigator              │  │
│   │ (<10ms)      │  │ (Jetson/Colab)│  │ (unified fallback)     │  │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                          LLM LAYER                                   │
│   ┌──────────────┐         ┌───────────────────┐                    │
│   │ Ollama       │         │ Gemini            │                    │
│   │ (local, main)│         │ (cloud, fallback) │                    │
│   └──────────────┘         └───────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** running locally (`ollama pull qwen3:8b`)
- **Neo4j** (optional, for knowledge graph)

### Installation

```bash
git clone git@github.com:miclaldogan/bantzv2.git
cd bantzv2
bash setup.sh
```

`setup.sh` creates a venv, installs dependencies, and sets up `.env`.

### Optional Extras

```bash
pip install 'bantz[translation]'   # MarianMT TR↔EN translation
pip install 'bantz[docs]'          # PDF, DOCX reader
pip install 'bantz[graph]'         # Neo4j graph memory
pip install 'bantz[automation]'    # PyAutoGUI + pynput (OS control)
```

### Google Services Setup

```bash
bantz --setup google gmail
bantz --setup google calendar
bantz --setup google classroom
```

### Other Setup

```bash
bantz --setup profile    # Name, preferences
bantz --setup schedule   # Weekly timetable
bantz --setup places     # Named locations
bantz --setup telegram   # Telegram bot token
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `bantz` | Launch the Textual TUI |
| `bantz --once "question"` | Quick one-shot query, no UI |
| `bantz --doctor` | Health check all services |
| `bantz --daemon` | Background daemon (Telegram + reminders + jobs) |
| `bantz --setup <thing>` | Interactive setup wizards |
| `bantz --cache-stats` | Show spatial cache statistics |

You can also run with `python -m bantz`.

---

## Configuration

Key settings in `.env`:

```env
# ── LLM ──────────────────────────────────────
BANTZ_OLLAMA_MODEL=qwen3:8b
BANTZ_OLLAMA_BASE_URL=http://localhost:11434
BANTZ_GEMINI_KEY=                    # optional cloud fallback
BANTZ_GEMINI_MODEL=gemini-2.0-flash

# ── Embeddings & Memory ─────────────────────
BANTZ_EMBEDDINGS_ENABLED=true
BANTZ_EMBEDDINGS_MODEL=nomic-embed-text
BANTZ_DISTILLATION_ENABLED=true

# ── Shell ────────────────────────────────────
BANTZ_SHELL_CONFIRM_DESTRUCTIVE=true
BANTZ_SHELL_TIMEOUT_SECONDS=30

# ── Location ─────────────────────────────────
BANTZ_LOCATION_CITY=Istanbul
BANTZ_TIMEZONE=Europe/Istanbul

# ── Neo4j (optional) ────────────────────────
BANTZ_NEO4J_ENABLED=false
BANTZ_NEO4J_URI=bolt://localhost:7687

# ── Telegram (optional) ─────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=

# ── Agent System ─────────────────────────────
BANTZ_OBSERVER_ENABLED=true
BANTZ_RL_ENGINE_ENABLED=true
BANTZ_JOB_SCHEDULER_ENABLED=true
BANTZ_TTS_ENABLED=true
BANTZ_TTS_MODEL=en_US-lessac-medium

# ── Vision (optional) ───────────────────────
BANTZ_VISION_ENABLED=false
BANTZ_VLM_ENDPOINT=http://localhost:8080
BANTZ_INPUT_CONTROL_ENABLED=false
```

See [src/bantz/config.py](src/bantz/config.py) for the full list (~60 settings).

---

## How Routing Works

1. **Quick route** — keyword matching for obvious patterns (weather, email, GPS). No LLM call.
2. **LLM router** — Ollama picks the right tool and args from the registry via CoT.
3. **Workflow engine** — detects multi-step commands, chains tool calls.
4. **Plan-and-Solve** — complex requests → LLM decomposes into numbered JSON steps → Executor runs sequentially with context passing.
5. **Fallback** — no tool match → conversational chat.

---

## GPS Tracking

Bantz receives real-time GPS from your phone:

1. **Same network:** Phone opens `http://<laptop-ip>:9777`
2. **Any network:** Phone uses ntfy.sh relay for cross-network GPS

Data used for weather auto-detection, geofencing, place-based reminders, and stationary alerts.

---

## Project Structure

```
src/bantz/                        # ~34,800 LOC across 104 modules
├── __main__.py                   # CLI entry point (--once, --daemon, --doctor, --setup)
├── app.py                        # Textual main app (alternate entry)
├── config.py                     # ~60 settings from .env (Pydantic Settings)
│
├── core/                         # Brain, routing, memory, briefing, habits
│   ├── brain.py                  # Main orchestrator (2103 LOC)
│   ├── briefing.py               # Daily briefing (parallel API calls)
│   ├── butler.py                 # Context-aware proactive greeting
│   ├── digest.py                 # Daily/weekly digest via Gemini
│   ├── habits.py                 # Usage pattern mining by time segment
│   ├── router.py                 # One-shot tool routing classifier
│   ├── workflow.py               # Multi-tool chain execution
│   ├── gps_server.py             # Phone GPS receiver (LAN + ntfy.sh relay)
│   └── ...                       # session, schedule, places, time_parser, etc.
│
├── data/                         # Unified Data Access Layer (#115-#117)
│   ├── store.py                  # Abstract base classes (7 store contracts)
│   ├── models.py                 # Pydantic v2 models (Message, Reminder, Place, etc.)
│   ├── layer.py                  # Singleton DataLayer — composes all stores
│   ├── sqlite_store.py           # Concrete SQLite implementations
│   └── migration.py              # JSON → SQLite migration
│
├── memory/                       # Semantic & Graph Memory (#116-#118)
│   ├── vector_store.py           # Pure SQLite vector store (cosine similarity)
│   ├── embeddings.py             # Ollama nomic-embed-text (768-dim)
│   ├── distiller.py              # Session distillation (LLM summary → vector)
│   ├── graph.py                  # Neo4j knowledge graph
│   ├── nodes.py                  # Entity schema (9 node types, 8 relationships)
│   └── context_builder.py        # Graph → LLM context injection
│
├── agent/                        # Background Agent System (#124-#187)
│   ├── observer.py               # Stderr observer with error classification
│   ├── rl_engine.py              # Q-learning RL (1680 states, SQLite Q-table)
│   ├── interventions.py          # Priority queue + rate limiting + focus mode
│   ├── app_detector.py           # Active window → activity category
│   ├── job_scheduler.py          # APScheduler (cron + interval jobs)
│   ├── tts.py                    # Piper + aplay streaming TTS
│   ├── notifier.py               # Desktop notifications (notify-send)
│   ├── wake_word.py              # Offline Porcupine wake word detection (#165)
│   ├── ambient.py                # Periodic ambient audio sampling (#166)
│   ├── proactive.py              # Idle detection → proactive engagement (#167)
│   ├── health.py                 # Telemetry-driven health interventions (#168)
│   ├── audio_ducker.py           # System audio ducking during TTS (#171)
│   ├── planner.py                # Plan-and-Solve decomposition (#187)
│   ├── executor.py               # Sequential plan step runner (#187)
│   └── workflows/
│       ├── maintenance.py        # 3 AM: Docker, temp, disk, services, logs
│       ├── reflection.py         # 11 PM: Hierarchical summarization
│       └── overnight_poll.py     # 2h: Gmail/Calendar/Classroom polling
│
├── personality/                  # Persona & Bonding (#169, #172)
│   ├── system_prompt.py          # Dynamic system prompt generation
│   ├── greeting.py               # Butler greeting templates
│   ├── persona.py                # LLM persona adaptation (#169)
│   └── bonding.py                # RL bonding meter — progressive formality (#172)
│
├── vision/                       # OS Control & Screen Reading (#119-#123)
│   ├── navigator.py              # Unified fallback chain with per-app analytics
│   ├── screenshot.py             # Screen capture (gnome-screenshot/scrot/Pillow)
│   ├── remote_vlm.py             # REST client for VLM (Jetson/Colab/local)
│   └── spatial_cache.py          # UI element coordinate cache (SQLite, 24h TTL)
│
├── tools/                        # 18 tool modules
│   ├── shell.py                  # Bash with security controls
│   ├── gmail.py                  # Full Gmail integration + auto-chain compose/send
│   ├── calendar.py               # Google Calendar CRUD
│   ├── classroom.py              # Google Classroom
│   ├── filesystem.py             # File ops + LLM auto-chain create-on-miss
│   ├── accessibility.py          # AT-SPI2 accessibility reader
│   ├── input_control.py          # Mouse/keyboard simulation
│   └── ...                       # weather, news, docs, reminders, web_search, etc.
│
├── interface/
│   ├── telegram_bot.py           # Telegram integration (async progress, MarkdownV2)
│   └── tui/
│       ├── app.py                # BantzApp (Textual, 769 LOC)
│       ├── telemetry.py          # Hardware telemetry collector (#133)
│       ├── styles.tcss            # Dark green terminal theme
│       └── panels/
│           ├── system.py          # Real-time sparkline telemetry panel
│           └── chat.py            # Chat log with streaming
│
├── llm/                          # LLM clients
│   ├── ollama.py                 # Ollama API (streaming, embeddings)
│   └── gemini.py                 # Gemini API (fallback)
│
├── auth/                         # Google OAuth
└── i18n/                         # TR↔EN translation bridge
```

---

## Test Suite

**46 test files — 2261 tests — ~25,700 LOC of test code**

```bash
# Run all tests
pip install -e ".[dev]"
PYTHONPATH=src pytest --ignore=tests/agent/test_observer.py -q

# With coverage
PYTHONPATH=src pytest --cov=bantz --cov-report=html
```

Test breakdown by area:
| Area | Tests | Covers |
|------|-------|--------|
| Data layer | 111 | store ABCs, SQLite, models, JSON migration |
| Memory | 109 | vector store, distiller, embeddings |
| Vision | 187 | navigator, spatial cache, VLM, screenshot |
| Agent | 716 | observer, RL engine, interventions, planner, executor, TTS, health, bonding, wake word |
| Workflows | 160 | maintenance, reflection, overnight poll |
| Core | 425 | brain integrations, router, shell, stabilization, regex audit, RLHF |
| TUI | 325 | header, input, mood, streaming, telemetry, toast |
| Interface | 76 | Telegram LLM, async progress, MarkdownV2 |
| Personality | 103 | bonding meter, persona, progressive formality |
| Tools | 49 | filesystem auto-chain, Gmail auto-chain, web search |

---

## v3 Completed Issues

All 38 issues from Phase 1–7 have been implemented and merged:

### Phase 1: Data Layer Revolution & Semantic Memory
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #115 | Unified Data Access Layer — abstract `store.py` | ✅ | 44 |
| #116 | Vector DB for semantic cross-session memory | ✅ | 30 |
| #117 | Migrate JSON stores into unified DB schema | ✅ | 12 |
| #118 | Automatic session distillation to long-term memory | ✅ | 22 |

### Phase 2: Eyes & Hands — Hybrid OS Control
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #119 | AT-SPI accessibility reader (<10ms UI location) | ✅ | 52 |
| #120 | Remote VLM screenshot analysis (Jetson/Colab) | ✅ | 40 |
| #121 | Spatial memory — cache UI element coordinates | ✅ | 60 |
| #122 | PyAutoGUI/pynput input simulation | ✅ | 50 |
| #123 | Unified navigation pipeline — fallback chain | ✅ | 41 |

### Phase 3: Background Observer & Reinforcement Learning
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #124 | Background stderr observer — proactive error detection | ✅ | 35 |
| #125 | RL framework for routine optimization (Q-learning) | ✅ | 60 |
| #126 | Proactive intervention — non-intrusive suggestions | ✅ | 42 |
| #127 | Application state detector | ✅ | 46 |

### Phase 4: Autonomous Night Shift
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #128 | APScheduler for robust background job scheduling | ✅ | 72 |
| #129 | Autonomous system maintenance (3 AM) | ✅ | 45 |
| #130 | Nightly memory reflection (11 PM) | ✅ | 67 |
| #131 | Audio morning briefing with Piper TTS | ✅ | 62 |
| #132 | Overnight email/calendar polling | ✅ | 50 |

### Phase 5: Dynamic Host TUI
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #133 | Real-time hardware telemetry (psutil + pynvml) | ✅ | 43 |

### Phase 6: Senses & Personality
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #165 | Offline wake word detection ("Hey Bantz") via Porcupine | ✅ | — |
| #166 | Periodic ambient audio sampling for environment awareness | ✅ | — |
| #167 | Proactive engagement and idle conversation initiation | ✅ | — |
| #168 | Proactive break & health interventions from telemetry | ✅ | — |
| #169 | Dynamic LLM persona adaptation based on system state | ✅ | — |
| #170 | Spontaneous vector memory retrieval for conversational depth | ✅ | — |
| #171 | System audio ducking during TTS playback and wake word | ✅ | — |
| #172 | RL-based bonding meter for progressive formality | ✅ #179 | 103 |

### Phase 7: Communication Parity & Intelligence
| # | Issue | PR | Tests |
|---|-------|-----|-------|
| #180 | Direct RLHF via Sentiment & Feedback Keywords | ✅ #190 | RL |
| #181 | Telegram async progress indicators & message editing | ✅ #191 | 76 |
| #183 | Auto-chaining Gmail compose & send actions | ✅ #194 | — |
| #187 | Plan-and-Solve multi-step decomposition (planner + executor) | ✅ #198 | 39 |

### Bug Fixes & Stabilization (Phase 6–7)
| PR | Description |
|----|-------------|
| #192 | Identity, routing, and live streaming fixes |
| #193 | MarkdownV2 trap, bracket bug, URL dedup |
| #195 | Terminal Parity — Telegram same quality as TUI (serial lock, maintenance filter, Ollama warm-up) |
| #196–197 | Filesystem auto-chaining v1 (regex) → v2 (LLM-based param extraction) |
| #177 | Strict context guards for 5 `_quick_route` regex false positives |
| #173 | Systemd linger for true 24/7 background execution |

### Remaining Roadmap
| # | Issue | Status |
|---|-------|--------|
| #182 | Strict source citation for web search — Telegraph references | Planned |
| #184 | Context window loop breaker & background spam filter | Planned |
| #185 | Visual UI automation (Computer Use) — the butler's eyes | Planned |
| #186 | Demonstration learning (macro recording) — the butler's apprenticeship | Planned |
| #188 | Autonomous VLM web navigation loop | Planned |
| #189 | Remote visual operation & telemetry via Telegram | Planned |

---

## Dependencies

**Core (12 packages):**
`textual` · `httpx` · `aiosqlite` · `pydantic` · `pydantic-settings` · `python-dotenv` · `psutil` · `pynvml` · `rich` · `apscheduler` · `sqlalchemy`

**Optional extras:**
| Extra | Packages | Purpose |
|-------|----------|---------|
| `translation` | transformers, torch, sentencepiece | MarianMT TR↔EN |
| `docs` | pymupdf, python-docx | PDF/DOCX reading |
| `graph` | neo4j | Knowledge graph memory |
| `automation` | pyautogui, pynput | OS input control |
| `dev` | pytest, pytest-asyncio, ruff | Development |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Bantz** — *Not just an assistant. The host.*

Built with love by [@miclaldogan](https://github.com/miclaldogan)

</div>
