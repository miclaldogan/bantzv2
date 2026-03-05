<div align="center">

# BANTZ

**Your AI-Powered Personal Host — Terminal Assistant, System Observer, Autonomous Agent**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)
[![Textual](https://img.shields.io/badge/TUI-Textual-purple.svg)](https://textual.textualize.io)

*Bantz isn't just an assistant — it's the host of your machine.*

</div>

---

## What is Bantz?

Bantz is a **local-first, privacy-respecting AI assistant** that lives in your terminal. It combines:

- **Conversational AI** via Ollama (local LLMs) + Gemini (cloud fallback)
- **Persistent Memory** with SQLite, Neo4j knowledge graphs, and semantic vector search
- **Tool Execution** — shell commands, Gmail, Google Calendar, Classroom, file operations
- **Smart Scheduling** — reminders, habit tracking, proactive suggestions
- **Location Awareness** — GPS relay from phone, place-based context
- **Rich TUI** built with Textual — chat, system panels, and live indicators
- **Fully Local** — your data stays on your machine

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **Shell execution** | Bash commands with security controls — blocked commands, destructive ops require confirmation |
| **Gmail** | Read, search, filter, compose, send, reply. NL queries: "unread emails from professor" |
| **Google Calendar** | View today/week, create/update/delete events |
| **Google Classroom** | Assignments, due dates, course announcements |
| **Weather** | Current conditions + 3-day forecast via wttr.in (no API key) |
| **News** | Hacker News + Google News headlines with LLM summarization |
| **System monitor** | CPU, RAM, disk, uptime in the sidebar |
| **File operations** | Read, write, list — sandboxed to home directory |
| **Documents** | Read and summarize PDF, DOCX, TXT, MD files |
| **Phone GPS** | Real-time location from phone via LAN or ntfy.sh relay |
| **Named places** | Save locations with geofence detection and stationary alerts |
| **Morning briefing** | Schedule + weather + mail + calendar + assignments |
| **Conversation memory** | SQLite-backed persistent history with full-text search |
| **Graph memory** | Optional Neo4j knowledge graph — people, topics, decisions |
| **Multi-tool workflows** | Chain: "send email to prof, add to calendar, remind me tomorrow" |
| **Telegram bot** | Access from phone — `/briefing`, `/mail`, `/weather` |
| **Proactive butler** | Context-aware greeting — knows how long you've been away |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                       │
│   ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
│   │ TUI App  │  │ Telegram  │  │ CLI (--doctor,       │  │
│   │ (Textual)│  │   Bot     │  │  --ask, --daemon)    │  │
│   └────┬─────┘  └─────┬─────┘  └───────────┬──────────┘  │
├────────┴───────────────┴────────────────────┴─────────────┤
│                      CORE LAYER                           │
│   ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────────┐  │
│   │ Brain  │  │ Router │  │ Butler  │  │ Session      │  │
│   │ (LLM   │  │ (Tool  │  │ (Greet  │  │ Tracker      │  │
│   │ logic) │  │ select)│  │ & mood) │  │              │  │
│   └────┬───┘  └────┬───┘  └─────────┘  └──────────────┘  │
├────────┴────────────┴─────────────────────────────────────┤
│                      DATA LAYER                           │
│   ┌──────────┐  ┌───────────┐  ┌────────────────────────┐ │
│   │ Memory   │  │ Graph     │  │ Data Access Layer      │ │
│   │ (SQLite  │  │ Memory    │  │ (store.py — unified    │ │
│   │  + FTS5) │  │ (Neo4j)   │  │  gateway for all DB)   │ │
│   └──────────┘  └───────────┘  └────────────────────────┘ │
├───────────────────────────────────────────────────────────┤
│                     TOOLS LAYER                           │
│  ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ Shell │ │ Gmail │ │ Calendar │ │    Classroom      │  │
│  └───────┘ └───────┘ └──────────┘ └───────────────────┘  │
│  ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────────────────┐  │
│  │ Files │ │Weather│ │Web Search│ │    Reminder       │  │
│  └───────┘ └───────┘ └──────────┘ └───────────────────┘  │
├───────────────────────────────────────────────────────────┤
│                      LLM LAYER                            │
│   ┌──────────────┐         ┌───────────────────┐          │
│   │ Ollama       │         │ Gemini            │          │
│   │ (local, main)│         │ (cloud, fallback) │          │
│   └──────────────┘         └───────────────────┘          │
└───────────────────────────────────────────────────────────┘
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
pip install 'bantz[translation]'  # MarianMT TR↔EN
pip install 'bantz[docs]'         # PDF, DOCX reader
pip install 'bantz[graph]'        # Neo4j graph memory
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
| `bantz --once "question"` | Quick one-shot |
| `bantz --doctor` | Health check all services |
| `bantz --daemon` | Background daemon (Telegram + reminders) |
| `bantz --setup <thing>` | Interactive setup wizards |

---

## Configuration

Key settings in `.env`:

```env
BANTZ_OLLAMA_MODEL=qwen3:8b
BANTZ_OLLAMA_BASE_URL=http://localhost:11434
BANTZ_SHELL_CONFIRM_DESTRUCTIVE=true
BANTZ_SHELL_TIMEOUT_SECONDS=30

# Optional: Neo4j graph memory
BANTZ_NEO4J_ENABLED=false
BANTZ_NEO4J_URI=bolt://localhost:7687

# Optional: Telegram bot
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=

# Optional: Gemini cloud fallback
BANTZ_GEMINI_KEY=
```

See `config.py` for the full list.

---

## How Routing Works

1. **Quick route** — keyword matching for obvious patterns (weather, email, GPS). No LLM call.
2. **LLM router** — Ollama picks the right tool and args from the registry.
3. **Workflow engine** — detects multi-step commands, chains tool calls.
4. **Fallback** — no tool match → conversational chat.

---

## GPS Tracking

Bantz receives real-time GPS from your phone:

1. **Same network:** Phone opens `http://<laptop-ip>:9777`
2. **Any network:** Phone uses ntfy.sh relay for cross-network GPS

Data used for weather auto-detection, geofencing, and stationary alerts.

---

## v3 Roadmap — The Master Plan

> Bantz v3 transforms from a terminal assistant into the **autonomous host** of your machine.

5 phases, 25 issues, building from data foundation to sentient-feeling TUI:

### Phase 1: Data Layer Revolution & Semantic Memory

*Unify all storage and add meaning-based recall*

| # | Issue | Priority |
|---|-------|----------|
| [#115](https://github.com/miclaldogan/bantzv2/issues/115) | Unified Data Access Layer — abstract `store.py` | Critical |
| [#116](https://github.com/miclaldogan/bantzv2/issues/116) | Vector DB for semantic cross-trial memory | High |
| [#117](https://github.com/miclaldogan/bantzv2/issues/117) | Migrate JSON stores into unified DB schema | Medium |
| [#118](https://github.com/miclaldogan/bantzv2/issues/118) | Automatic session distillation to long-term memory | Medium |

**Goal:** All data flows through a single DAL. Bantz remembers by *meaning*, not just keywords.

---

### Phase 2: Eyes & Hands — Hybrid OS Control

*Give Bantz the ability to see and interact with the GUI*

| # | Issue | Priority |
|---|-------|----------|
| [#119](https://github.com/miclaldogan/bantzv2/issues/119) | AT-SPI accessibility reader for instant UI location | High |
| [#120](https://github.com/miclaldogan/bantzv2/issues/120) | Remote VLM screenshot analysis (Jetson/Colab) | Medium |
| [#121](https://github.com/miclaldogan/bantzv2/issues/121) | Spatial memory — cache UI element coordinates | Medium |
| [#122](https://github.com/miclaldogan/bantzv2/issues/122) | PyAutoGUI/pynput input simulation (the hands) | Medium |
| [#123](https://github.com/miclaldogan/bantzv2/issues/123) | Unified navigation pipeline — fallback chain | Medium |

**Pipeline:** `AT-SPI (< 10ms) → Cache (< 5ms) → Remote VLM (2-3s) → Ask User`

---

### Phase 3: Background Observer & Reinforcement Learning

*Bantz learns your habits and catches errors before you notice*

| # | Issue | Priority |
|---|-------|----------|
| [#124](https://github.com/miclaldogan/bantzv2/issues/124) | Background stderr observer — proactive error detection | High |
| [#125](https://github.com/miclaldogan/bantzv2/issues/125) | RL framework for routine optimization (Q-learning) | Medium |
| [#126](https://github.com/miclaldogan/bantzv2/issues/126) | Proactive intervention — non-intrusive suggestions | Medium |
| [#127](https://github.com/miclaldogan/bantzv2/issues/127) | Application state detector (active apps/workspace) | Medium |

**RL Loop:** `State (time, location, apps) → Q-table → Action → User response → Reward → Learn`

---

### Phase 4: Autonomous Night Shift

*Bantz works while you sleep*

| # | Issue | Priority |
|---|-------|----------|
| [#128](https://github.com/miclaldogan/bantzv2/issues/128) | APScheduler for robust background job scheduling | High |
| [#129](https://github.com/miclaldogan/bantzv2/issues/129) | Autonomous system maintenance (Docker prune, cleanup) | Medium |
| [#130](https://github.com/miclaldogan/bantzv2/issues/130) | Nightly memory reflection — compress, summarize, learn | Medium |
| [#131](https://github.com/miclaldogan/bantzv2/issues/131) | Audio morning briefing with local TTS | Medium |
| [#132](https://github.com/miclaldogan/bantzv2/issues/132) | Overnight email/calendar polling | Medium |

**Night Schedule:**
```
11 PM → Memory Reflection (summarize day's conversations)
 3 AM → System Maintenance (Docker prune, disk cleanup)
 2h   → Email/Calendar polls
 7 AM → Audio Morning Briefing on first unlock
```

---

### Phase 5: Dynamic Host TUI

*The interface that breathes*

| # | Issue | Priority |
|---|-------|----------|
| [#133](https://github.com/miclaldogan/bantzv2/issues/133) | Real-time hardware telemetry with psutil | Medium |
| [#134](https://github.com/miclaldogan/bantzv2/issues/134) | ASCII sparkline charts in TUI sidebar | Medium |
| [#135](https://github.com/miclaldogan/bantzv2/issues/135) | Dynamic mood system — UI theme reacts to load | Medium |
| [#136](https://github.com/miclaldogan/bantzv2/issues/136) | Live header with service status indicators | Medium |
| [#137](https://github.com/miclaldogan/bantzv2/issues/137) | Non-blocking toast notification system | Medium |

**Mood States:**

| State | When | Border | Face |
|-------|------|--------|------|
| Chill | CPU < 20%, idle | Green | `(◕‿◕)` |
| Focused | Tool running | Blue | `(•̀ᴗ•́)` |
| Busy | CPU 50-80% | Orange | `(⊙_⊙)` |
| Stressed | CPU > 80% or errors | Red | `(╥﹏╥)` |
| Sleeping | Night, no input | Purple | `(-.-)zzz` |

---

### Cross-cutting Infrastructure

| # | Issue |
|---|-------|
| [#138](https://github.com/miclaldogan/bantzv2/issues/138) | Test infrastructure for all v3 components |
| [#139](https://github.com/miclaldogan/bantzv2/issues/139) | Extend Config for vision, RL, TTS, APScheduler |

---

### Dependency Graph

```
Phase 1 ─── DAL ──→ Vector DB ──→ Distillation
  │                     │
  │          ┌──────────┘
  ▼          ▼
Phase 2 ─── Spatial Cache ──→ AT-SPI ──→ VLM ──→ Navigator
  │                                                   │
  ▼                                                   │
Phase 3 ─── Observer + App Detector ──→ RL Engine ◄───┘
  │                                        │
  ▼                                        ▼
Phase 4 ─── APScheduler ──→ Maintenance + Reflection ──→ Briefing + TTS
  │
  ▼
Phase 5 ─── Telemetry ──→ Sparklines + Mood ──→ Toast + Header
```

---

## Project Structure

```
src/bantz/
├── __main__.py              # CLI entry point
├── app.py                   # Textual main app
├── config.py                # Settings (.env)
├── auth/                    # Google OAuth
├── core/                    # Brain, Router, Memory, Butler, Habits, Places, Schedule
├── data/                    # Data Access Layer (v3: store.py, models.py)
├── memory/                  # Neo4j graph + entity extraction
├── llm/                     # Ollama + Gemini clients
├── tools/                   # Shell, Gmail, Calendar, Classroom, Weather, Files, etc.
├── interface/               # Textual TUI + Telegram bot
├── i18n/                    # TR↔EN translation bridge
├── agent/                   # [v3] Observer, RL, Interventions, TTS, Workflows
└── vision/                  # [v3] Navigator, Screenshot, Remote VLM
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
pytest --cov=bantz --cov-report=html
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Bantz** — *Not just an assistant. The host.*

Built with love by [@miclaldogan](https://github.com/miclaldogan)

</div>
