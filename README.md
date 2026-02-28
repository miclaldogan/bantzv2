# Bantz v2

Local-first AI assistant that lives in your terminal. Runs on your machine, talks to your shell, manages your email, tracks your classes, knows where you are.

Built with [Ollama](https://ollama.com) + [Textual](https://textual.textualize.io) — no cloud APIs for core functionality.

## What it does

You type in English. Bantz figures out what you need — runs shell commands, checks your email, manages your calendar, fetches weather, reads documents, tracks your GPS, gives you a morning briefing — and responds naturally. Everything runs locally via Ollama. No data leaves your machine unless you explicitly use Google services (Gmail, Calendar, Classroom).

## Features at a glance

| Feature | Description |
|---------|-------------|
| **Shell execution** | Runs bash commands with security controls — blocked commands, destructive ops require confirmation |
| **Gmail** | Read, search, filter, compose, send, reply. Natural language queries: "unread emails from professor", "starred emails" |
| **Google Calendar** | View today/week, create/update/delete events |
| **Google Classroom** | Assignments, due dates, course announcements |
| **Weather** | Current conditions + 3-day forecast via wttr.in (no API key) |
| **News** | Hacker News + Google News headlines with LLM summarization |
| **System monitor** | CPU, RAM, disk, uptime — always visible in the sidebar |
| **File operations** | Read, write, list files — sandboxed to home directory |
| **Documents** | Read and summarize PDF, DOCX, TXT, MD files. Ask questions about content |
| **Phone GPS** | Real-time location from your phone via LAN or ntfy.sh relay — works across any network |
| **Named places** | Save locations ("dorm", "campus"), 100m geofence detection, stationary alerts |
| **Morning briefing** | One command: schedule + weather + mail + calendar + assignments |
| **Class schedule** | Weekly timetable with next-class countdown |
| **Conversation memory** | SQLite-backed persistent history across sessions, full-text search |
| **Graph memory** | Optional Neo4j knowledge graph — people, topics, decisions, tasks |
| **Multi-tool workflows** | Chain commands: "send email to prof, add it to calendar, remind me tomorrow" |
| **Telegram bot** | Access tools from your phone — `/briefing`, `/mail`, `/weather` |
| **Proactive butler** | Context-aware greeting on launch — knows how long you've been away, what's pending |

## Architecture

```
User input
  → Quick route (keyword match) or LLM router (Ollama)
  → Tool execution
  → Finalizer (context-aware natural response)
  → Textual TUI output
```

```
src/bantz/
├── __main__.py              # CLI: --doctor, --once, --setup
├── app.py                   # Textual TUI with chat + system sidebar
├── config.py                # Pydantic settings (.env)
├── auth/
│   ├── google_oauth.py      # Browser-based OAuth2 for Google services
│   └── token_store.py       # Per-service token persistence
├── core/
│   ├── brain.py             # Orchestrator — routing, tool dispatch, memory
│   ├── briefing.py          # Parallel morning briefing generator
│   ├── butler.py            # Proactive greeting with live data
│   ├── date_parser.py       # Natural language date resolution
│   ├── finalizer.py         # LLM response formatting
│   ├── gps_server.py        # Phone GPS receiver (LAN + ntfy.sh relay)
│   ├── habits.py            # Usage pattern mining for suggestions
│   ├── location.py          # Multi-source location chain
│   ├── memory.py            # SQLite conversation memory + FTS5 search
│   ├── places.py            # Named locations, geofencing, stationary detection
│   ├── profile.py           # User identity and preferences
│   ├── router.py            # Ollama-based intent routing
│   ├── schedule.py          # University class timetable
│   ├── session.py           # Launch tracking for absence-aware greetings
│   ├── time_context.py      # Time-of-day awareness
│   └── workflow.py          # Multi-tool command chaining
├── i18n/
│   └── bridge.py            # TR↔EN translation (MarianMT, optional)
├── integrations/
│   └── telegram_bot.py      # Telegram bot for phone access
├── llm/
│   └── ollama.py            # Ollama HTTP client
├── memory/
│   └── graph.py             # Neo4j knowledge graph (optional)
└── tools/
    ├── calendar.py          # Google Calendar CRUD
    ├── classroom.py         # Google Classroom read
    ├── document.py          # PDF/DOCX/TXT reader + summarizer
    ├── filesystem.py        # Sandboxed file operations
    ├── gmail.py             # Full Gmail client
    ├── news.py              # HN + Google News
    ├── shell.py             # Bash execution with safety
    ├── system.py            # System metrics via psutil
    └── weather.py           # wttr.in weather
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a model (default: `qwen2.5:7b`)
- Linux (tested on Linux Mint 21.3)

## Setup

```bash
git clone git@github.com:miclaldogan/bantzv2.git
cd bantzv2
bash setup.sh
```

`setup.sh` creates a venv, installs dependencies, and sets up `.env`.

### Optional extras

```bash
# Translation support (MarianMT — TR↔EN)
pip install 'bantz[translation]'

# Document reading (PDF, DOCX)
pip install 'bantz[docs]'

# Graph memory (Neo4j)
pip install 'bantz[graph]'
```

### Google services setup

```bash
bantz --setup google gmail       # Gmail access
bantz --setup google calendar    # Google Calendar
bantz --setup google classroom   # Google Classroom (can use a different account)
```

### Other setup

```bash
bantz --setup profile    # Name, university, preferences
bantz --setup schedule   # Weekly class timetable
bantz --setup places     # Named locations (home, campus, etc.)
bantz --setup telegram   # Telegram bot token
```

## Usage

```bash
# Full TUI
bantz

# Single command mode
bantz --once "how many unread emails do I have"

# Health check
bantz --doctor
```

## GPS tracking

Bantz can receive real-time GPS from your phone:

1. **Same network:** Phone opens `http://<laptop-ip>:9777` → auto-starts tracking
2. **Any network (relay):** Phone downloads the tracker app at `/app` → sends GPS via [ntfy.sh](https://ntfy.sh) relay

The standalone phone tracker runs in your browser with continuous `watchPosition`, wake lock, and 60-second send intervals. GPS data is used for weather auto-detection, named place geofencing, and stationary alerts.

## Configuration

Key settings in `.env`:

```env
BANTZ_OLLAMA_MODEL=qwen2.5:7b
BANTZ_OLLAMA_BASE_URL=http://localhost:11434
BANTZ_SHELL_CONFIRM_DESTRUCTIVE=true
BANTZ_SHELL_TIMEOUT_SECONDS=30

# Optional: Neo4j graph memory
BANTZ_NEO4J_ENABLED=false
BANTZ_NEO4J_URI=bolt://localhost:7687

# Optional: Telegram bot
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=
```

See `config.py` for the full list of options.

## How routing works

1. **Quick route** — keyword matching for obvious patterns (weather, email, news, GPS, places). No LLM call needed.
2. **LLM router** — for ambiguous requests, Ollama picks the right tool and args from the tool registry.
3. **Workflow engine** — detects multi-step commands and chains tool calls sequentially.
4. **Fallback** — if no tool matches, falls through to conversational chat.

## License

MIT
