# Bantz v2

Local-first AI terminal assistant. Runs on your machine, talks to your shell, understands Turkish.

Built with [Ollama](https://ollama.com) + [Textual](https://textual.textualize.io) + [MarianMT](https://huggingface.co/Helsinki-NLP).

## What it does

You type in Turkish (or English). Bantz figures out what you need — runs a shell command, checks system stats, fetches weather, pulls news headlines — and responds in Turkish.

No cloud APIs for core functionality. LLM runs locally via Ollama. Translation runs locally via MarianMT.

## Tools

| Tool | What it does | Source |
|------|-------------|--------|
| **shell** | Runs bash commands with security controls | blocked/destructive command detection |
| **system** | CPU, RAM, disk, uptime | psutil |
| **filesystem** | Read/write/list files | path-restricted to home directory |
| **weather** | Current conditions + 3-day forecast | wttr.in (no API key) |
| **news** | Headlines from Hacker News + Google News | HN API + RSS, 15min cache |

## Architecture

```
User input (Turkish)
  → MarianMT bridge (TR → EN)
  → Quick route (keyword match) or LLM router (Ollama)
  → Tool execution
  → Finalizer (responds in Turkish, time-aware)
  → Textual TUI output
```

**Key modules:**

```
src/bantz/
├── __main__.py          # CLI entry point (--doctor, --once)
├── app.py               # Textual TUI
├── config.py            # Pydantic settings (.env)
├── core/
│   ├── brain.py         # Orchestrator — routing, LLM calls, tool dispatch
│   ├── location.py      # IP geolocation (ipinfo.io) with session cache
│   └── time_context.py  # Time-of-day awareness (greeting, prompt hints)
├── i18n/
│   └── bridge.py        # TR↔EN translation (MarianMT, local inference)
├── llm/
│   └── ollama.py        # Ollama HTTP client
└── tools/
    ├── __init__.py      # ToolResult, BaseTool, ToolRegistry
    ├── shell.py         # Bash execution
    ├── system.py        # System metrics
    ├── filesystem.py    # File operations
    ├── weather.py       # wttr.in weather
    └── news.py          # HN + Google News
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a model (e.g. `qwen2.5-coder:7b`)
- Linux (tested on Linux Mint 21.3)

## Setup

```bash
git clone git@github.com:miclaldogan/bantzv2.git
cd bantzv2
bash setup.sh
```

`setup.sh` creates a venv, installs dependencies, and copies `.env.example` to `.env`.

To install translation support (MarianMT):

```bash
source .venv/bin/activate
pip install transformers torch sentencepiece
```

## Usage

```bash
# Full TUI
python -m bantz

# Single command mode
python -m bantz --once "diskimi kontrol et"

# Health check
python -m bantz --doctor
```

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```env
BANTZ_OLLAMA_MODEL=qwen2.5-coder:7b
BANTZ_OLLAMA_BASE_URL=http://localhost:11434
BANTZ_SHELL_TIMEOUT=30
BANTZ_SHELL_CONFIRM_DESTRUCTIVE=true
BANTZ_BRIDGE_ENABLED=true
```

## How routing works

1. **Quick route** — keyword matching for obvious patterns (disk, ram, weather, news, file creation). No LLM call needed.
2. **LLM router** — for ambiguous requests, Ollama picks the right tool and args.
3. **Command generator** — for write/create operations, LLM generates a bash command from both the Turkish original and English translation (prevents content loss from translation).

## License

MIT
