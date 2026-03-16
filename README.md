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

**Your AI-Powered Personal Host — Terminal Assistant, System Observer, Autonomous Agent**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests: 2261](https://img.shields.io/badge/tests-2261-brightgreen.svg)](#test-suite)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)
[![Textual](https://img.shields.io/badge/TUI-Textual-purple.svg)](https://textual.textualize.io)

*Bantz isn't just an assistant — it's the meticulous host of your machine.*

</div>

---

## 🎩 What is Bantz?

Bantz started as a script, grew into a router, and has now evolved into a **local-first, pro-active AGI-Lite Operating System** living directly inside your terminal. It watches your system telemetry, reads your screen when needed, manages your calendar, sends your emails, and proactively suggests taking a break when your RAM (and your brain) is overheating.

**Key principles:**
- **Zero-Cloud Dependency (Local First):** Powered by Ollama (`qwen3:8b`/`llama3`), SQLite, and local Graph memory. Your data stays in your house.
- **The Butler Persona:** Polite, discreet, unbothered by chaos, and progressively formal—using Reinforcement Learning (RLHF) to adapt to your mood and feedback over time.
- **Autonomous Agent Layer:** Operates completely in the background via Systemd. Runs nightly reflections, clears cache, listens to the microphone for context, and checks for wake words offline.

---

## 🧠 System 2 "Scratchpad" Reasoning

Bantz doesn't just act blindly. Before making any decisions or firing tool outputs, it opens a strictly guarded quarantine room to internally extract entities, double-check logic, and prevent hallucinations.

```text
┌─────────────────────────┐
│ 🗣️ USER REQUEST        │
│ "Check the PDF and  "   │
│ "email the summary."    │
└───────────┬─────────────┘
            │
            ▼
╔═════════════════════════╗
║ 💭 THE SCRATCHPAD       ║
║ <thinking>              ║
║  1. Extract: PDF path?  ║
║     Email address?      ║
║  2. Tools: filesystem + ║
║     process_text + gmail║
║  3. Audit: Do NOT fake  ║
║     text! Always read!  ║
║ </thinking>             ║
╚═══════════╦═════════════╝
            │
            ▼
┌───────────┴─────────────┐
│ 🎯 ACTION (Strict JSON) │
│  [web_search, read_url, │
│   process_text, gmail]  │
└─────────────────────────┘
```

---

## 🏛️ System Architecture

Bantz is built on a highly decoupled, layered architecture.

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                              │
│   ┌──────────────┐  ┌───────────┐  ┌───────────────────────────┐     │
│   │ TUI App      │  │ Telegram  │  │ CLI (--once, --daemon,    │     │
│   │ (Textual)    │  │ Bot (async│  │  --doctor, --setup)       │     │
│   │ + Telemetry  │  │ progress) │  │                           │     │
│   └──────┬───────┘  └─────┬─────┘  └─────────────┬─────────────┘     │
├──────────┴─────────────────┴──────────────────────┴──────────────────┤
│                          CORE LAYER                                  │
│   ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐    │
│   │ Brain  │  │ Router │  │ Butler  │  │ Planner  │  │ Workflow │    │
│   │ (LLM   │  │ (Tool  │  │ (Greet  │  │ (Plan &  │  │ (multi-  │    │
│   │ orch.) │  │ select)│  │ & mood) │  │  Solve)  │  │  tool)   │    │
│   └────┬───┘  └────┬───┘  └─────────┘  └──────────┘  └──────────┘    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                           │
│   │ Executor │  │ Session  │  │ Bonding  │                           │
│   │ (step    │  │ Tracker  │  │ Meter    │                           │
│   │  runner) │  │          │  │ (RL)     │                           │
│   └──────────┘  └──────────┘  └──────────┘                           │
├──────────────────────────────────────────────────────────────────────┤
│                        AGENT LAYER                                   │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐    │
│   │ Observer │  │ RL      │  │ Interventions│  │ Job Scheduler  │    │
│   │ (stderr) │  │ Engine  │  │ (queue +     │  │ (APScheduler)  │    │
│   │          │  │(Q-learn)│  │  rate limit) │  │                │    │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘    │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐    │
│   │ App      │  │ TTS     │  │ Maintenance  │  │ Reflection     │    │
│   │ Detector │  │ (Piper) │  │ (3 AM)       │  │ (11 PM)        │    │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘    │
│   ┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌────────────────┐    │
│   │ Wake     │  │ Ambient │  │ Health       │  │ Audio Ducker   │    │
│   │ Word     │  │ Audio   │  │ Monitor      │  │ (volume ctrl)  │    │
│   └──────────┘  └─────────┘  └──────────────┘  └────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                          DATA LAYER                                  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐   │
│   │ Memory       │  │ Vector Store  │  │ Data Access Layer       │   │
│   │ (SQLite+FTS5)│  │ (embeddings)  │  │ (store.py — unified     │   │
│   │              │  │               │  │  ABCs for all storage)  │   │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘   │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐   │
│   │ Graph Memory │  │ Distiller     │  │ Spatial Cache           │   │
│   │ (Neo4j)      │  │ (LLM summary  │  │ (UI element coords,     │   │
│   │              │  │  → vectors)   │  │  24h TTL, SQLite)       │   │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                        TOOLS LAYER (18 tools)                        │
│  ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────────┐ ┌───────────────┐    │
│  │ Shell │ │ Gmail │ │ Calendar │ │ Classroom │ │ Accessibility │    │
│  └───────┘ └───────┘ └──────────┘ └───────────┘ └───────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                        VISION LAYER                                  │
│   ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐   │
│   │ AT-SPI       │  │ Remote VLM    │  │ Navigator               │   │
│   │ (<10ms)      │  │ (Jetson/Colab)│  │ (unified fallback)      │   │
│   └──────────────┘  └───────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

~34,800 Lines of Code across 104 modules meticulously carved.

```text
src/bantz/
├── __main__.py                   # CLI entry point (--once, --daemon, --doctor)
├── app.py                        # Textual main app (alternate entry)
├── config.py                     # ~60 settings from .env (Pydantic Settings)
│
├── core/                         # The Heart & Cerebrum
│   ├── brain.py                  # Main orchestrator (2103 LOC)
│   ├── router.py                 # One-shot tool routing classifier
│   └── workflow.py               # Multi-tool chain execution
│
├── data/                         # Unified Data Access Layer (DAL)
│   ├── store.py                  # Abstract base classes (7 store contracts)
│   └── layer.py                  # Singleton DataLayer — composes all stores
│
├── memory/                       # Omni-Memory Subsystems
│   ├── vector_store.py           # Pure SQLite vector store (cosine similarity)
│   ├── graph.py                  # Neo4j knowledge graph engine
│   └── context_builder.py        # Graph → LLM context injection
│
├── agent/                        # The Autonomous Subconscious
│   ├── observer.py               # Stderr observer with error classification
│   ├── rl_engine.py              # Q-learning RL (1680 states, SQLite Q-table)
│   ├── interventions.py          # Priority queue + rate limiting + focus mode
│   ├── tts.py                    # Piper + aplay streaming TTS
│   ├── proactive.py              # Idle detection → proactive engagement
│   ├── planner.py                # Plan-and-Solve decomposition
│   ├── executor.py               # Sequential plan step runner
│   └── workflows/                # Nightly / Scheduled chores
│
├── personality/                  # The Butler's Soul
│   ├── system_prompt.py          # Dynamic system prompt generation
│   ├── persona.py                # LLM persona adaptation (mood scaling)
│   └── bonding.py                # RL bonding meter — progressive formality
│
├── vision/                       # The Butler's Eyes
│   ├── navigator.py              # Unified fallback chain with per-app analytics
│   └── remote_vlm.py             # REST client for VLM validation (Jetson)
│
├── tools/                        # Arsenal of Actions
│   ├── shell.py                  # Bash with security controls
│   ├── gmail.py                  # Full Gmail integration + auto-chain compose/send
│   ├── calendar.py               # Google Calendar CRUD
│   ├── filesystem.py             # File ops + LLM auto-chain create-on-miss
│   └── ...                       # weather, news, docs, reminders, web_search, etc.
│
├── interface/                    # The Window to the World
│   ├── telegram_bot.py           # Telegram integration ("Hold the Line" UX)
│   └── tui/                      # Textual App, Telemetry, Panels, Sparklines
│
├── llm/                          # The Synapses
│   ├── ollama.py                 # Local operations (Main)
│   └── gemini.py                 # Cloud operations (Fallback)
```

---

## 🚀 Recent Roadmap Triumphs
| # | Feature | Status |
|---|---------|--------|
| **#183** | Async Telegram Progress Indicators ("Hold the Line" UX) | 🟢 Completed |
| **#181** | Direct RLHF via Sentiment & Feedback Keywords | 🟢 Completed |
| **#180** | Strict context guards against `_quick_route` hallucination | 🟢 Completed |
| **#177** | System 2 Reasoning: Pre-JSON `<thinking>` Scratchpad | 🟢 Completed |
| **#172** | System audio ducking during TTS playback | 🟢 Completed |
| **#170** | Spontaneous vector memory retrieval for deep chat | 🟢 Completed |

---

## 🎙️ Voice Pipeline (Optional)

Bantz supports fully hands-free voice interaction: wake word detection, speech-to-text,
text-to-speech, and ambient sound analysis. Enable it all with a single flag:

```bash
# .env — one flag to rule them all
BANTZ_VOICE_ENABLED=true
```

### Prerequisites (Linux)

Voice features require the **PortAudio** C library for microphone access:

```bash
# Ubuntu / Debian / Mint
sudo apt install portaudio19-dev

# Fedora
sudo dnf install portaudio-devel

# Arch
sudo pacman -S portaudio
```

Then install the Python voice extras:

```bash
pip install pyaudio faster-whisper webrtcvad pvporcupine piper-tts
```

> **First-run note:** The Whisper STT model (~39 MB for `tiny`) is downloaded
> from HuggingFace on first use. The TUI shows a "Downloading Whisper model…"
> status — this is normal and only happens once.

Run `bantz --doctor` to verify all voice dependencies are satisfied.

---

## 📜 Contributing & License

Feel free to browse our [CONTRIBUTING.md](CONTRIBUTING.md) to join the Butler's academy.
Authorized under the Apache License 2.0 — see [LICENSE](LICENSE).

<div align="center">

*Bantz — A sophisticated reflection of its creator.*

Built with love and caffeine by [@miclaldogan](https://github.com/miclaldogan)

</div>
