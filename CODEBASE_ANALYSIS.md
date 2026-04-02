# Bantz v3 — Comprehensive Codebase Analysis

> Generated: 2 April 2026  
> Codebase: `bantzv2/` — Local-first AI assistant ("your terminal, your host")

---

## Table of Contents

1. [Project Structure Overview](#1-project-structure-overview)
2. [Module/Component Breakdown](#2-modulecomponent-breakdown)
3. [Core Architecture](#3-core-architecture)
4. [Capabilities Implemented](#4-capabilities-implemented)
5. [External Dependencies](#5-external-dependencies)
6. [Test Coverage](#6-test-coverage)
7. [Known Issues / Technical Debt](#7-known-issues--technical-debt)

---

## 1. Project Structure Overview

### Root-Level Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build config (hatchling), dependencies, tool settings (ruff, pytest, coverage) |
| `docker-compose.yml` | 3-service stack: Neo4j graph, Ollama LLM, Bantz daemon |
| `Dockerfile` | ARM64 container (Python 3.11-slim + pup + readability-cli) for Oracle Free Tier |
| `setup.sh` | Interactive quickstart — venv, pip install, optional MarianMT, .env creation |
| `.env.example` | 221-line reference of every environment variable (>80 settings) |
| `client_secret.json` | Google OAuth2 credentials for Gmail/Calendar/Classroom |
| `Bantz_en_linux_v4_0_0.ppn` | Picovoice Porcupine wake-word model binary ("Hey Bantz") |
| `LICENSE` / `LICENSE.txt` | Project license |
| `README.md` | User-facing documentation |
| `CODE_OF_CONDUCT.md` | Community standards |
| `CONTRIBUTING.md` | Contribution guidelines |
| `SECURITY.md` | Security policy |
| `last_bantz_session.txt` / `second_last_bantz_session.txt` / `latest_bantz_chat.txt` | Debug dumps of recent sessions |
| `pytest_final.txt` | Last test run output |

### Top-Level Directories

| Directory | Purpose |
|-----------|---------|
| `src/bantz/` | Main application source (16 sub-packages) |
| `tests/` | Pytest test suite (76 test files across 10 sub-directories) |
| `db/migrations/` | Neo4j Cypher migration scripts |
| `deploy/` | Deployment assets (systemd units, VLM server, Colab notebook) |

### `db/migrations/`

| File | Purpose |
|------|---------|
| `001_initial_schema.cypher` | Creates property indexes on all 11 Neo4j node labels |
| `002_fulltext_index.cypher` | Creates `bantz_fulltext` cross-label full-text search index |

### `deploy/`

| File | Purpose |
|------|---------|
| `bantz@.service` | systemd user service unit for Bantz daemon |
| `bantz-telegram@.service` | systemd unit for Telegram bot mode |
| `vlm_server.py` | Remote VLM inference server (for Jetson/Colab GPU) |
| `vlm_colab.ipynb` | Google Colab notebook for running VLM remotely |

### `src/bantz/` Top-Level Files

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point — 15+ subcommands (`--daemon`, `--doctor`, `--setup`, `--once`, etc.) |
| `app.py` | Backward-compat shim re-exporting `BantzApp` from `interface.tui.app` |
| `config.py` | Pydantic-settings `Config` class reading 80+ env vars from `.env` |

### `src/bantz/` Sub-Packages

| Package | # Files | Purpose |
|---------|---------|---------|
| `core/` | 30 | Central orchestration — brain, routing, memory injection, context, scheduling, location, time, translation, types |
| `agent/` | 18 + 3 workflows | Background agents — TTS, STT, wake word, ghost loop, observer, planner, executor, health, interventions, proactive, affinity, ambient, app detector, audio ducker, job scheduler |
| `memory/` | 11 | Memory subsystem — SQLite vectors, Neo4j graph, embeddings, distillation, deep probe, omni-memory |
| `data/` | 9 | Data access layer — connection pool, abstract stores, SQLite/JSON implementations, migrations, models |
| `tools/` | 24 | Tool implementations — shell, gmail, calendar, classroom, weather, news, web search, filesystem, browser, screenshot, input control, visual click, computer use, accessibility, contacts, document, reminder, summarizer |
| `llm/` | 2 | LLM clients — Ollama (local) and Gemini (Google API) |
| `interface/` | 8 | User interfaces — Textual TUI (panels, widgets, mood, telemetry) and Telegram bot |
| `personality/` | 4 | Persona engine — bonding meter, dynamic persona builder, greeting manager, system prompts |
| `auth/` | 2 | Google OAuth2 — interactive setup flow and secure token persistence |
| `i18n/` | 1 | Turkish↔English translation via MarianMT |
| `scheduler/` | 1 | APScheduler wrapper with Redis/memory job stores |
| `vision/` | 6 | Computer vision — screenshot capture, remote VLM client, spatial cache, browser vision, autonomous navigation loop, navigator pipeline |
| `integrations/` | 1 | Backward-compat shim for Telegram bot import path |

---

## 2. Module/Component Breakdown

### 2.1 `src/bantz/__main__.py` — CLI Entry Point (1720 lines)

**Responsibility:** Parses CLI arguments and dispatches to the appropriate mode: TUI, daemon, single-query, doctor, setup wizards, maintenance, and reflection commands.

| Function | Signature | Summary |
|----------|-----------|---------|
| `main` | `() -> None` | CLI entry point. Parses 15+ flags (--once, --daemon, --doctor, --setup, --jobs, --run-job, --maintenance, --reflect, --overnight-poll, --mood-history, --config, --cache-stats, --reflections, --dry-run) and dispatches to internal handlers. |
| `_handle_setup` | `(parts: list[str]) -> None` | Routes `--setup` subcommands (profile, google, schedule, telegram, places, gemini, systemd) to interactive wizards. |
| `_setup_telegram` | `() -> None` | Interactive Telegram bot token + proxy configuration wizard. |
| `_setup_gemini` | `() -> None` | Interactive Gemini API key setup wizard. |
| `_setup_places` | `async () -> None` | Interactive known-places wizard with Nominatim geocoding and IP-based auto-detection. |
| `_setup_profile` | `() -> None` | Interactive user profile setup (name, university, year, preferences). |
| `_setup_schedule` | `() -> None` | Interactive weekly class timetable builder. |
| `_setup_systemd` | `() -> None` | Installs `bantz.service` as a systemd user unit. |
| `_doctor` | `async () -> None` | System health check — probes Ollama, Neo4j, Gemini, Google OAuth, translations, embeddings, piper TTS, wake word, disk space. |
| `_daemon` | `async () -> None` | Headless daemon mode — starts data layer, Telegram bot, job scheduler, GPS server. No TUI. |
| `_once` | `async (query: str) -> None` | Single-query mode — processes one input through Brain and prints response. |
| `_list_jobs` | `async () -> None` | Lists all active APScheduler jobs. |
| `_run_job` | `async (job_id: str) -> None` | Manually triggers a specific scheduled job. |
| `_maintenance` | `async (dry_run: bool) -> None` | Runs the 6-step maintenance workflow. |
| `_reflect` | `async (dry_run: bool) -> None` | Runs the nightly reflection workflow. |
| `_overnight_poll` | `async (dry_run: bool) -> None` | Runs one overnight poll cycle. |
| `_view_reflections` | `() -> None` | Displays recent daily reflections from storage. |
| `_mood_history` | `() -> None` | Shows 24-hour mood transition history. |
| `_show_config` | `() -> None` | Displays current configuration with secrets masked. |
| `_cache_stats` | `() -> None` | Displays spatial cache statistics. |

**Key imports:** `argparse`, `asyncio`; all others are lazy-imported within handlers.

---

### 2.2 `src/bantz/config.py` — Configuration

**Responsibility:** Single `Config` class (Pydantic-settings) loading 80+ environment variables from `.env`.

| Class | Signature | Summary |
|-------|-----------|---------|
| `Config(BaseSettings)` | 80+ fields with `Field(alias="BANTZ_*")` | Pydantic-settings model reading all configuration from environment. Groups: Ollama, Embeddings, Distillation, VLM/Vision, Input Control, Gemini, Language/Translation, Shell Security, Location, GPS, Neo4j, Storage, Briefing/Digests, Scheduler, Telegram, Observer, RL Engine, Bonding, Interventions, App Detector, Notifications, Voice Pipeline (TTS, Audio Ducking, Wake Word, STT/Ghost Loop, Ambient, Proactive, Health, Persona, Deep Memory). Includes `_voice_master_switch` model validator that cascades `BANTZ_VOICE_ENABLED=true` into 6 sub-flags. Properties: `db_path`, `ensure_dirs()`. |

**Key imports:** `pydantic`, `pydantic_settings`

**Module singleton:** `config = Config()`

---

### 2.3 `src/bantz/core/` — Core Orchestration (30 files)

#### `brain.py` — Central Orchestrator
| Class | Summary |
|-------|---------|
| `Brain` | God-object orchestrator. `process()` pipeline: translate → quick_route (regex) → cot_route (LLM CoT) → execute tool → finalize → persist to memory/graph. Handles streaming, hallucination checking, RLHF feedback, VLM screen descriptions, and dynamic tool-context injection. |

#### `context.py` — Request-Scoped Dataclass
| Class | Summary |
|-------|---------|
| `BantzContext` | ~30-field dataclass carrier for pipeline data (input → translation → time → RLHF → memory → routing → execution → finalisation). Dependency-free. Properties: `is_tool_call`, `has_memory`, `elapsed_ms`. |

#### `types.py` — Shared Types
| Class | Summary |
|-------|---------|
| `BrainResult` | Standard response payload with `response`, `tool_used`, `needs_confirm`, `stream`, `attachments`. |
| `Attachment` | File attachment (type, data bytes, caption, filename, mime_type). |

#### `event_bus.py` — Pub/Sub Bus
| Class | Summary |
|-------|---------|
| `EventBus` | Thread-safe async pub/sub singleton. `emit_threadsafe()` bridges daemon threads into asyncio. Wildcard `"*"` subscriptions. Methods: `on()`, `off()`, `emit()`, `emit_threadsafe()`. |

#### `intent.py` — Chain-of-Thought Router
| Function | Summary |
|----------|---------|
| `cot_route(en_input, tool_schemas, ...)` | Streams LLM chain-of-thought via `ollama.chat_stream()`, emits thinking tokens to TUI, extracts JSON routing decision. Retries once on parse failure. |

#### `routing_engine.py` — Routing Logic
| Function | Summary |
|----------|---------|
| `quick_route(orig, en)` | Regex fast-path for hardware/UI controls (TTS, wake word, audio, clear memory, app launch, navigation, desktop click). |
| `dispatch_internal(tool, args, ...)` | Executes internal underscore-prefixed tools (`_tts_stop`, `_briefing`, `_maintenance`, `_location`, `_schedule_*`, etc.). |
| `execute_plan(user_input, en_input, ...)` | Decomposes complex requests via planner agent, executes via plan executor. |

#### `router.py` — Legacy Router
| Function | Summary |
|----------|---------|
| `route(en_input, tool_schemas)` | Legacy one-shot Ollama JSON routing classifier. Superseded by `intent.py`. |

#### `finalizer.py` — Response Post-Processing
| Function | Summary |
|----------|---------|
| `finalize(en_input, result, ...)` | Post-processes tool output through LLM for butler-persona responses. Short output (<800 chars) returned verbatim. |
| `hallucination_check(response, tool_output)` | Compares finalizer response against tool output; detects fabricated data. |

#### `prompt_builder.py` — System Prompt Assembly
| Function | Summary |
|----------|---------|
| `build_chat_system(ctx, tc)` | Renders CHAT_SYSTEM template with persona, style, memory, desktop context, time, profile, and feedback hints. |

#### `memory_injector.py` — Context Enrichment
| Function | Summary |
|----------|---------|
| `inject(ctx, en_input)` | One-call context enrichment — populates all memory/persona/formality fields on BantzContext concurrently via OmniMemoryManager. |

#### `translation_layer.py` — Language Bridge
| Function | Summary |
|----------|---------|
| `to_en(text)` | Translates to English via MarianMT bridge. |
| `detect_feedback(raw_input)` | Detects positive/negative feedback in English+Turkish. |
| `resolve_message_ref(text, messages)` | Resolves "the first one" → email message ID. |

#### `location.py` — 7-Source Location Service
| Class | Summary |
|-------|---------|
| `LocationService` | Resolves location via: .env → GPS → WiFi SSID → places.json → GeoClue2 → ipinfo.io → fallback. Cached. |

#### `places.py` — Named Place Management
| Class | Summary |
|-------|---------|
| `PlaceService` | Geofence detection, stationary tracking (45+ min → suggest save), location-triggered reminders, travel hints (walking time to campus). Haversine distance. |

#### `gps_server.py` — Phone GPS Receiver
| Class | Summary |
|-------|---------|
| `GPSServer` | Dual-mode: LAN HTTP server (port 9777) + ntfy.sh relay. Includes PWA phone app, persists last GPS to disk. |

#### `briefing.py` — Morning Briefing Generator
| Class | Summary |
|-------|---------|
| `Briefing` | `generate()` fires parallel coroutines for weather, calendar, Gmail, classroom; formats into broadcaster-style text. |

#### `butler.py` — Launch Greeting Composer
| Class | Summary |
|-------|---------|
| `Butler` | `greet(session_info)` generates absence-aware greeting with parallel data fetchers for mail/calendar/classroom/schedule. |

#### `digest.py` — Daily/Weekly Digest Generator
| Class | Summary |
|-------|---------|
| `DigestManager` | `daily_digest()` and `weekly_digest()` gather data from all sources, synthesize through Gemini/Ollama. |

#### `scheduler.py` (core) — Persistent Task Scheduler
| Class | Summary |
|-------|---------|
| `Scheduler(ReminderStore)` | SQLite-backed reminders with one-shot, recurring (daily/weekly/weekdays/custom), location-triggered, and snooze support. |

#### `schedule.py` — University Timetable
| Class | Summary |
|-------|---------|
| `Schedule` | Reads weekly timetable from SQLite/JSON. Provides `today()`, `next_class()`, `format_week()`. |

#### `session.py` — Session Tracker
| Class | Summary |
|-------|---------|
| `SessionTracker` | Tracks launch timestamps and session count for absence-aware greetings ("Been about a week"). |

#### `profile.py` — User Profile
| Class | Summary |
|-------|---------|
| `Profile` | User identity (name, university, year, response style, pronoun). `prompt_hint()` generates one-line LLM hint. |

#### `time_context.py` — Time-of-Day Awareness
| Class | Summary |
|-------|---------|
| `TimeContext` | `snapshot()` returns hour, segment, greeting, date. `greeting_line()` produces personalized greeting. |

#### `time_parser.py` — Natural Time Resolution
| Function | Summary |
|----------|---------|
| `resolve_time(text)` | "5pm" → "17:00", "noon" → "12:00", etc. |

#### `date_parser.py` — Natural Date Resolution
| Function | Summary |
|----------|---------|
| `resolve_date(text)` | "tomorrow" → datetime, "next thursday" → datetime, ISO passthrough. |

#### `habits.py` — Usage Pattern Mining
| Class | Summary |
|-------|---------|
| `HabitEngine` | Mines SQLite messages table for habitual tool usage by time segment. |

#### `memory.py` — Conversation Memory
| Class | Summary |
|-------|---------|
| `Memory(ConversationStore)` | Full conversation store with FTS5, vector embeddings, hybrid search, session distillation, backfill. |

#### `notification_manager.py` — Notification Router
| Function | Summary |
|----------|---------|
| `notify_toast(title, reason, ...)` | Routes toasts to TUI → Textual → notify-send → silent no-op. |

#### `rl_hooks.py` — RL Reward Signals
| Function | Summary |
|----------|---------|
| `rl_reward_hook(tool_name, result)` | +1.0 on success, -0.5 on failure to affinity engine. |
| `rl_feedback_reward(feedback, ...)` | +2.0 positive, -2.0 negative. |

#### `workflow.py` — Multi-Tool Chain Detector
| Class | Summary |
|-------|---------|
| `WorkflowEngine` | Splits input on conjunctions, maps sub-sentences to tools, executes sequentially. |

#### `location_handler.py` — Location Query Handlers
| Function | Summary |
|----------|---------|
| `handle_location()` | "Where am I?" — shows GPS/location info. |
| `handle_save_place(name)` / `handle_list_places()` / `handle_delete_place(name)` | CRUD for named places. |

---

### 2.4 `src/bantz/agent/` — Background Agents (18 + 3 workflow files)

#### `affinity_engine.py` — Cumulative Affinity Score
| Class | Summary |
|-------|---------|
| `AffinityEngine` | Thread-safe [-100, 100] score → persona mood directives. Persisted in SQLite KV store. Replaces Q-learning with simpler accumulator. |

#### `planner.py` — Multi-Step Decomposition
| Class | Summary |
|-------|---------|
| `PlannerAgent` | LLM breaks complex requests into JSON step arrays. Streaming `<thinking>` blocks to TUI. Tool name normalization. `replan()` for dynamic Plan B after failures. |

#### `executor.py` — Plan Executor
| Class | Summary |
|-------|---------|
| `PlanExecutor` | Runs PlanSteps sequentially with `$REF_STEP_N` context threading, circuit breaker on failure, dynamic replanning (#216). |

#### `observer.py` — Stderr Observer
| Class | Summary |
|-------|---------|
| `Observer` | Event-driven stderr classifier. ~16 regex patterns + optional LLM escalation for critical errors. Deduplication and batching. |

#### `interventions.py` — Proactive Intervention System
| Class | Summary |
|-------|---------|
| `InterventionQueue` | Priority queue with rate limiting (N/hour), focus mode, quiet mode, TTL auto-dismiss. SQLite logging. |

#### `health.py` — Health Rule Engine
| Class | Summary |
|-------|---------|
| `HealthRuleEvaluator` | 5 rules: late-night load, marathon session, eye strain, thermal stress, late-night music. Per-rule cooldowns. |

#### `proactive.py` — Proactive Engagement
| Class | Summary |
|-------|---------|
| `ProactiveEngine` | Initiates conversations during idle periods. Activity gates (blocks coding/productivity), RL-adaptive daily limits (1→3), ambient-aware LLM prompts. |

#### `job_scheduler.py` — APScheduler Job Manager
| Class | Summary |
|-------|---------|
| `JobScheduler` | Cron jobs: maintenance@03:00, reflection@23:00, briefing prep@06:00. Overnight poll every 2h. Reminder check every 30s. Dynamic reminder injection. Retry with exponential backoff. Sleep-inhibit. |

#### `tts.py` — Text-to-Speech
| Class | Summary |
|-------|---------|
| `TTSEngine` | Streaming Piper TTS with sentence-level producer-consumer pipeline. Audio ducking integration, SoX animatronic filter, `PULSE_PROP` tagging. |

#### `stt.py` — Speech-to-Text
| Class | Summary |
|-------|---------|
| `STTEngine` | Local faster-whisper (CTranslate2). PCM→float32→inference with VAD filter and beam search. |

#### `wake_word.py` — "Hey Bantz" Detection
| Class | Summary |
|-------|---------|
| `WakeWordListener` | Daemon-thread Porcupine listener. Interrupts active TTS, plays 880Hz beep, emits `wake_word_detected`. Piggybacks ambient analyser on the mic stream. |

#### `ghost_loop.py` — Voice Interaction Loop
| Class | Summary |
|-------|---------|
| `GhostLoop` | Wake → capture (VAD) → STT (faster-whisper) → EventBus dispatch. 60-second conversation-mode follow-ups. |

#### `voice_capture.py` — Microphone Recorder
| Class | Summary |
|-------|---------|
| `VoiceCapture` | Blocking recorder with WebRTC VAD silence detection. 30-second safety cap. |

#### `ambient.py` — Environmental Audio Classifier
| Class | Summary |
|-------|---------|
| `AmbientAnalyzer` | Classifies SILENCE/SPEECH/NOISY using RMS+ZCR (no FFT). 24h rolling history. |

#### `audio_ducker.py` — Volume Ducking
| Class | Summary |
|-------|---------|
| `AudioDucker` | Lowers other apps' audio via `pactl` while TTS speaks. 3-step fade support. |

#### `app_detector.py` — Active Window Detector
| Class | Summary |
|-------|---------|
| `AppDetector` | X11 PropertyNotify / D-Bus / slow-poll backends. Classifies activity (CODING/BROWSING/ENTERTAINMENT/COMMUNICATION/PRODUCTIVITY/IDLE). Browser title parsing, IDE context extraction, Docker container listing. |

#### `notifier.py` — Desktop Notification Dispatcher
| Class | Summary |
|-------|---------|
| `Notifier` | `notify-send` with TUI-active detection skip, focus/quiet mode filtering, priority→urgency mapping. |

#### `workflows/maintenance.py` — Nightly Maintenance
| Function | Summary |
|----------|---------|
| `run_maintenance(*, dry_run)` | 6 steps: Docker cleanup → temp purge → disk health → service ping → log rotation → report. Sleep-inhibit, 5-min cap. |

#### `workflows/reflection.py` — Nightly Reflection
| Function | Summary |
|----------|---------|
| `run_reflection(*, dry_run)` | 7 steps: collect data → LLM reflect → parse → entity extraction → graph store → prune old messages → report. 10-min cap. |

#### `workflows/overnight_poll.py` — Overnight Data Collection
| Function | Summary |
|----------|---------|
| `run_overnight_poll(*, dry_run)` | Polls Gmail + Calendar + Classroom every 2h. KV-store caching for morning briefing. Urgent email detection. |

---

### 2.5 `src/bantz/memory/` — Memory Subsystem (11 files)

#### `omni_memory.py` — Unified Memory Orchestrator
| Class | Summary |
|-------|---------|
| `OmniMemoryManager` | Budget-aware parallel hybrid search: graph 35% + vector 40% + deep 25%. Token cap (400). CRUD + transactional writes. |

#### `graph.py` — Neo4j Knowledge Graph
| Class | Summary |
|-------|---------|
| `GraphMemory` | Neo4j graph lifecycle: connect, index creation, entity extraction + upsert, context retrieval, stats, growth tracking, delete operations. |

#### `nodes.py` — Entity Extraction
| Function | Summary |
|----------|---------|
| `extract_entities(user_msg, assistant_msg, ...)` | Rule-based extraction of 11 entity types (Person, Topic, Decision, Task, Event, Location, Document, Reminder, Commitment, Project, Fact) with cross-reference relationships. |

#### `context_builder.py` — Graph Context Builder
| Function | Summary |
|----------|---------|
| `build_context(user_msg, query_fn)` | Orchestrates 7 graph queries (people, tasks, decisions, events, commitments, reminders, keywords). Token-budget enforced. |

#### `embeddings.py` — Embedding Client
| Class | Summary |
|-------|---------|
| `Embedder` | Async wrapper for Ollama `/api/embed` endpoint. Auto-detects vector dimension. |

#### `vector_store.py` — SQLite Vector Store
| Class | Summary |
|-------|---------|
| `VectorStore` | Pure-SQLite vector store. Float32 blob storage. Brute-force cosine search with optional time-decay. Orphan pruning. |

#### `deep_probe.py` — Spontaneous Recall
| Class | Summary |
|-------|---------|
| `DeepMemoryProbe` | Probabilistic vector retrieval for conversational depth. Rate-limited (every Nth message), time-decayed ranking, déjà-vu deduplication. |

#### `distiller.py` — Session Distillation
| Function | Summary |
|----------|---------|
| `distill_session(session_id, ...)` | Full pipeline: fetch messages → threshold check → LLM summarise → embed → store → graph entity extraction. |

#### `memory_manager.py` — High-Level Graph API
| Class | Summary |
|-------|---------|
| `MemoryManager` | Thin wrapper: `store()`, `query()` (full-text search), `summarize_context()` (narrative summary). |

#### `session_store.py` — Redis/Memory Session Store
| Class | Summary |
|-------|---------|
| `SessionStore` | Redis HSET with TTL; `_MemorySessionStore` fallback. Also: `TaskQueue`, `PubSub`, `RateLimiter`. |

---

### 2.6 `src/bantz/data/` — Data Access Layer (9 files)

#### `store.py` — Abstract Interfaces
7 ABCs: `ConversationStore`, `ReminderStore`, `ProfileStore`, `PlaceStore`, `ScheduleStore`, `SessionStore`, `GraphStore`.

#### `sqlite_store.py` — SQLite Implementations
7 concrete stores: `SQLiteConversationStore`, `SQLiteReminderStore`, `SQLiteProfileStore`, `SQLitePlaceStore`, `SQLiteScheduleStore`, `SQLiteSessionStore`, `SQLiteKVStore`.

#### `json_store.py` — JSON File Fallbacks
4 stores: `JSONProfileStore`, `JSONPlaceStore`, `JSONScheduleStore`, `JSONSessionStore`.

#### `connection_pool.py` — SQLite Pool
| Class | Summary |
|-------|---------|
| `SQLitePool` | Thread-safe singleton pool. WAL mode, single-writer lock, 5 connections, PRAGMA optimizations. |

#### `async_executor.py` — Async DB Access
| Function | Summary |
|----------|---------|
| `run_in_db(fn, *, write)` | Wraps blocking SQLite calls in a 5-thread `ThreadPoolExecutor`. |

#### `layer.py` — Unified Data Singleton
| Class | Summary |
|-------|---------|
| `DataLayer` | Composes all stores, auto-migrates JSON→SQLite, initializes optional subsystems (spatial cache, navigator, affinity engine, etc.). |

#### `models.py` — Pydantic Models
7 models: `Message`, `Conversation`, `Reminder`, `Place`, `ScheduleEntry`, `UserProfile`, `SessionInfo`.

#### `migration.py` — JSON→SQLite Migration
| Function | Summary |
|----------|---------|
| `validate_json_files(data_dir)` | Checks v2 JSON files for existence and parseability. |
| `migrate_to_sqlite(db_path, data_dir)` | Imports v2 JSON data into unified SQLite tables. |

---

### 2.7 `src/bantz/llm/` — LLM Clients (2 files)

#### `ollama.py`
| Class | Summary |
|-------|---------|
| `OllamaClient` | Local Ollama `/api/chat` client. `chat()` single response, `chat_stream()` NDJSON streaming. Health check via `/api/tags`. |

#### `gemini.py`
| Class | Summary |
|-------|---------|
| `GeminiClient` | REST-based Google Gemini client (no SDK). `chat()` single response, `chat_stream()` SSE streaming. Message format conversion from OpenAI-style to Gemini format. |

---

### 2.8 `src/bantz/tools/` — Tool Implementations (24 files)

#### Registry Framework (`__init__.py`)
| Class | Summary |
|-------|---------|
| `ToolResult` | Standardized return value: `success`, `output`, `data`, `error`. |
| `BaseTool(ABC)` | Abstract base: `name`, `description`, `risk_level`, `execute(**kwargs)`, `schema()`. |
| `ToolRegistry` | Singleton registry with fuzzy lookup. `register()`, `get()`, `all_schemas()`, `names()`. |

#### Implemented Tools

| Tool | File | Actions | Risk |
|------|------|---------|------|
| `shell` | `shell.py` | Execute bash commands | Destructive detection, blocked list |
| `system` | `system.py` | CPU, RAM, disk, uptime | Safe |
| `filesystem` | `filesystem.py` | ls, read, write, create | Home-dir sandbox |
| `gmail` | `gmail.py` | 17 actions: summary, count, read, thread, search, filter, send, compose, reply, forward, star, mark_read, labels, contacts | Moderate |
| `calendar` | `calendar.py` | today, week, date, upcoming, create, delete, update, conflicts | Safe |
| `classroom` | `classroom.py` | assignments, announcements, due_today, courses | Safe |
| `weather` | `weather.py` | Current + 3-day forecast via wttr.in | Safe |
| `news` | `news.py` | Hacker News + Google News RSS | Safe |
| `web_search` | `web_search.py` | DuckDuckGo search with cache + follow-up | Safe |
| `read_url` | `web_reader.py` | Fetch + extract text from URLs | Safe |
| `document` | `document.py` | PDF, DOCX, TXT, CSV reading + LLM Q&A | Safe |
| `reminder` | `reminder.py` | add, list, cancel, snooze + location triggers + Neo4j | Safe |
| `summarizer` | `summarizer.py` | LLM text summarization/analysis/rewriting | Safe |
| `screenshot` | `screenshot_tool.py` | Desktop screenshot with region crop | Safe |
| `browser_control` | `browser_control.py` | open, navigate, screenshot, click, type, scroll | Moderate |
| `accessibility` | `accessibility.py` | AT-SPI tree inspection + VLM fallback | Safe |
| `visual_click` | `visual_click.py` | Cache→AT-SPI→VLM click pipeline | Moderate |
| `input_control` | `input_control.py` | Mouse/keyboard with pyautogui/pynput/xdotool | Dangerous |
| `computer_use` | `computer_use.py` | Autonomous vision loop (screenshot→VLM→act) | Dangerous |
| `contacts` | `contacts.py` | Google Contacts search, list, sync | Safe |
| `gui_action` | `gui_action.py` | *Deprecated* — legacy AT-SPI GUI action | — |
| `mail` | `mail.py` | *Empty placeholder* | — |
| `system_tool` | `system_tool.py` | Internal process runner with denylist + audit | — |
| `contact_resolver` | `contact_resolver.py` | 3-tier email resolution: alias → Google People → passthrough | — |

---

### 2.9 `src/bantz/interface/` — User Interfaces (8 files)

#### `telegram_bot.py` — Telegram Bot
| Function | Summary |
|----------|---------|
| `run_bot()` | Builds `Application`, registers 13+ command handlers (/start, /briefing, /hava, /mail, /takvim, /odev, /ders, /siradaki, /haber, /hatirlatici, /digest, /ekran, /weekly), scheduled jobs (daily/weekly digest, reminder check), and a spam filter. Streaming replies via placeholder editing. |

#### `tui/app.py` — Textual TUI
| Class | Summary |
|-------|---------|
| `BantzApp(App)` | Main TUI. Composes OperationsHeader, ChatLog, SystemStatus, ToastContainer, Footer. Starts ~15 background workers: GPS, briefing, reminders, digests, observer, interventions, RL suggestions, ghost loop, wake word, stationary checker, Ollama warmup. EventBus→Textual main-thread bridging. |

#### `tui/panels/chat.py` — Chat Panel
| Class | Summary |
|-------|---------|
| `ChatLog(RichLog)` | Scrollable chat with streaming support, thinking panel, user/system/error/tool message types. |
| `ThinkingPanel(Static)` | Collapsible LLM reasoning stream. |

#### `tui/panels/header.py` — Operations Header
| Class | Summary |
|-------|---------|
| `OperationsHeader(Static)` | Live health dots (Ollama/Neo4j/Gemini/Telegram), uptime, session count, memory count, voice status. |

#### `tui/panels/system.py` — Telemetry Panel
| Class | Summary |
|-------|---------|
| `SystemStatus(Vertical)` | CPU, RAM, Disk, Net TX/RX, CPU°C, GPU°C, VRAM sparkline graphs with peak tracking. |

#### `tui/mood.py` — Mood State Machine
| Class | Summary |
|-------|---------|
| `MoodStateMachine` | 5 moods: CHILL, FOCUSED, BUSY, STRESSED, SLEEPING. Hysteresis (10s). SQLite-backed transition history. |

#### `tui/telemetry.py` — Hardware Collector
| Class | Summary |
|-------|---------|
| `TelemetryCollector` | 2-second interval metrics: CPU, RAM, disk, net I/O, CPU temp, GPU temp (pynvml), VRAM. 60-sample ring buffer. |

#### `tui/widgets/toast.py` — Toast Notifications
| Class | Summary |
|-------|---------|
| `ToastContainer` | Max 3 visible toasts with overflow queue, auto-dismiss, keyboard accept/dismiss. |

---

### 2.10 `src/bantz/personality/` — Persona Engine (4 files)

#### `system_prompt.py` — Prompt Templates
4 templates: `BANTZ_IDENTITY` (1920s butler core), `BANTZ_ROUTER` (JSON routing), `BANTZ_FINALIZER` (tool result presentation), `BANTZ_CHAT` (conversational).

#### `persona.py` — Dynamic Persona Builder
| Class | Summary |
|-------|---------|
| `PersonaStateBuilder` | 7 states (STRAINED→ENERGETIC→SLEEPY→FOCUSED→RELAXED→BONDED→NEUTRAL) from CPU/RAM/thermal/activity/RL/time. Priority rules with hysteresis. |

#### `bonding.py` — Formality Meter
| Class | Summary |
|-------|---------|
| `BondingMeter` | Sigmoid mapping of cumulative RL reward → formality index (0.0–1.0). 5 tiers: Ultra Formal → Bonded. Highwater drop-limit (never drops >10% below peak). |

#### `greeting.py` — Morning Briefing Manager
| Class | Summary |
|-------|---------|
| `GreetingManager` | Fires daily briefing once/day within a 15-minute window at configured hour (default 08:00). |

---

### 2.11 `src/bantz/vision/` — Computer Vision (6 files)

#### `screenshot.py` — Multi-Backend Capture
5 backends: `grim` (Wayland), `gnome-screenshot`, `scrot` (X11), `import` (ImageMagick), `Pillow` (fallback). ROI cropping, JPEG compression, window-specific capture.

#### `remote_vlm.py` — VLM Client
| Function | Summary |
|----------|---------|
| `analyze_screenshot(image_b64, ...)` | Remote VLM (Jetson/Colab) with local Ollama fallback (llava/bakllava/moondream). Multi-model cascade. |

#### `spatial_cache.py` — Persistent UI Element Cache
| Class | Summary |
|-------|---------|
| `SpatialCacheDB` | SQLite-backed cache for UI element coordinates. Resolution-aware, TTL (24h), confidence decay, hit-count tracking, LRU eviction (1000 entries). |

#### `navigator.py` — Unified Navigation Pipeline
| Class | Summary |
|-------|---------|
| `Navigator` | Cache → AT-SPI → VLM pipeline for locating UI elements. Per-app analytics for method ordering optimization. Supports click/double-click/right-click/type/focus actions. |

#### `browser_vision.py` — Browser Automation Vision
| Function | Summary |
|----------|---------|
| `wait_for_load(app, ...)` | Page-load detection via screenshot stability (MD5 hashing). |
| `find_element(element_description, ...)` | Multi-strategy: VLM direct coords → VLM region → layout heuristics. |
| `find_and_click_element(...)` | Full observe→find→click loop. |

#### `computer_use.py` — Autonomous Vision Loop
| Class | Summary |
|-------|---------|
| `AutonomousVisionLoop` | Closed-loop: Observe (screenshot) → Analyze (VLM) → Act (input control) → Verify (VLM goal check). Stuck detection, domain allowlists, safety gates. |

---

### 2.12 Other Packages

#### `src/bantz/auth/` (2 files)
- `google_oauth.py` — Interactive OAuth2 flow (port 8765) for Gmail/Calendar/Classroom
- `token_store.py` — Secure token persistence in `~/.local/share/bantz/tokens/` with `0o600` permissions, auto-refresh

#### `src/bantz/i18n/bridge.py`
| Class | Summary |
|-------|---------|
| `LanguageBridge` | Async TR↔EN translation via Helsinki-NLP MarianMT. Lazy model loading, thread-pool execution to avoid blocking async. |

#### `src/bantz/scheduler/scheduler.py`
| Class | Summary |
|-------|---------|
| `Scheduler` | APScheduler wrapper with Redis (or MemoryJobStore) persistence. Cron, one-shot, and interval job types. Per-job error logging. |

#### `src/bantz/integrations/telegram_bot.py`
Backward-compat shim re-exporting `run_bot` from `bantz.interface.telegram_bot`.

---

## 3. Core Architecture

### 3.1 Entry Points

```
bantz CLI (__main__.py)
  ├── TUI Mode (default)     → interface/tui/app.py → BantzApp
  ├── Daemon Mode (--daemon)  → Telegram bot + Job scheduler + GPS server
  ├── Single Query (--once)   → Brain.process() → stdout
  ├── Doctor (--doctor)       → Health checks
  └── Setup Wizards (--setup) → Interactive configuration
```

### 3.2 Request Processing Pipeline

```
User Input (TUI / Telegram / --once)
  │
  ├─ 1. Translation Layer    → to_en() via MarianMT (if Turkish)
  ├─ 2. Feedback Detection   → detect_feedback() for RLHF signals
  ├─ 3. Memory Injection     → OmniMemoryManager.recall() (parallel):
  │     ├─ Graph Memory (Neo4j) — 35% budget
  │     ├─ Vector Memory (SQLite cosine search) — 40% budget
  │     └─ Deep Memory (spontaneous recall) — 25% budget
  ├─ 4. Context Enrichment   → persona state, formality tier, desktop context
  ├─ 5. Quick Route           → Regex fast-path for hardware/UI controls
  ├─ 6. CoT Route             → LLM chain-of-thought routing (intent.py)
  │     └─ Streams <thinking> tokens to TUI
  ├─ 7. Tool Execution        → ToolRegistry.get(name).execute(**args)
  │     └─ Complex: Planner → Executor (multi-step with replanning)
  ├─ 8. Finalization          → LLM butler-persona post-processing
  │     └─ Hallucination check against tool output
  ├─ 9. Translation (back)    → to_turkish() if needed
  ├─ 10. Memory Persist       → SQLite messages + vector embeddings + Neo4j graph
  └─ 11. RL Hooks             → Affinity engine reward signals
```

### 3.3 Background Services

```
Job Scheduler (APScheduler)
  ├─ 03:00  Maintenance Workflow (Docker cleanup, disk, logs)
  ├─ 06:00  Briefing Prep (pre-fetch weather, calendar, email)
  ├─ 08:00  Morning Briefing (via GreetingManager)
  ├─ 20:00  Daily Digest (via DigestManager)
  ├─ 23:00  Nightly Reflection (LLM summarize → graph entities → prune)
  ├─ 0,2,4,6h  Overnight Poll (Gmail, Calendar, Classroom)
  ├─ Every 30s  Reminder Check
  ├─ Every 10s  Briefing Watcher (idle→active TTS trigger)
  └─ Configurable  Proactive Engagement, Health Checks

Voice Pipeline (daemon threads)
  ├─ WakeWordListener (Porcupine on dedicated thread)
  ├─ GhostLoop (wake → capture → STT → dispatch)
  ├─ AmbientAnalyzer (piggybacks on wake word mic)
  └─ AudioDucker (pactl volume management)

Event-Driven
  ├─ AppDetector (X11/D-Bus/poll → app_changed events)
  ├─ Observer (stderr_line events → classification → alerts)
  └─ InterventionQueue (priority queue → TUI toasts / desktop notifications)
```

### 3.4 Design Patterns Used

| Pattern | Where | Details |
|---------|-------|---------|
| **Singleton** | Every module | `config`, `brain`, `memory`, `scheduler`, `data_layer`, `bus`, etc. — module-level instances created at import time |
| **Abstract Factory / Strategy** | `data/store.py` | 7 ABC interfaces with SQLite + JSON implementations, swappable at init |
| **Observer / Pub-Sub** | `core/event_bus.py` | Thread-safe async event bus bridging daemon threads with asyncio |
| **Pipeline** | `core/brain.py` | Sequential processing pipeline (translate → route → execute → finalize → persist) |
| **Chain of Responsibility** | `vision/navigator.py` | Cache → AT-SPI → VLM fallback chain, analytics-optimized ordering |
| **Command** | `tools/__init__.py` | `BaseTool.execute(**kwargs)` — uniform command interface for all tools |
| **State Machine** | `tui/mood.py` | 5-state mood machine with hysteresis |
| **Circuit Breaker** | `agent/executor.py` | Aborts remaining plan steps after consecutive failures |
| **Producer-Consumer** | `agent/tts.py` | Sentence N+1 synthesized while sentence N plays |
| **Decorator** | `interface/telegram_bot.py` | `@_authorized` restricts handlers to allowed users |
| **Facade** | `memory/omni_memory.py` | Unified API over graph + vector + deep memory subsystems |
| **Data Transfer Object** | `core/context.py` | `BantzContext` carries all pipeline data between stages |

### 3.5 Component Dependency Graph

```
__main__.py
  ├── interface/tui/app.py ← BantzApp
  │     ├── core/brain.py ← Brain (central orchestrator)
  │     │     ├── core/routing_engine.py ← quick_route, dispatch_internal
  │     │     ├── core/intent.py ← cot_route (CoT LLM routing)
  │     │     ├── core/memory_injector.py ← inject (context enrichment)
  │     │     │     └── memory/omni_memory.py ← OmniMemoryManager
  │     │     │           ├── memory/graph.py ← GraphMemory (Neo4j)
  │     │     │           ├── memory/vector_store.py ← VectorStore (SQLite)
  │     │     │           └── memory/deep_probe.py ← DeepMemoryProbe
  │     │     ├── core/prompt_builder.py ← build_chat_system
  │     │     ├── core/finalizer.py ← finalize, hallucination_check
  │     │     ├── core/translation_layer.py ← to_en, detect_feedback
  │     │     │     └── i18n/bridge.py ← LanguageBridge (MarianMT)
  │     │     ├── llm/ollama.py ← OllamaClient
  │     │     ├── llm/gemini.py ← GeminiClient
  │     │     ├── tools/* ← All registered tool implementations
  │     │     └── core/rl_hooks.py → agent/affinity_engine.py
  │     ├── agent/* (background workers started by TUI)
  │     └── core/event_bus.py ← EventBus (cross-cutting)
  ├── interface/telegram_bot.py ← run_bot
  └── data/layer.py ← DataLayer (init all stores)
        ├── data/connection_pool.py ← SQLitePool
        ├── data/sqlite_store.py ← All SQLite stores
        └── data/json_store.py ← JSON fallback stores
```

---

## 4. Capabilities Implemented

### 4.1 Fully Implemented Features

| Feature | Components | Maturity |
|---------|-----------|----------|
| **TUI (Textual)** | `interface/tui/` — chat, system panel, header, toasts, mood | Production |
| **Telegram Bot** | `interface/telegram_bot.py` — 13+ commands, streaming replies, photo handling | Production |
| **LLM Chat** | `llm/ollama.py`, `llm/gemini.py` — dual backend with streaming | Production |
| **Shell Command Execution** | `tools/shell.py` — sandboxed with destructive/blocked detection | Production |
| **Gmail Integration** | `tools/gmail.py` — 17 actions, natural language queries, contact resolution | Production |
| **Google Calendar** | `tools/calendar.py` — CRUD, conflict detection, recurring events | Production |
| **Google Classroom** | `tools/classroom.py` — assignments, announcements, courses | Production |
| **Weather** | `tools/weather.py` — wttr.in with auto city detection | Production |
| **News** | `tools/news.py` — Hacker News + Google News RSS | Production |
| **Web Search** | `tools/web_search.py` — DuckDuckGo with cache + follow-up | Production |
| **Web Reader** | `tools/web_reader.py` — HTML→text extraction | Production |
| **Document Reader** | `tools/document.py` — PDF, DOCX, TXT, CSV + LLM Q&A | Production |
| **Reminders** | `tools/reminder.py` + `core/scheduler.py` — one-shot, recurring, location-triggered | Production |
| **Conversation Memory** | `core/memory.py` — FTS5 + vector hybrid search | Production |
| **Session Distillation** | `memory/distiller.py` — LLM summarization of sessions | Production |
| **Neo4j Graph Memory** | `memory/graph.py` — 11 entity types, 14 relationship types | Production |
| **Vector Embeddings** | `memory/embeddings.py` + `memory/vector_store.py` — Ollama nomic-embed-text | Production |
| **Deep Memory Probe** | `memory/deep_probe.py` — spontaneous recall with time-decay | Production |
| **Omni Memory Manager** | `memory/omni_memory.py` — budget-aware parallel hybrid search | Production |
| **Turkish↔English Translation** | `i18n/bridge.py` — MarianMT offline | Production |
| **Chain-of-Thought Routing** | `core/intent.py` — streaming thinking + JSON routing | Production |
| **Multi-Step Planning** | `agent/planner.py` + `agent/executor.py` — decompose + execute + replan | Production |
| **Morning Briefing** | `core/briefing.py` — parallel data gathering | Production |
| **Daily/Weekly Digests** | `core/digest.py` — LLM-synthesized summaries | Production |
| **Nightly Maintenance** | `agent/workflows/maintenance.py` — 6-step automated workflow | Production |
| **Nightly Reflection** | `agent/workflows/reflection.py` — LLM reflection + entity extraction | Production |
| **Overnight Polling** | `agent/workflows/overnight_poll.py` — Gmail/Calendar/Classroom | Production |
| **University Schedule** | `core/schedule.py` — weekly timetable management | Production |
| **Location Service** | `core/location.py` — 7-source priority chain | Production |
| **GPS Server** | `core/gps_server.py` — LAN HTTP + ntfy.sh relay | Production |
| **Named Places** | `core/places.py` — geofence detection, travel hints | Production |
| **Dynamic Persona** | `personality/persona.py` — 7 states from system metrics | Production |
| **Bonding Meter** | `personality/bonding.py` — sigmoid formality adaptation | Production |
| **Affinity Engine** | `agent/affinity_engine.py` — cumulative reward scoring | Production |
| **Interventions** | `agent/interventions.py` — priority queue with rate limiting | Production |
| **Health Monitoring** | `agent/health.py` — 5 rules (late night, marathon, eye strain, thermal, music) | Production |
| **Desktop Notifications** | `agent/notifier.py` — `notify-send` integration | Production |
| **APScheduler Jobs** | `agent/job_scheduler.py` — cron/interval with SQLAlchemy persistence | Production |
| **Hardware Telemetry** | `tui/telemetry.py` — CPU, RAM, disk, net, GPU, temps | Production |
| **Mood State Machine** | `tui/mood.py` — 5 states with hysteresis | Production |
| **Event Bus** | `core/event_bus.py` — thread-safe async pub/sub | Production |
| **Google OAuth** | `auth/` — interactive setup + secure token persistence | Production |
| **Data Layer** | `data/` — SQLite pool, abstract stores, JSON→SQLite migration | Production |

### 4.2 Partially Implemented / Has TODOs

| Feature | Status | Notes |
|---------|--------|-------|
| **TTS (Piper)** | Functional | Depends on external `piper` binary being installed; gain/animatronic filter via SoX optional |
| **Wake Word (Porcupine)** | Functional | Requires Picovoice API key (`BANTZ_PICOVOICE_ACCESS_KEY=`) |
| **STT (faster-whisper)** | Functional | Requires `faster-whisper`, `webrtcvad`, `pyaudio` — separate install |
| **Ghost Loop** | Functional | Full pipeline works but depends on all voice dependencies |
| **Ambient Analysis** | Functional | Simple RMS+ZCR classifier; no ML-based classification |
| **Proactive Engagement** | Functional | Conservative defaults (1/day), RL-adaptive scaling |
| **Computer Use / Autonomous Vision** | Experimental | Requires VLM + input control; safety gates in place but complex |
| **Browser Control** | Experimental | AT-SPI + VLM-based element finding; reliability depends on VLM quality |
| **Input Control** | Functional | Backend auto-detection (pyautogui/pynput/xdotool); safety classification |
| **Redis Session Store** | Optional | Falls back to in-memory; Redis never required |
| **Filesystem Tool** | Functional but limited | Sandboxed to `$HOME` only |
| `mail.py` | **Empty placeholder** | Unused — Gmail is in `gmail.py` |
| `gui_action.py` | **Deprecated** | Marked `#185`, not auto-registered; replaced by `visual_click.py` |

### 4.3 Not Yet Started / Missing

| Feature | Evidence |
|---------|---------|
| **Multi-user support** | All singletons assume single user; `TELEGRAM_ALLOWED_USERS` only whitelists |
| **Encryption at rest** | SQLite DB and tokens stored plaintext (tokens have `0o600` perms) |
| **Plugin system** | Tools are registered at Brain init; no hot-loading or third-party plugin API |
| **Web UI** | Only TUI and Telegram; no browser-based interface |
| **Persistent Redis requirement** | Always falls back to memory; no mandatory Redis deployment |
| **Rate limiting on LLM calls** | No token budget or cost tracking for Ollama/Gemini |
| **Structured logging** | Uses stdlib `logging` throughout; no structured JSON logging |
| **Metrics/observability** | No Prometheus/OpenTelemetry integration |

---

## 5. External Dependencies

### 5.1 Python Packages (from `pyproject.toml`)

#### Core Dependencies
| Package | Purpose |
|---------|---------|
| `textual>=0.59.0` | Terminal User Interface framework |
| `httpx>=0.27.0` | Async HTTP client (Ollama, Gemini, weather, news, web search) |
| `aiosqlite>=0.20.0` | Async SQLite driver |
| `pydantic>=2.0.0` | Data validation and models |
| `pydantic-settings>=2.0.0` | Environment-variable configuration |
| `python-dotenv>=1.0.0` | `.env` file loading |
| `psutil>=5.9.0` | System metrics (CPU, RAM, disk, process management) |
| `pynvml>=11.5.0` | NVIDIA GPU monitoring |
| `rich>=13.0.0` | Rich text rendering (used by Textual) |
| `apscheduler>=3.10.0` | Job scheduling (cron, interval, one-shot) |
| `sqlalchemy>=2.0.0` | APScheduler persistent job store backend |

#### Optional: Translation
| Package | Purpose |
|---------|---------|
| `transformers>=4.40.0` | MarianMT model loading/inference |
| `torch>=2.0.0` | PyTorch backend for transformers |
| `sentencepiece>=0.2.0` | Tokenizer for MarianMT |

#### Optional: Development
| Package | Purpose |
|---------|---------|
| `pytest>=8.0.0` | Test framework |
| `pytest-asyncio>=0.23.0` | Async test support |
| `pytest-cov>=5.0.0` | Coverage reporting |
| `ruff>=0.4.0` | Linter + formatter |

#### Optional: Document Reading
| Package | Purpose |
|---------|---------|
| `pymupdf>=1.24.0` | PDF text extraction (fitz) |
| `python-docx>=1.0.0` | DOCX reading |

#### Optional: Graph Memory
| Package | Purpose |
|---------|---------|
| `neo4j>=5.0.0` | Neo4j Python driver |

#### Optional: Desktop Automation
| Package | Purpose |
|---------|---------|
| `pyautogui>=0.9.54` | Mouse/keyboard automation (X11) |
| `pynput>=1.7.0` | Mouse/keyboard automation (Wayland) |

#### Implied Runtime Dependencies (not in pyproject.toml but used)
| Package / Tool | Purpose |
|----------------|---------|
| `python-telegram-bot` | Telegram bot framework |
| `google-api-python-client` | Google API client (Gmail, Calendar, Classroom) |
| `google-auth-oauthlib` | Google OAuth2 flow |
| `faster-whisper` | Local STT inference |
| `webrtcvad` | Voice Activity Detection |
| `pyaudio` | Microphone access |
| `pvporcupine` | Picovoice wake word |
| `piper` (binary) | TTS synthesis |
| `Pillow` | Image processing (screenshots, JPEG encoding) |
| `numpy` | Audio array conversion for STT |
| `redis` | Optional session store |
| `pdfplumber` | Alternative PDF table extraction |
| `sox` (binary) | Audio filter/gain for TTS |
| `xdotool`, `wmctrl`, `xprop` | X11 window management |
| `grim`, `scrot` | Screenshot capture |
| `notify-send` | Desktop notifications |
| `pup` (binary) | HTML CSS selector (browser tool) |
| `readability-cli` (npm) | Article extraction |

### 5.2 Environment Variables (from `.env.example`)

| Variable | Default | Purpose |
|----------|---------|---------|
| **Ollama** | | |
| `BANTZ_OLLAMA_MODEL` | `llama3.1:8b` | LLM model for chat/routing |
| `BANTZ_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| **Embeddings** | | |
| `BANTZ_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vector memory |
| `BANTZ_EMBEDDING_ENABLED` | `true` | Enable/disable vector embeddings |
| `BANTZ_VECTOR_SEARCH_WEIGHT` | `0.5` | Weight for vector vs FTS in hybrid search |
| **Distillation** | | |
| `BANTZ_DISTILLATION_ENABLED` | `true` | Auto-summarize completed sessions |
| `BANTZ_DISTILLATION_MIN_EXCHANGES` | `5` | Minimum exchanges before distilling |
| **Vision/VLM** | | |
| `BANTZ_VLM_ENABLED` | `false` | Enable remote VLM for screen analysis |
| `BANTZ_VLM_ENDPOINT` | `http://localhost:8090` | VLM server URL |
| `BANTZ_VLM_TIMEOUT` | `5` | VLM request timeout (seconds) |
| `BANTZ_SCREENSHOT_QUALITY` | `70` | JPEG quality for screenshots |
| **Input Control** | | |
| `BANTZ_INPUT_CONTROL_ENABLED` | `false` | Enable mouse/keyboard automation |
| `BANTZ_INPUT_CONFIRM_DESTRUCTIVE` | `true` | Confirm dangerous input actions |
| `BANTZ_INPUT_TYPE_INTERVAL_MS` | `50` | Typing speed (ms between chars) |
| **Gemini** | | |
| `BANTZ_GEMINI_ENABLED` | `false` | Enable Google Gemini as finalizer |
| `BANTZ_GEMINI_API_KEY` | *(empty)* | Gemini API key |
| `BANTZ_GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| **Language** | | |
| `BANTZ_LANGUAGE` | `tr` | Primary language |
| `BANTZ_TRANSLATION_ENABLED` | `true` | Enable MarianMT translation |
| **Shell** | | |
| `BANTZ_SHELL_CONFIRM_DESTRUCTIVE` | `true` | Confirm destructive shell commands |
| `BANTZ_SHELL_TIMEOUT_SECONDS` | `30` | Shell command timeout |
| **Location** | | |
| `BANTZ_CITY` | *(empty)* | Manual city override |
| `BANTZ_REGION` | *(empty)* | Manual region |
| `BANTZ_COUNTRY` | `TR` | Country code |
| `BANTZ_TIMEZONE` | `Europe/Istanbul` | Timezone |
| `BANTZ_LAT` / `BANTZ_LON` | `0.0` | Manual coordinates |
| `BANTZ_GPS_RELAY_TOKEN` | *(empty)* | ntfy.sh relay channel token |
| **Neo4j** | | |
| `BANTZ_NEO4J_ENABLED` | `false` | Enable graph memory |
| `BANTZ_NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `BANTZ_NEO4J_USER` / `PASSWORD` | `neo4j`/`bantzpass` | Neo4j credentials |
| **Storage** | | |
| `BANTZ_DATA_DIR` | *(empty → ~/.local/share/bantz)* | Data directory override |
| **Briefing/Digests** | | |
| `BANTZ_MORNING_BRIEFING` | `true` | Enable morning briefing |
| `BANTZ_MORNING_HOUR` / `MINUTE` | `8` / `0` | Briefing time |
| `BANTZ_DAILY_DIGEST_ENABLED` | `true` | Enable daily digest |
| `BANTZ_DAILY_DIGEST_HOUR` | `20` | Digest time |
| `BANTZ_WEEKLY_DIGEST_*` | sunday, 20:00 | Weekly digest schedule |
| **Scheduler** | | |
| `BANTZ_REMINDER_CHECK_INTERVAL` | `30` | Reminder poll interval (seconds) |
| `BANTZ_JOB_SCHEDULER_ENABLED` | `true` | Enable APScheduler |
| `BANTZ_MAINTENANCE_HOUR` | `3` | Maintenance cron hour |
| `BANTZ_REFLECTION_HOUR` | `23` | Reflection cron hour |
| `BANTZ_OVERNIGHT_POLL_HOURS` | `0,2,4,6` | Overnight poll schedule |
| `BANTZ_URGENT_KEYWORDS` | `urgent,acil,...` | Email urgency keywords |
| **Telegram** | | |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Telegram bot token |
| `TELEGRAM_ALLOWED_USERS` | *(empty)* | Comma-separated allowed user IDs |
| `TELEGRAM_PROXY` | *(empty)* | HTTPS/SOCKS5 proxy URL |
| `TELEGRAM_LLM_MODE` | `true` | Enable LLM-powered responses |
| **Observer** | | |
| `BANTZ_OBSERVER_ENABLED` | `false` | Enable stderr observer |
| `BANTZ_OBSERVER_*` | *(various)* | Severity threshold, batch timing, dedup, LLM analysis |
| **RL/Bonding** | | |
| `BANTZ_RL_ENABLED` | `false` | Enable RL engine |
| `BANTZ_BONDING_ENABLED` | `true` | Enable formality adaptation |
| `BANTZ_BONDING_SIGMOID_RATE` | `0.04` | Sigmoid curvature |
| `BANTZ_BONDING_SIGMOID_MIDPOINT` | `25` | Sigmoid center point |
| **Interventions** | | |
| `BANTZ_INTERVENTION_RATE_LIMIT` | `3` | Max interventions per hour |
| `BANTZ_INTERVENTION_QUIET_MODE` | `false` | Suppress non-critical |
| `BANTZ_INTERVENTION_FOCUS_MODE` | `false` | Only HIGH+ priority |
| **App Detector** | | |
| `BANTZ_APP_DETECTOR_ENABLED` | `false` | Enable active window detection |
| **Notifications** | | |
| `BANTZ_DESKTOP_NOTIFICATIONS` | `true` | Enable notify-send |
| **Voice Pipeline** | | |
| `BANTZ_VOICE_ENABLED` | `false` | Master switch for all voice features |
| `BANTZ_TTS_ENABLED` | `false` | Text-to-speech |
| `BANTZ_TTS_MODEL` | `en_US-danny-low` | Piper voice model |
| `BANTZ_TTS_GAIN` | `12.0` | SoX volume boost (dB) |
| `BANTZ_AUDIO_DUCK_ENABLED` | `false` | Lower other app audio during TTS |
| `BANTZ_WAKE_WORD_ENABLED` | `false` | "Hey Bantz" detection |
| `BANTZ_PICOVOICE_ACCESS_KEY` | *(empty)* | Picovoice API key |
| `BANTZ_STT_ENABLED` / `GHOST_LOOP_ENABLED` | `false` | Speech-to-text + voice loop |
| `BANTZ_STT_MODEL` | `tiny` | Whisper model size |
| `BANTZ_AMBIENT_ENABLED` | `false` | Passive ambient classification |
| `BANTZ_PROACTIVE_ENABLED` | `false` | Idle conversation initiation |
| `BANTZ_HEALTH_ENABLED` | `false` | Health/break interventions |

---

## 6. Test Coverage

### 6.1 Test Directory Structure

```
tests/ (76 test files)
├── conftest.py          — Shared fixtures
├── agent/               — 17 test files
├── core/                — 20 test files
├── data/                — 3 test files
├── interface/           — 3 test files
├── memory/              — 6 test files
├── personality/         — 2 test files
├── scheduler/           — 1 test file
├── tools/               — 8 test files
├── tui/                 — 7 test files
├── vision/              — 5 test files
└── workflows/           — 3 test files
```

### 6.2 What IS Tested

| Area | Test Files | Coverage Quality |
|------|-----------|-----------------|
| Agent subsystems | `test_affinity_engine`, `test_ambient`, `test_ambient_integration`, `test_app_detector`, `test_audio_ducker`, `test_dynamic_replanning`, `test_executor`, `test_ghost_loop`, `test_health`, `test_interventions`, `test_job_scheduler`, `test_notifier`, `test_observer`, `test_planner`, `test_proactive`, `test_tts`, `test_wake_word` | Comprehensive |
| Core pipeline | `test_brain_integrations`, `test_context`, `test_core_stabilization`, `test_event_bus`, `test_intent`, `test_location_handler`, `test_loop_breaker`, `test_memory_injector`, `test_notification_manager`, `test_prompt_builder`, `test_regex_audit`, `test_rl_hooks`, `test_rlhf_feedback`, `test_router`, `test_routing_engine`, `test_shell`, `test_systemd`, `test_translation_layer`, `test_voice_master_switch`, `test_config_cli` | Good |
| Data layer | `test_async_executor`, `test_json_migration`, `test_store` | Moderate |
| Memory | `test_deep_probe`, `test_distiller`, `test_memory_manager`, `test_omni_manager`, `test_omni_memory`, `test_session_store`, `test_vector_memory` | Good |
| Interface | `test_telegram_llm`, `test_telegram_screenshot`, `test_telegram_spam` | Telegram only |
| Personality | `test_bonding`, `test_persona` | Moderate |
| Tools | `test_contact_resolver`, `test_filesystem`, `test_filesystem_autochain`, `test_gmail_autochain`, `test_system_tool`, `test_visual_click`, `test_web_reader`, `test_web_search` | Partial |
| TUI | `test_event_bridge`, `test_header`, `test_input_control`, `test_mood`, `test_streaming`, `test_telemetry`, `test_toast` | Good |
| Vision | `test_accessibility`, `test_computer_use`, `test_navigator`, `test_spatial_cache`, `test_vision` | Moderate |
| Workflows | `test_maintenance`, `test_overnight_poll`, `test_reflection` | Good |

### 6.3 What is NOT Tested

#### Entirely Untested Packages
- `llm/` — `gemini.py`, `ollama.py` (zero test files)
- `auth/` — `google_oauth.py`, `token_store.py` (zero test files)
- `i18n/` — `bridge.py` (zero test files)

#### Major Source Files Without Dedicated Tests
| File | Notes |
|------|-------|
| `core/brain.py` | Only integration test via `test_brain_integrations.py` |
| `core/briefing.py` | No test |
| `core/butler.py` | No test |
| `core/finalizer.py` | No test |
| `core/digest.py` | No test |
| `core/habits.py` | No test |
| `core/memory.py` | No test |
| `core/places.py` | No test |
| `core/profile.py` | No test |
| `core/schedule.py` | No test |
| `core/session.py` | No test |
| `core/time_context.py` | No test |
| `core/time_parser.py` | No test |
| `core/date_parser.py` | No test |
| `core/types.py` | No test |
| `core/workflow.py` | No test |
| `personality/greeting.py` | No test |
| `personality/system_prompt.py` | No test |
| `agent/stt.py` | No test |
| `agent/voice_capture.py` | No test |

#### Tools Without Dedicated Tests
`browser_control.py`, `calendar.py`, `classroom.py`, `gmail.py` (only autochain test), `news.py`, `weather.py`, `reminder.py`, `document.py`, `shell.py` (tested in core/), `summarizer.py`, `screenshot_tool.py`, `computer_use.py`, `accessibility.py` (tested in vision/), `contacts.py`, `input_control.py` (tested in tui/)

---

## 7. Known Issues / Technical Debt

### 7.1 Hardcoded Values

| Location | Value | Issue |
|----------|-------|-------|
| `core/gps_server.py` | `GPS_PORT = 9777` | Hardcoded port, not configurable |
| `core/places.py` | `GEOFENCE_RADIUS = 150` metres, `STATIONARY_TIMEOUT = 2700` (45 min) | Not env-configurable |
| `memory/deep_probe.py` | `rate_every_n = 3` | Fixed rate — maybe should be configurable |
| `memory/omni_memory.py` | `MAX_MEMORY_TOKENS = 400`, budget splits 35/40/25 | Fixed token budget |
| `tools/web_reader.py` | Max 6000 chars output | Hardcoded truncation limit |
| `tools/web_search.py` | Cache TTL 15 min, max 50 entries | Hardcoded cache parameters |
| `vision/spatial_cache.py` | `MAX_ENTRIES = 1000`, `TTL_HOURS = 24` | Hardcoded, not env-configurable |
| `agent/health.py` | `idle_threshold_s = 900` (15 min) | Hardcoded idle detection threshold |
| `core/finalizer.py` | `< 800 chars` verbatim threshold | Not configurable |
| `tui/telemetry.py` | 2-second collection interval | Hardcoded |

### 7.2 Architectural Concerns

| Issue | Details |
|-------|---------|
| **God-object Brain** | `core/brain.py` is the central orchestrator touching nearly every module. High coupling coefficient. |
| **Excessive singletons** | Almost every module creates a module-level singleton. Makes testing harder and prevents multi-instance deployments. |
| **Lazy imports everywhere** | Many modules use inline `try: import X except ImportError: X = None` patterns. Hard to trace actual dependency graph. |
| **Missing type annotations on some functions** | Some older functions lack return type annotations |
| **No dependency injection** | Components reach for global singletons rather than receiving dependencies through constructors |

### 7.3 Missing Error Handling

| Location | Issue |
|----------|-------|
| `llm/ollama.py`, `llm/gemini.py` | No retry logic on transient failures (only `is_available` health check) |
| `tools/gmail.py` | Google API quota errors not specifically handled |
| `core/gps_server.py` | HTTP server runs on daemon thread with basic exception handling |
| `memory/graph.py` | Neo4j connection drops handled via `enabled` property but no auto-reconnect |
| `agent/job_scheduler.py` | Job failures logged but no alerting beyond desktop notifications |

### 7.4 Incomplete Logic

| Location | Issue |
|----------|-------|
| `tools/mail.py` | **Completely empty** — dead placeholder file |
| `tools/gui_action.py` | Deprecated (#185), not registered, kept for reference |
| `core/router.py` | Legacy router superseded by `intent.py` but still importable |
| `memory/session_store.py` | Redis integration optional; `PubSub.subscribe()` is essentially a no-op in memory fallback mode |
| `scheduler/scheduler.py` | Redis job store attempted but falls back to memory — persisted jobs are lost on restart without Redis |

### 7.5 Security Considerations

| Issue | Severity | Details |
|-------|----------|---------|
| OAuth tokens on disk | Medium | Stored in `~/.local/share/bantz/tokens/` with `0o600` perms — no encryption at rest |
| `.env` file secrets | Medium | API keys, passwords in plaintext `.env` |
| Shell execution | Mitigated | Destructive detection + blocked list, but `shell_confirm_destructive` can be disabled |
| Input control | Mitigated | Safety classification (safe/moderate/dangerous) but can be overridden via config |
| SQLite without encryption | Low | Conversation history stored in plaintext SQLite |
| GPS relay via ntfy.sh | Low | Location data sent through third-party public relay service |

### 7.6 Performance Considerations

| Issue | Details |
|-------|---------|
| **Brute-force vector search** | `vector_store.py` loads all embeddings into memory for cosine similarity — O(n) per query. No indexing (no FAISS/annoy). |
| **Serial embedding** | `embeddings.py` `embed_batch()` processes texts sequentially, not in parallel |
| **Full graph scan on query miss** | `context_builder.py` keyword search falls back to property regex scan when fulltext index fails |
| **15 background workers in TUI** | `BantzApp.on_mount()` starts many `@work(thread=True)` workers; thread pool pressure on low-end hardware |
| **No connection pooling for httpx** | Each LLM/API call creates a new httpx client |
