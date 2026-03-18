# Bantz v3 — GitHub Issues

> Tüm issue'lar `enhancement` ve ilgili katman label'ı ile açılabilir.
> Label önerileri: `layer:web`, `layer:system`, `layer:memory`, `layer:scheduler`, `core`, `good first issue`

---

## Issue #1 — [layer:web] Implement BrowserTool: curl + pup + readability pipeline

**Labels:** `enhancement`, `layer:web`, `core`

### Summary
Build the core `BrowserTool` class that replaces all outbound HTTP API calls with terminal-native equivalents. The agent should fetch, parse, and return clean content from any URL using only local CLI tools — no API keys, no rate limits.

### Motivation
Current weather/news tools depend on third-party APIs (rate-limited, key-dependent). A subprocess-based pipeline using `curl`, `pup`, and `readability-cli` gives the agent the same capability with zero external dependencies and full offline resilience.

### Acceptance Criteria
- [ ] `BrowserTool.fetch(url)` → returns raw HTML via `curl -sL`
- [ ] `BrowserTool.extract_text(url)` → pipes through `readability-cli`, returns article body as plain text
- [ ] `BrowserTool.query(url, css_selector)` → uses `pup` to extract elements by CSS selector
- [ ] `BrowserTool.extract_images(url)` → returns list of absolute image URLs via `pup 'img attr{src}'`
- [ ] All methods raise a typed `BrowserToolError` on non-zero exit codes
- [ ] Unit tests cover: valid page, 404, network timeout, malformed HTML

### Implementation Notes
```python
# tools/browser_tool.py
import subprocess, shlex

class BrowserTool:
    def _run(self, cmd: str) -> str:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            raise BrowserToolError(result.stderr)
        return result.stdout

    def fetch(self, url: str) -> str:
        return self._run(f"curl -sL --max-time 12 {url}")

    def extract_text(self, url: str) -> str:
        return self._run(f"readability-cli {url}")

    def query(self, url: str, selector: str) -> str:
        html = self.fetch(url)
        proc = subprocess.run(
            ["pup", selector],
            input=html, capture_output=True, text=True
        )
        return proc.stdout
```

### Dependencies
```bash
sudo apt install curl
npm install -g @mozilla/readability-cli
pip install pup  # or: go install github.com/ericchiang/pup@latest
```

### Related
- Issue #2 (RSS feed parser)
- Issue #3 (chafa image renderer)

---

## Issue #2 — [layer:web] RSS/Atom feed parser via xmllint (no API)

**Labels:** `enhancement`, `layer:web`

### Summary
Implement `FeedTool` that fetches and parses RSS/Atom feeds using `curl` + `xmllint` — replacing any news API dependency with direct feed consumption.

### Motivation
Most news sources (BBC, HackerNews, Reddit, NTV, Habertürk) expose RSS feeds. Parsing them with `xmllint` is faster, more stable, and completely free compared to a news aggregator API. Images (`<media:content>` or `<enclosure>`) are extractable from the same feed.

### Acceptance Criteria
- [ ] `FeedTool.fetch(feed_url)` → returns list of `FeedItem(title, link, summary, image_url, published_at)`
- [ ] Supports both RSS 2.0 and Atom 1.0 formats
- [ ] `image_url` extracted from `<media:content url=...>` or `<enclosure url=...>` when present
- [ ] Results sorted by `published_at` descending
- [ ] Feed URL registry in `config/feeds.yaml` (editable without code changes)
- [ ] Gracefully handles feeds with missing fields (no crash on empty `<description>`)

### Implementation Notes
```yaml
# config/feeds.yaml
feeds:
  tech:
    - name: Hacker News
      url: https://hnrss.org/frontpage
    - name: The Verge
      url: https://www.theverge.com/rss/index.xml
  tr_news:
    - name: NTV
      url: https://www.ntv.com.tr/son-dakika.rss
```

```python
# Parse with xmllint
result = subprocess.run(
    ["xmllint", "--xpath", "//item/title/text()", "-"],
    input=raw_xml, capture_output=True, text=True
)
```

### Related
- Issue #1 (BrowserTool base)
- Issue #3 (image rendering)

---

## Issue #3 — [layer:web] Terminal image rendering via chafa (TUI + Telegram fallback)

**Labels:** `enhancement`, `layer:web`

### Summary
Add image rendering capability to Bantz. In the Textual TUI, images should render inline using `chafa`. For Telegram, the raw image file should be sent directly. The agent decides which path to use based on the active interface context.

### Motivation
News articles, weather maps, and scraped content include images. Without a rendering layer, the agent discards visual information entirely. `chafa` converts any image URL/file to terminal-compatible Unicode art with configurable quality.

### Acceptance Criteria
- [ ] `ImageTool.render_terminal(url_or_path)` → downloads image, pipes to `chafa`, returns ANSI string
- [ ] `ImageTool.download(url)` → saves to `~/.bantz/cache/images/` with content-hash filename (deduplication)
- [ ] Cache TTL: 24h (stale images purged on startup)
- [ ] Textual TUI: rendered via `Static` widget accepting ANSI markup
- [ ] Telegram path: sends raw file via `bot.send_photo()`
- [ ] `chafa` quality flag configurable in `config.yaml` (`--size`, `--colors`)
- [ ] Graceful fallback: if `chafa` not installed, log warning and skip render (no crash)

### Implementation Notes
```python
# tools/image_tool.py
def render_terminal(self, source: str) -> str:
    path = self.download(source) if source.startswith("http") else source
    result = subprocess.run(
        ["chafa", "--size", "60x30", "--colors", "256", str(path)],
        capture_output=True, text=True
    )
    return result.stdout  # ANSI art string → pass to Textual Static()
```

### Dependencies
```bash
sudo apt install chafa
```

### Related
- Issue #1, #2

---

## Issue #4 — [layer:system] SystemTool: unified subprocess + process management interface

**Labels:** `enhancement`, `layer:system`, `core`

### Summary
Build `SystemTool` — the agent's primary interface for running shell commands, managing processes, and interacting with the OS. This is the backbone of all automation capabilities.

### Motivation
Ad-hoc `subprocess.run()` calls scattered across the codebase are untestable and unsafe. A unified `SystemTool` with timeout enforcement, output sanitization, and permission gating makes system access auditable and controllable.

### Acceptance Criteria
- [ ] `SystemTool.run(cmd, timeout=30, safe_mode=True)` → executes shell command, returns `ShellResult(stdout, stderr, returncode, duration_ms)`
- [ ] `safe_mode=True` blocks commands matching a denylist (e.g. `rm -rf /`, `dd if=`, `:(){ ... }`)
- [ ] `SystemTool.list_processes()` → returns running processes via `psutil` (name, pid, cpu%, mem%)
- [ ] `SystemTool.kill(pid)` → terminates process by pid with confirmation log
- [ ] `SystemTool.open_app(app_name)` → resolves app binary and launches detached
- [ ] All commands logged to `~/.bantz/logs/system_audit.log` with timestamp + caller
- [ ] Unit tests: safe_mode blocks, timeout fires, audit log written

### Implementation Notes
```python
DENYLIST = [r"rm\s+-rf\s+/", r"dd\s+if=", r"mkfs", r">\s+/dev/sd"]

def run(self, cmd: str, timeout: int = 30, safe_mode: bool = True) -> ShellResult:
    if safe_mode:
        for pattern in DENYLIST:
            if re.search(pattern, cmd):
                raise DangerousCommandError(f"Blocked: {cmd}")
    start = time.monotonic()
    proc = subprocess.run(cmd, shell=True, capture_output=True,
                          text=True, timeout=timeout)
    duration = int((time.monotonic() - start) * 1000)
    self._audit_log(cmd, proc.returncode, duration)
    return ShellResult(proc.stdout, proc.stderr, proc.returncode, duration)
```

### Related
- Issue #5 (pyautogui GUI automation)
- Issue #8 (scheduler uses SystemTool)

---

## Issue #5 — [layer:system] GUI automation layer: pyautogui + xdotool bridge

**Labels:** `enhancement`, `layer:system`

### Summary
Implement `GUITool` wrapping `pyautogui` (cross-platform) and `xdotool` (X11-native, more reliable on Linux Mint) for mouse/keyboard automation. This enables the agent to control any desktop application.

### Motivation
Some tasks cannot be automated via CLI (e.g. interacting with a GUI app that has no API). `pyautogui` + `xdotool` gives the agent full desktop control: clicking, typing, window focus, screenshot-based verification.

### Acceptance Criteria
- [ ] `GUITool.click(x, y)` and `GUITool.click_image(template_path, confidence=0.9)` (screenshot-based)
- [ ] `GUITool.type(text, interval=0.05)` → types text with realistic keystroke delay
- [ ] `GUITool.focus_window(title_pattern)` → uses `xdotool search --name` to focus by title
- [ ] `GUITool.screenshot(region=None)` → saves to cache, returns path
- [ ] `GUITool.scroll(x, y, clicks)` → mouse scroll at position
- [ ] All actions preceded by a 300ms safety delay (prevents runaway automation)
- [ ] `DRY_RUN` env flag: logs intended actions without executing (for testing)

### Implementation Notes
```python
# xdotool preferred on Linux, pyautogui as fallback
def focus_window(self, title: str):
    try:
        subprocess.run(["xdotool", "search", "--name", title,
                        "windowactivate", "--sync"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # fallback: pyautogui has no window focus, warn and skip
        logger.warning("xdotool not available, window focus skipped")
```

### Dependencies
```bash
pip install pyautogui pillow
sudo apt install xdotool
```

### Related
- Issue #4 (SystemTool — command execution)

---

## Issue #6 — [layer:memory] Neo4j memory layer: entity graph + conversation context

**Labels:** `enhancement`, `layer:memory`, `core`

### Summary
Implement the `MemoryManager` class backed by Neo4j. Every conversation turn extracts entities and relations, stores them as graph nodes, and retrieves relevant context for subsequent queries.

### Motivation
Flat key-value memory (SQLite) loses relational context. Neo4j lets the agent answer questions like "what did I decide about X last week?" by traversing `(:Decision)-[:ABOUT]->(:Topic)` relationships — impossible with flat storage.

### Acceptance Criteria
- [ ] `MemoryManager.store(turn)` → extracts entities (NER via spaCy or LLM prompt) and writes nodes/relations to Neo4j
- [ ] `MemoryManager.query(text)` → returns top-k relevant memories via Cypher full-text search
- [ ] Node types: `Person`, `Project`, `Decision`, `Task`, `Event`, `Fact`
- [ ] Relation types: `ABOUT`, `DECIDED_BY`, `DEPENDS_ON`, `REFERENCES`, `HAPPENED_AT`
- [ ] `MemoryManager.summarize_context(topic)` → returns narrative summary for a given topic node
- [ ] Migrations handled via versioned Cypher scripts in `db/migrations/`
- [ ] Connection pool configured via `config.yaml` (uri, user, password)
- [ ] Unit tests use Neo4j test container (testcontainers-python)

### Schema Example
```cypher
// Store a decision
MERGE (d:Decision {id: $id})
SET d.text = $text, d.created_at = $ts
MERGE (t:Topic {name: $topic})
MERGE (d)-[:ABOUT]->(t)

// Query recent decisions about a topic
MATCH (d:Decision)-[:ABOUT]->(t:Topic {name: $topic})
RETURN d ORDER BY d.created_at DESC LIMIT 5
```

### Dependencies
```bash
pip install neo4j spacy
python -m spacy download en_core_web_sm
# docker-compose up neo4j (in infra/)
```

### Related
- Issue #7 (Redis session layer)

---

## Issue #7 — [layer:memory] Redis session store + task queue backbone

**Labels:** `enhancement`, `layer:memory`

### Summary
Add Redis as the short-term session store and task queue backbone. Redis handles in-flight conversation state, pending tool calls, and inter-process messaging between the TUI and Telegram interfaces.

### Motivation
Neo4j is for long-term memory; Redis is for "what's happening right now." Active sessions, pending async tasks, rate-limit counters, and pub/sub between the TUI process and Telegram bot process all need a fast in-memory store.

### Acceptance Criteria
- [ ] `SessionStore.set(session_id, data, ttl=3600)` / `.get(session_id)` via Redis hashes
- [ ] `TaskQueue.push(task)` / `.pop()` using Redis lists (LPUSH/BRPOP)
- [ ] Pub/sub channel `bantz:events` for TUI ↔ Telegram process communication
- [ ] `RateLimiter.check(key, limit, window_sec)` → sliding window counter
- [ ] Connection via `redis.asyncio` (async-native, compatible with Textual's event loop)
- [ ] Redis config in `config.yaml` (host, port, db index, password optional)
- [ ] Graceful degradation: if Redis unavailable, fall back to in-memory dict with warning

### Implementation Notes
```python
# bantz/memory/session_store.py
import redis.asyncio as aioredis

class SessionStore:
    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.r = aioredis.from_url(url, decode_responses=True)

    async def set(self, key: str, data: dict, ttl: int = 3600):
        await self.r.hset(key, mapping=data)
        await self.r.expire(key, ttl)

    async def get(self, key: str) -> dict | None:
        return await self.r.hgetall(key) or None
```

### Dependencies
```bash
pip install redis[asyncio]
sudo apt install redis-server  # or docker
```

### Related
- Issue #6 (Neo4j), Issue #8 (Scheduler uses queue)

---

## Issue #8 — [layer:scheduler] APScheduler integration: cron jobs + one-shot tasks

**Labels:** `enhancement`, `layer:scheduler`, `core`

### Summary
Integrate `APScheduler` as the primary task scheduler. The agent must be able to schedule both recurring jobs (daily summaries, morning briefings) and one-shot deferred tasks ("remind me at 14:00", "run this script tonight at 02:00").

### Motivation
Gece bir görev verip sabah hazır bulmak için scheduler şart. APScheduler is lightweight, async-native, and supports cron, interval, and date triggers — no separate daemon needed.

### Acceptance Criteria
- [ ] `Scheduler.add_cron(func, cron_expr, job_id)` → schedules recurring job
- [ ] `Scheduler.add_once(func, run_at: datetime, job_id)` → schedules one-shot job
- [ ] `Scheduler.add_interval(func, seconds, job_id)` → schedules interval job
- [ ] `Scheduler.list_jobs()` → returns list of `ScheduledJob(id, next_run, trigger_type)`
- [ ] `Scheduler.cancel(job_id)` → cancels by id
- [ ] Jobs persisted to Redis (via `RedisJobStore`) — survive process restarts
- [ ] Failed jobs logged with full traceback to `~/.bantz/logs/scheduler.log`
- [ ] Agent can create jobs via natural language → parsed to cron/datetime by LLM

### Implementation Notes
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.redis import RedisJobStore

jobstores = {
    "default": RedisJobStore(
        jobs_key="bantz:scheduler:jobs",
        run_times_key="bantz:scheduler:run_times",
        host="localhost", port=6379
    )
}

scheduler = AsyncIOScheduler(jobstores=jobstores)
scheduler.start()

# Natural language → schedule (LLM call)
# "her sabah 08:00'de haber özeti" → cron: "0 8 * * *"
```

### Dependencies
```bash
pip install apscheduler[redis]
```

### Related
- Issue #7 (Redis job store), Issue #4 (SystemTool runs scheduled commands)

---

## Quick Reference — Issue Dependency Graph

```
#1 BrowserTool ──► #2 FeedTool
      │
      └──► #3 ImageTool (chafa)

#4 SystemTool ──► #5 GUITool

#6 Neo4j Memory ──► (used by all tools for context)
#7 Redis Session ──► #8 APScheduler

Build order: #4 → #1 → #7 → #6 → #2 → #3 → #5 → #8
```

---

*Generated for: bantz-v3 · March 2026*
