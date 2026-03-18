# Issue #9 — [core] Migrate TUI from Textual to Rich Live + mouse support

**Labels:** `enhancement`, `core`, `refactor`

---

## Summary

Replace the current Textual-based TUI with a **Rich Live** layout. Textual's async/thread integration overhead and CSS re-render cost are incompatible with Bantz v3's requirements: high-frequency log streaming, live system panels, readable long-form text output, and mouse interaction — all running concurrently in a single asyncio loop.

---

## Motivation

Current pain points with Textual:
- `call_from_thread()` required for every external update (subprocess, tool output, Telegram events) — brittle and verbose
- Full widget tree re-render on state change — causes visible lag on log streams
- CSS-based layout is hard to debug and unpredictable under rapid updates
- Mouse support added recently but still coupled to Textual's event system

Rich Live advantages for this use case:
- Diff-only terminal rendering — only changed lines are redrawn, no full-screen flicker
- Direct `asyncio` compatibility — no thread bridging needed
- `rich.text.Text` with markup + scrollback handles long-form readable output natively
- Mouse events capturable via `readchar` or low-level `termios` without a widget framework
- Rich is already a dependency (Textual depends on it) — zero added weight

---

## Target Layout

```
┌─────────────────────────────────────────────────────┐
│ BANTZ // OPERATIONS CENTER          [sys] [neo4j] … │  ← header (2 lines)
├──────────────────────┬──────────────────────────────┤
│  SYSTEM              │  LOG STREAM                  │
│  CPU  ██░░░  18%     │  [14:02:31] BrowserTool OK   │
│  RAM  ████░  11/16GB │  [14:02:32] fetch https://…  │
│  VRAM ███░░  4/6GB   │  [14:02:33] chafa render OK  │
│  DISK ██░░░  234GB   │  > scrollable, mouse select  │
├──────────────────────┴──────────────────────────────┤
│  CHAT                                               │
│  Bantz › Good morning. 3 tasks pending.             │
│  > _                                                │  ← input line
└─────────────────────────────────────────────────────┘
```

---

## Acceptance Criteria

- [ ] `Layout` split into three regions: `header`, `main` (stats + logs side by side), `chat`
- [ ] `Live(layout, refresh_per_second=4)` — 4fps ceiling, no CPU spike
- [ ] Stats panel: CPU, RAM, VRAM, DISK via `psutil`, updated every 2s via async task
- [ ] Log panel: `asyncio.Queue` fed by all tool calls; newest entry at bottom; last 200 lines kept
- [ ] Log panel: **mouse scroll supported** via `readchar` — scroll up/down through history
- [ ] Log panel: **text selectable** — `xdotool` or raw terminal mouse mode (`\033[?1000h`) for copy
- [ ] Chat input: single `input()` line replaced by async `aioconsole.ainput()` — non-blocking
- [ ] Chat output: `rich.text.Text` with syntax highlighting for code blocks, word-wrap for long responses
- [ ] Long text (articles, summaries): rendered in log panel with `rich.markdown.Markdown` — readable, scrollable
- [ ] Header bar: live status indicators (Neo4j, Redis, Qwen, Gemini, Telegram) — green/red dots
- [ ] All Textual imports removed from codebase
- [ ] Existing tool integrations (`BrowserTool`, `SystemTool`, etc.) write to the shared `asyncio.Queue` — no interface-layer changes needed in tools themselves

---

## Implementation Plan

### Step 1 — Shared log queue (no UI change yet)
```python
# bantz/core/log_bus.py
import asyncio

log_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)

async def emit(msg: str):
    if log_queue.full():
        log_queue.get_nowait()  # drop oldest
    await log_queue.put(msg)
```
All tools call `await log_bus.emit(...)` — decoupled from UI entirely.

### Step 2 — Rich Live scaffold
```python
# bantz/ui/live_ui.py
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="chat", size=5),
    )
    layout["main"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="logs", ratio=2),
    )
    return layout
```

### Step 3 — Async update tasks
```python
async def update_stats(layout: Layout):
    while True:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        text = Text()
        text.append(f"CPU  {make_bar(cpu, 20)}  {cpu:.0f}%\n")
        text.append(f"RAM  {make_bar(ram.percent, 20)}  "
                    f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB\n")
        layout["stats"].update(Panel(text, title="system", box=box.SIMPLE))
        await asyncio.sleep(2)

async def stream_logs(layout: Layout, log_history: list[str]):
    while True:
        msg = await log_queue.get()
        log_history.append(msg)
        if len(log_history) > 200:
            log_history.pop(0)
        text = Text("\n".join(log_history[-scroll_offset:]))
        layout["logs"].update(Panel(text, title="log stream", box=box.SIMPLE))
```

### Step 4 — Mouse scroll for log panel
```python
# Enable terminal mouse reporting
import sys
sys.stdout.write("\033[?1000h")  # enable mouse click events
sys.stdout.flush()

# In input handler: detect scroll wheel (button 64=up, 65=down)
async def handle_mouse(layout, log_history, scroll_state):
    reader = asyncio.StreamReader()
    # ... read raw escape sequences, adjust scroll_state["offset"]
    # scroll up → show older logs, scroll down → back to latest
```

### Step 5 — Chat input (non-blocking)
```python
import aioconsole

async def run_chat(layout: Layout):
    while True:
        user_input = await aioconsole.ainput("")
        layout["chat"].update(Panel(f"ZD › {user_input}", box=box.SIMPLE))
        response = await agent.process(user_input)
        # Long text → Markdown render in log panel
        if len(response) > 300:
            await log_bus.emit(f"[response]\n{response}")
        else:
            layout["chat"].update(Panel(
                f"Bantz › {response}", box=box.SIMPLE
            ))
```

### Step 6 — Main entrypoint
```python
async def run():
    layout = make_layout()
    log_history = []
    scroll_state = {"offset": 40}

    async with asyncio.TaskGroup() as tg:
        tg.create_task(update_stats(layout))
        tg.create_task(stream_logs(layout, log_history))
        tg.create_task(run_chat(layout))
        tg.create_task(handle_mouse(layout, log_history, scroll_state))
        # ... other background tasks (scheduler, Telegram listener)

    with Live(layout, refresh_per_second=4, screen=True):
        await run()
```

---

## Migration Checklist (existing code)

- [ ] `bantz/ui/app.py` (Textual `App` subclass) → delete, replace with `bantz/ui/live_ui.py`
- [ ] All `self.query_one()` / `self.post_message()` calls → replace with `log_bus.emit()`
- [ ] `Worker` / `@work` decorators → replace with `asyncio.create_task()`
- [ ] Textual CSS files (`*.tcss`) → delete entirely
- [ ] `call_from_thread()` wrappers → remove, all paths now async-native

---

## Dependencies

```bash
pip install rich aioconsole psutil
# rich is already installed (Textual dep) — just needs aioconsole added
pip uninstall textual  # after migration complete
```

---

## Notes

- `refresh_per_second=4` is deliberate — 10+ fps causes noticeable CPU usage for no visual benefit on a terminal
- Mouse text selection: terminal emulators (Konsole, GNOME Terminal) handle this natively when mouse reporting is OFF — only enable `\033[?1000h` for scroll events, disable it for copy mode. Toggle with a keybind (e.g. `Ctrl+M`).
- Long article text: pipe through `readability-cli` first (Issue #1), then render via `rich.markdown.Markdown` in the log panel — this gives clean readable output with proper word-wrap and heading styles
- VRAM reading: `pynvml` or `nvidia-smi` subprocess — `psutil` doesn't cover GPU

---

## Related

- Issue #1 (BrowserTool — log output buraya akar)
- Issue #4 (SystemTool — subprocess output buraya akar)
- Issue #7 (Redis — Telegram events log_bus'a yazılır)
- Issue #8 (Scheduler — job status header bar'da görünür)

---

*Generated for: bantz-v3 · March 2026*
