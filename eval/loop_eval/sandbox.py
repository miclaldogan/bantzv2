"""Per-task sandbox bootstrap for loop_eval (issue #505).

Two verified contamination mechanisms would silently invalidate eval runs:

1. **C2b — shared conversation context.** The data layer opens ONE
   conversation at startup and never switches: ``data_layer.conversations
   .context(n=12)`` (brain.py) reads the same ``conversation_id`` for every
   session, so task B's prompt would contain task A's history. The primitive
   to fix it exists and is simply never called between tasks:
   ``new_session()`` (src/bantz/data/sqlite_store.py).

2. **ChromaDB quarantine.** A second process on the live ``~/.mempalace``
   reliably corrupts an HNSW segment (AUDIT.md M10). Eval must NEVER touch
   the live palace.

This module kills both:

- :func:`bootstrap` redirects every persistent path into a fresh temp root
  via environment variables — and it MUST run **before any bantz import**,
  because ``bantz.config`` constructs its pydantic-settings ``Config()``
  singleton at import time from ``os.environ``.
- A **hard guard** then imports ``bantz.config`` and refuses to run
  (``SystemExit``) unless every resolved path actually landed inside the
  temp root — defense against an import that snuck in before bootstrap or
  a stray ``.env`` override.
- :func:`new_task` gives per-task isolation: a fresh conversation row via
  ``new_session()`` AND a distinct ``session_key`` for ``brain.process``
  (belt and braces — the session key wipes Brain follow-up state, the new
  conversation row empties ``context()``).

Usage — in-process dev loop::

    import sandbox
    root = sandbox.bootstrap()              # BEFORE any bantz import
    from bantz.core.brain import brain
    key = sandbox.new_task("gmail_read_01.base")
    result = await brain.process(prompt, session_key=key)

Usage — subprocess-per-task (the runner's default): the child process calls
``bootstrap()`` first thing; each child gets its own temp root, so tasks
cannot see each other at all.

Pure stdlib until the guard runs (the guard performs the first bantz import
on purpose).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

#: Environment variables redirected into the sandbox temp root.
ENV_KEYS = (
    "BANTZ_DATA_DIR",
    "BANTZ_PALACE_PATH",
    "BANTZ_MEMPALACE_KG_PATH",
    "BANTZ_MEMPALACE_IDENTITY_PATH",
)

#: Live locations that a sandboxed run must never write to.
LIVE_ROOTS = (
    Path.home() / ".mempalace",
    Path.home() / ".local" / "share" / "bantz",
)

# Set by bootstrap() so repeated calls in one process are idempotent.
_ROOT: Path | None = None


class SandboxViolation(SystemExit):
    """Raised (and exits the process) when sandbox isolation is broken."""


def _bantz_modules_loaded() -> list[str]:
    return sorted(
        m for m in sys.modules if m == "bantz" or m.startswith("bantz.")
    )


def bootstrap(root: str | Path | None = None) -> Path:
    """Redirect all bantz persistence into a temp root. Idempotent.

    MUST be called before any ``bantz`` import: ``bantz.config`` reads
    ``os.environ`` when the module-level ``Config()`` is constructed, so a
    prior import freezes the LIVE paths — in that case this trips the guard
    and exits rather than silently contaminating ``~/.mempalace``.

    Returns the sandbox root directory.
    """
    global _ROOT
    if _ROOT is not None:
        _assert_sandboxed(_ROOT)  # re-verify, cheap
        return _ROOT

    loaded = _bantz_modules_loaded()
    if loaded:
        raise SandboxViolation(
            "SANDBOX VIOLATION: bantz was imported before sandbox.bootstrap() "
            f"— Config() already resolved LIVE paths. Modules: {loaded[:8]}. "
            "Fix the import order: bootstrap() must be the first thing the "
            "task process runs."
        )

    root_path = Path(root) if root is not None else Path(
        tempfile.mkdtemp(prefix="bantz_loop_eval_")
    )
    root_path.mkdir(parents=True, exist_ok=True)
    data_dir = root_path / "data"
    data_dir.mkdir(exist_ok=True)

    os.environ["BANTZ_DATA_DIR"] = str(data_dir)
    os.environ["BANTZ_PALACE_PATH"] = str(root_path / "palace")
    os.environ["BANTZ_MEMPALACE_KG_PATH"] = str(
        root_path / "knowledge_graph.sqlite3"
    )
    os.environ["BANTZ_MEMPALACE_IDENTITY_PATH"] = str(root_path / "identity.txt")

    _assert_sandboxed(root_path)
    _ROOT = root_path
    return root_path


def _assert_sandboxed(root: Path) -> None:
    """Hard guard: every resolved bantz path must live under *root*.

    Performs the first ``bantz.config`` import. Exits loudly on violation —
    a run that could touch the live palace must never proceed.
    """
    from bantz.config import config  # first bantz import happens HERE

    root = root.resolve()
    resolved = {
        "db_path (BANTZ_DATA_DIR)": Path(config.db_path),
        "resolved_palace_path": Path(config.resolved_palace_path),
        "resolved_kg_path": Path(config.resolved_kg_path),
        "resolved_identity_path": Path(config.resolved_identity_path),
    }
    offenders = [
        f"{name} -> {path}"
        for name, path in resolved.items()
        if not path.resolve().is_relative_to(root)
    ]
    if offenders:
        raise SandboxViolation(
            "SANDBOX VIOLATION: resolved path(s) escape the sandbox root "
            f"{root}: {'; '.join(offenders)}. Refusing to run — this would "
            "write to live data (AUDIT M10 / C2b)."
        )


def sandbox_root() -> Path:
    """The active sandbox root; raises if bootstrap() has not run."""
    if _ROOT is None:
        raise SandboxViolation(
            "sandbox_root() called before bootstrap() — no sandbox is active."
        )
    return _ROOT


# ── per-task reset ────────────────────────────────────────────────────────────

def new_task(task_id: str) -> str:
    """Reset per-task state; returns the ``session_key`` for ``brain.process``.

    Belt and braces against C2b:
    - ``new_session()`` opens a fresh conversation row, so
      ``data_layer.conversations.context(n=12)`` starts empty;
    - the distinct returned session key makes ``Brain.process`` wipe its
      follow-up state (``_state_owner`` ownership check).

    Initializes the data layer on first call (inside the sandbox paths).
    """
    _assert_sandboxed(sandbox_root())

    from bantz.config import config
    from bantz.data.layer import data_layer

    data_layer.init(config)  # no-op if already initialized
    data_layer.conversations.new_session()
    return f"eval:{task_id}"


# ── live-write detection (acceptance: "zero bytes under live roots") ─────────

def live_marker() -> float:
    """Timestamp marker; pair with :func:`assert_no_live_writes`."""
    return time.time()


def live_writes_since(marker: float) -> list[str]:
    """Files under the live roots modified/created after *marker*."""
    dirty: list[str] = []
    for live in LIVE_ROOTS:
        if not live.exists():
            continue
        for p in live.rglob("*"):
            try:
                if p.is_file() and p.stat().st_mtime > marker:
                    dirty.append(str(p))
            except OSError:
                continue  # racing deletion — not a write by us
    return dirty


def assert_no_live_writes(marker: float) -> None:
    """Exit loudly if anything under the live roots changed since *marker*."""
    dirty = live_writes_since(marker)
    if dirty:
        raise SandboxViolation(
            "SANDBOX VIOLATION: live files were written during a sandboxed "
            f"run: {dirty[:10]}"
        )
