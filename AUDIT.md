# Bantz v2 — System Audit

**Date:** 2026-06-25
**Method:** Full-tree read of `src/bantz/` by five parallel focused passes (control loop, memory/MemPalace, tools, planning+eval, safety), with every flagship bug re-verified directly against source and the installed `mempalace` package.
**Scope:** `src/bantz/` at current `main`. Citations are `file:line` + symbol. Each finding is marked **VERIFIED** (re-checked in code while writing this file) or **REPORTED** (located by audit pass, citation given).

> Status legend — **REAL**: implemented and working. **PARTIAL**: works in a narrow path / degraded. **STUB**: scaffolding, not reachable or not wired. **BUG**: actively broken (often silently).
> Severity — **P0**: silent data loss / safety hole / user-invisible failure. **P1**: dead feature or functional gap. **P2**: smell / maintenance hazard. **P3**: cleanup.

---

## 0. Executive summary

Bantz is a **single-shot router + dispatcher**, not a ReAct agent. The routing/selection layer is genuinely mature (a full eval harness and a written paper back it). What sits *around* selection — memory consolidation, the autonomy/safety dials, and post-selection execution recovery — is where the system is hollow or silently broken.

The three most important facts:

1. **The "4-layer MemPalace memory" reduces to plain L3 vector recall in practice.** The knowledge-graph write path executed nightly **throws and is swallowed** (M1), the orphan-vector prune targets a **table that does not exist** (M2), decay is declared and never applied (M5), and the identity/story layers never reach a prompt (M6). Memory grows unbounded with no quality signal and no eval.
2. **The safety dials are dead.** The autonomy `requires_confirm` flag is written and **read nowhere** (S1); `shell.is_destructive()` is **never called** (S2). Destructive-command confirmation rides entirely on the routing LLM's self-reported `risk_level` string. There is no cost/token ceiling anywhere (S3).
3. **Execution is open-loop.** After `tool.execute()` the result is never re-fed to the model; there is no retry, fallback, or recovery on the single-tool path (C1). This is the seam the closed-loop research targets.

Scorecard:

| Subsystem | Status | Headline |
|---|---|---|
| Routing / selection | **REAL** | Mature, eval-backed, paper written. Leave alone. |
| Control loop | **REAL** (single-shot) | No observe→re-decide; open-loop execution (C1). |
| Planning / executor | **REAL** (plan-once) | Sequential, one-shot replan; no cross-turn task state. |
| Tools | **REAL core / much dead scaffolding** | ~25 work; no arg-schema validation; silent ImportError swallow. |
| **Memory consolidation** | **BUG (P0)** | Nightly KG write + prune both broken & swallowed. |
| Memory retrieval | **PARTIAL** | L3 vector only; graph/decay/topic-boost mostly inert. |
| **Safety / autonomy** | **STUB / BUG (P0)** | Autonomy dial dead; destructive gate model-trusted; no cost cap. |
| Memory eval | **MISSING** | No relevance/recall measurement exists at all. |

---

## 1. Memory — the broken core (highest priority)

### M1 — Nightly KG entity-write throws `TypeError` and is silently swallowed — **P0, BUG, VERIFIED** — ✅ FIXED (`3774436`)
- **Evidence:** `agent/workflows/reflection.py:576` calls
  ```python
  kg.add_triple(subject=value, relation=f"is_{label.lower()}", obj=context or label)
  ```
  but the installed signature is `mempalace/knowledge_graph.py:237` `add_triple(self, subject, predicate, obj, ...)` — **there is no `relation` kwarg**. Every call raises `TypeError: add_triple() got an unexpected keyword argument 'relation'`, caught by the bare `except (ImportError, Exception)` at `reflection.py:581-583` which only `log.debug`s (invisible at default log level) and returns `0`.
- **Impact:** The nightly reflection job's entire entity-extraction-into-KG step is a no-op. The knowledge graph **never** receives LLM-extracted entities. Every "the assistant learned X about you overnight" claim is false. Combined with M6, the KG is fed only by the brittle regex extractor in `bridge.py`.
- **Fix:** Rename kwarg `relation=` → `predicate=` at `reflection.py:576-579`. Then **narrow the except** to log at `warning` (not `debug`) so future signature drift is visible. **Effort: 5 min + a smoke test that asserts `stored > 0`.**

### M2 — Orphan-vector prune deletes from a non-existent table — **P0, BUG, VERIFIED** — ✅ FIXED (`a8e6a5f`; real drawer eviction still open as M3)
- **Evidence:** `reflection.py:669` `DELETE FROM message_vectors WHERE message_id IN (...)`. Grep confirms **no `CREATE TABLE message_vectors` anywhere in `src/`** — message vectors live in ChromaDB, not SQLite. The statement raises (`no such table`), is caught at `:673`, and `vectors_deleted` is always `0`.
- **Impact:** The advertised orphan-vector cleanup is fiction. ChromaDB drawers are **never pruned** → unbounded vector-store growth (see M3). The raw `messages` prune at `:682` *does* run, so the system deletes the raw text it does **not** use for retrieval while keeping the drawer copy it **does** use, forever.
- **Fix:** Either (a) delete the phantom `message_vectors` block and implement real ChromaDB drawer pruning via the MemPalace API, or (b) if drawer pruning is out of scope now, remove the dead statement and log honestly that drawers are retained. **Effort: 15 min to remove; ~1 d to implement real drawer pruning.**

### M3 — Unbounded memory growth, no eviction — **P1, REPORTED**
- **Evidence:** `add_drawer` only upserts (`mempalace/miner.py:817`); nothing in the bantz integration deletes drawers. KG triples only grow (`invalidate()` exists at `knowledge_graph.py:328` but is **never called** by bantz). The only deletion that works is the >30d raw-`messages` prune (M2's neighbor).
- **Impact:** ChromaDB and the KG SQLite grow forever; retrieval quality and latency degrade with age; no backpressure.
- **Fix:** After M2, add a drawer-count cap or age-based eviction; wire `invalidate()` into reflection for superseded facts. **Effort: ~1–2 d (depends on M2).**

### M4 — `recall()` runs twice per tool turn — **P2, REPORTED**
- **Evidence:** `omni_memory.recall()` is invoked in the chat/routing context **and again** inside `_finalize`/`_finalize_stream` (`brain.py:1285`, `brain.py:1299`), with no memoization.
- **Impact:** Doubles memory-recall latency and token cost on every tool turn; two independent recalls can disagree (nondeterminism).
- **Fix:** Cache the recall result on the turn (e.g. on the `BantzContext` or a per-turn attribute) and reuse in finalize. **Effort: ~1 h.**

### M5 — Decay declared but never applied — **P1, VERIFIED**
- **Evidence:** `bridge.py:124` constructor param `decay_half_life_days: float = 30.0`, assigned at `:130` `self._decay_half_life = ...`, and **never read again** (grep: only those two lines). KG `access_count` only ever increments (`bridge.py:419`); never decremented, never time-decayed.
- **Impact:** No forgetting. Old + frequently-recalled facts dominate recall permanently; stale facts that were never `invalidate()`d are treated as current.
- **Fix:** Either implement time-decay in the deep-probe / KG importance multiplier, or delete the dead parameter and stop advertising decay. **Effort: 30 min to remove; ~0.5 d to implement.**

### M6 — Identity (L0) and Story (L1) layers never reach a prompt — **P1, REPORTED**
- **Evidence:** `bridge.wake_up_context()` (`bridge.py:331`) is defined but has **no caller** in any prompt path (grep). The chat system prompt is built only from `recall.combined` (`brain.py:1092/1285/1299`). Separately, `run_onboarding` is gated by `sys.stdin.isatty()` (`bridge.py:296`) so it **never runs under the systemd daemon** (no TTY) → `identity.txt` is empty in production unless seeded by hand.
- **Impact:** The "4-layer stack" is L3-only at inference time. L0/L1/L2 are dead weight in this integration.
- **Fix:** Call `wake_up_context()` in `_build_chat_system` (budget it like the rest), and provide a non-TTY onboarding seed path for the daemon. **Effort: ~0.5 d.**

### M7 — `Memory.hybrid_search` no longer hybridizes — **P2, REPORTED**
- **Evidence:** `core/memory.py:401` `hybrid_search` — its semantic half (`:380`) now just calls `palace_bridge.vector_context` (ChromaDB), and its FTS half searches the `messages` table that reflection prunes after 30d. It's now a fallback that duplicates the primary vector path.
- **Impact:** Misleading "hybrid" naming; dead-ish code path; the FTS side returns less over time as messages are pruned.
- **Fix:** Either restore a real lexical+vector blend or rename/retire it as a thin fallback with a comment. **Effort: ~0.5 d.**

### M8 — KG retrieval only fires on capitalized seeds — **P2, REPORTED**
- **Evidence:** `bridge.graph_search` (`bridge.py:387`) seeds from capitalized words `len>2` + `registry.extract_people_from_query`; **no seed → empty result** (`:398`).
- **Impact:** Lowercase / non-capitalized entity queries (common in Turkish and in casual typing) retrieve nothing from the graph — the graph silently contributes zero on a large class of queries.
- **Fix:** Add lemma/lowercase seeding and entity-alias lookup. **Effort: ~0.5 d.** (Lower priority — graph value is questionable until M1 is fixed and the KG actually has content.)

### M9 — No memory-quality evaluation exists — **P1, REPORTED**
- **Evidence:** Grep for `precision|recall@|mrr|ndcg|hit_rate|memory_quality` across `tests/` and `eval/` → **zero**. `tests/memory/*` assert truncation math and sort order only; `eval/` is routing/planner only.
- **Impact:** There is no evidence recall surfaces correct memories; correctness is assumed. Any memory change is unmeasurable.
- **Fix:** Build a planted-fact recall harness (reuse `eval/test_report_runner.py`'s DB+`~/.mempalace` snapshot/restore). **Effort: ~2–3 d** (this is also research infrastructure).

---

## 2. Safety / autonomy — dead dials (high priority)

### S1 — Autonomy dial (`requires_confirm`) is written and read nowhere — **P0, STUB, VERIFIED** — ✅ FIXED (`33ccb8f`)
- **Evidence:** Set at `intent.py:391/393/395/397` based on `config.autonomy` (`low`→always confirm, `absolute`→never, default→`risk=="destructive"`). Grep for read-sites across `src/` → **only those 4 write-sites, zero reads.** The live confirmation gate (`brain.py:834`) derives its own condition from `risk == "destructive" and config.shell_confirm_destructive`, never consulting `requires_confirm`.
- **Impact:** Setting **autonomy=low** ("confirm everything") has **no effect** — `safe`/`moderate` tools still execute unprompted. Setting **autonomy=absolute** also has no real effect. The dial is inert; the UI/Settings control is a placebo.
- **Fix:** In `brain.process()` at the confirmation gate (`:834`), read `plan.get("requires_confirm")` and require confirmation when it's `True`, OR delete the dead field and the UI control. Prefer wiring it. **Effort: ~0.5 d incl. test.**

### S2 — `shell.is_destructive()` is dead code; destructive gate is model-trusted — **P0, STUB, VERIFIED** — ✅ FIXED (`0b24686`)
- **Evidence:** `shell.py:45` `def is_destructive` — grep shows **only the definition, zero callers.** `ShellTool.execute()` runs the command once it passes `is_blocked()` (`shell.py:73`); the only deterministic guard that actually runs is `is_blocked()` (fork-bomb/wget/curl), which is bypassable (first-word `shlex` only — `bash -c "rm -rf ~"` evades it) and over-broad (substring match false-positives on "curl"). Whether `rm -rf` prompts for confirmation depends entirely on whether the routing LLM tagged the call `risk_level="destructive"`.
- **Impact:** A destructive shell command the model labels `"safe"` runs with no prompt. Safety is a function of an 8B/4B model's self-assessment, not a deterministic allow/deny list.
- **Fix:** Call `is_destructive(cmd)` inside `ShellTool.execute()` (or in the brain gate) and force confirmation independent of the model's `risk_level`. Harden `is_destructive` against wrapped invocations (`bash -c`, `env`, pipes). **Effort: ~0.5 d.**

### S3 — No cost / token / iteration budget — **P1, REPORTED**
- **Evidence:** No spend tracking, cumulative token budget, or rate limiter anywhere (grep for cost/budget/ratelimit → nothing operative). Real caps exist only for routing tokens (`intent.py:445` `num_predict=768`, `:426` thinking 512), routing retries (≤2), planner replan (1, `executor.py`), and the vision loop (`computer_use.py:162` `max_steps=15`). `web_research` is acknowledged able to make unbounded local-model calls.
- **Impact:** A runaway tool/plan or a deep research call has no global ceiling; on a memory-constrained box this stalls Ollama.
- **Fix:** Add a per-turn token/step budget threaded through `brain.process()` (needed anyway for the closed-loop work). **Effort: ~1 d.**

### S4 — Hallucination check skips the streaming and plan paths — **P1, REPORTED**
- **Evidence:** `finalizer.hallucination_check` (`finalizer.py:397-451`) runs in `finalize()` but **not** in `finalize_stream()` (`:150-221`) nor `finalize_plan()` (`:254-335`).
- **Impact:** Streamed responses (the common interactive path) and multi-step plan summaries are never checked for fabrication.
- **Fix:** Accumulate streamed text and run the check post-stream with a trailing warning event; add to plan finalize. **Effort: ~0.5 d.**

### S5 — No account-identity verification before calendar/gmail writes — **P1, REPORTED**
- **Evidence:** No code checks the authenticated Google identity against the expected account. CLAUDE.md documents the live bug (calendar token authenticates as `230291026@firat.edu.tr`, writes land on the wrong calendar silently).
- **Impact:** "Created OK but invisible" — writes to the wrong account with no warning.
- **Fix:** On gmail/calendar init, fetch the authenticated email and warn/refuse if it doesn't match the configured account. **Effort: ~0.5 d** (orthogonal to the token re-auth fix already noted in CLAUDE.md).

---

## 3. Control loop & execution

### C1 — Open-loop execution: tool result never re-fed to the model — **P1, REAL-by-design (research target), VERIFIED**
- **Evidence:** `brain.py:891-895` — `result = await tool.execute(**tool_args)`; on exception, a canned butler string is returned; on `ToolResult(success=False)`, `finalizer.py:128-131` returns the error verbatim and stops. The result reaches the LLM only as `FACTS` to *narrate* (`finalizer.py:162`), never to *re-decide*. No retry, no fallback, no observe→re-decide.
- **Impact:** The agent cannot recover from a recoverable tool failure on the single-tool path; it cannot even detect a plausible-but-wrong `success=True` result. This is the structural ceiling the closed-loop research measures.
- **Fix (research-gated, flag it):** Add a flag-gated `_execute_with_recovery` loop around `:891-902` that re-invokes `cot_route` with the observation in `tool_context`, capped at `config.tool_loop_max_steps` (default `1` = current behavior). **Effort: ~2.5–3 d.** *Do not ship enabled by default until evaluated.*

### C2 — `Brain` is a process-global singleton holding mutable per-conversation state — **P1, REPORTED**
- **Evidence:** `brain = Brain()` (`brain.py:1314`) holds `_last_messages`, `_last_tool_output`, `_last_draft`, `_turn_counter`, `_last_screen_description`, etc. (`brain.py:188-204`).
- **Impact:** Fine for the single-user desktop, but the **Telegram bot** serves multiple users through the same singleton → follow-up context (`[CONTEXT:...]` IDs, last tool output, drafts) leaks **across users**. A privacy/correctness hazard.
- **Fix:** Key the per-conversation state by conversation/user id, or instantiate per-session. **Effort: ~1–1.5 d.**

### C3 — Duplicate tool-alias/normalization logic in two files — **P2, REPORTED**
- **Evidence:** `brain.py:739-756` (`_TOOL_ALIASES`) and `intent.py` (`_ROUTER_TOOL_ALIASES` + `_extract_json` repairs) solve the same small-model-output problem with overlapping-but-not-identical maps.
- **Impact:** Two places to update; drift produces inconsistent normalization.
- **Fix:** Consolidate into one alias module imported by both. **Effort: ~0.5 d.**

---

## 4. Tools

### T1 — No argument-schema validation before execution — **P1, REPORTED**
- **Evidence:** `BaseTool.schema()` (`tools/__init__.py:32-38`) exposes only name/description/risk_level — no parameter schema. `tool_args` from the LLM is splatted directly: `await tool.execute(**tool_args)` (`brain.py:892`, `executor.py:292`). No `jsonschema`/signature inspection anywhere.
- **Impact:** Malformed/hallucinated args reach `execute()` and surface as ad-hoc per-tool errors or `TypeError`s; nothing validates intent vs schema pre-call.
- **Fix:** Add an optional `params_schema` to `BaseTool` and validate before dispatch; reject with a structured error the (future) loop can act on. **Effort: ~1 d.**

### T2 — Silent ImportError swallowing hides unregistered tools — **P2, VERIFIED (pattern)**
- **Evidence:** `brain.py:129-185` wraps optional tool imports in `try/except (ImportError, ModuleNotFoundError): pass`. A tool with a missing dependency never registers; the only trace is a later "Tool not found".
- **Impact:** A tool can silently vanish from the registry; hard to debug "tool does nothing".
- **Fix:** Log a `warning` on swallowed import (keep the graceful-degrade behavior, just make it visible). **Effort: 15 min.**

### T3 — Dead / unreachable tool code — **P2, VERIFIED**
- **Evidence:** `tools/computer_use.py` self-registers but is **not imported in `brain.py`'s load block** (grep confirmed) → never in the registry, unreachable. On-disk but unloaded: `browser_tool.py`, `gui_tool.py`, `gui_action.py` (its `register` is commented out, `gui_action.py:168`), `image_tool.py`, `contacts.py`, `feed_tool.py`, `contact_resolver.py`.
- **Impact:** Dead scaffolding inflates the apparent tool surface; confuses maintenance.
- **Fix:** Either wire them into the load block (if intended) or delete/quarantine. **Effort: ~0.5 d to triage.**

### T4 — `core/router.py` is dead code — **P2, VERIFIED**
- **Evidence:** Grep for importers → **none**. Superseded by `intent.cot_route`.
- **Fix:** Delete. **Effort: 5 min.**

### T5 — `core/context.py` (`BantzContext`) is a half-finished refactor — **P2, REPORTED**
- **Evidence:** Its docstring claims it threads through "every stage of the pipeline," but it's instantiated only in the chat paths (`brain.py:1089/1132`); routing/execution/finalize never use it. Many fields (`route`, `tool_*`, `needs_confirm`, `mark_complete`, `as_log_dict`) are never populated.
- **Fix:** Either complete the refactor (it would be the natural carrier for C1's loop state and M4's recall cache) or trim to what's used. **Effort: ~1 d if completing.**

---

## 5. Planning

### P1 — No cross-turn task state — **P1, REPORTED (by-design gap)**
- **Evidence:** `PlanExecutor.run`'s `context_store` lives only within one `run()` call (`executor.py:149`). Across turns only `_last_tool_output`/`_last_tool_name` (≤500 chars) persist. Each message re-enters routing fresh.
- **Impact:** No long-horizon task can span turns; "continue what we were doing" is unsupported.
- **Fix:** Out of scope for the current research; note as future work.

### P2 — Planner step count is prompt-capped, not code-capped — **P2, REPORTED**
- **Evidence:** `planner.py:242` "Maximum 4 steps" is prompt text only; validation (`planner.py:399-406`) drops unknown-tool steps but never truncates count. A model emitting 50 valid-tool steps gets 50 executed.
- **Fix:** Hard `steps[:N]` cap after parse. **Effort: 15 min.**

### P3 — Legacy `core/workflow.py` `WorkflowEngine` appears superseded — **P3, REPORTED**
- **Evidence:** Regex-split conjunction engine; no live caller found in the brain dispatch path. Superseded by the LLM planner.
- **Fix:** Confirm dead, then delete. **Effort: 15 min to confirm.**

---

## 6. What's actually solid (do not "fix")

- **Routing/selection stack** — `intent.cot_route` + regex fast-paths + JSON repair/normalization. Eval-backed (`eval/routing_eval.py`, 100 cases, ablations, caching), paper written (`paper/main.tex`). Mature.
- **Plan executor recovery primitives** — circuit breaker + `_FAILURE_MARKERS` false-success detection + one-shot replan (`executor.py:105-447`). REAL and the right building blocks to generalize for C1.
- **Anti-hallucination plumbing** — `finalizer.strip_internal`, FACTS grounding, `_investigate_stream` live-diagnostic grounding (`brain.py`), People-Pleaser `system_alert` guard. REAL.
- **Scheduled night jobs** — `overnight_poll`, `maintenance`, `reflection` are real APScheduler multi-step jobs (note: reflection's KG-write step is broken — M1).
- **Eval harness machinery** — `eval/test_report_runner.py` snapshot/restore of `bantz.db` + `~/.mempalace` is reusable for the memory + closed-loop evals.

---

## 7. Prioritized fix roadmap

**Wave 1 — silent P0 fixes — ✅ DONE (2026-06-25):**
1. ~~M1 — fix `add_triple(relation=` → `predicate=`, widen the except to `warning`.~~ `3774436`
2. ~~M2 — remove the phantom `message_vectors` delete; log drawer retention honestly.~~ `a8e6a5f`
3. ~~S1 — wire `requires_confirm` into the `brain.py:834` gate.~~ `33ccb8f`
4. ~~S2 — call `is_destructive()` in the shell gate, independent of model `risk_level`.~~ `0b24686`

**Wave 2 — observability & hygiene (≈1–1.5 days):**
5. T2 — log swallowed tool imports (15 min). 6. M4 — memoize per-turn recall (1 h). 7. P2 — hard step cap (15 min). 8. T4/P3 — delete dead `router.py` / confirm-delete `workflow.py` (30 min). 9. M5 — implement or remove decay (0.5 d). 10. S4 — hallucination check on stream/plan (0.5 d).

**Wave 3 — correctness & multi-user (≈3–4 days):**
11. C2 — fix Telegram cross-user state leak (1–1.5 d). 12. S5 — account-identity verification (0.5 d). 13. M6 — wire `wake_up_context` + non-TTY onboarding (0.5 d). 14. M3 — real drawer eviction (1–2 d). 15. T1 — tool arg-schema validation (1 d).

**Wave 4 — research enablement (≈5–7 days, separate track):**
16. M9 — planted-fact memory eval harness (2–3 d). 17. C1 — flag-gated `_execute_with_recovery` closed-loop (2.5–3 d, default off). 18. S3 — per-turn token/step budget (1 d, prerequisite for C1).

---

## 8. Verification log (claims re-checked while writing)

| Finding | Check run | Result |
|---|---|---|
| M1 | `add_triple` call kwargs vs mempalace signature | **Confirmed**: `relation=` passed; signature has `predicate`. Swallowed at `reflection.py:581`. |
| M2 | `grep "CREATE TABLE message_vectors"` in `src/` | **Confirmed**: none. Table does not exist. |
| M5 | `grep "_decay_half"` | **Confirmed**: assigned `bridge.py:130`, never read. |
| S1 | `grep "requires_confirm" src/` | **Confirmed**: 4 writes (`intent.py:391-397`), 0 reads. |
| S2 | `grep "is_destructive" src/` | **Confirmed**: definition only (`shell.py:45`), 0 callers. |
| T3 | `grep "computer_use" brain.py` | **Confirmed**: not imported. |
| T4 | grep importers of `core.router` | **Confirmed**: none. |
