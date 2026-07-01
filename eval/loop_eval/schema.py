"""loop_eval contract v1.0 — executable schema (issue #500).

The three-way contract between the task corpus (WS-B), the eval runner (WS-C),
and the analysis scripts (WS-D). SCHEMA.md in this directory is the prose form
and records the frozen decisions; this module is the enforcement:

    validate_task(obj)    -> list of error strings (empty = valid)
    validate_result(obj)  -> list of error strings (empty = valid)
    synthetic_results(n)  -> deterministic fake result records for WS-D

Self-check (validates the worked examples + 200 synthetic records):

    python eval/loop_eval/schema.py

Pure stdlib on purpose — the runner imports this inside per-task subprocesses.
"""
from __future__ import annotations

import json
import random
from typing import Any

SCHEMA_VERSION = "1.0"

CATEGORIES = {
    "gmail", "calendar", "filesystem", "shell", "reminder", "weather",
    "web_search",
}
FAILURE_CLASSES = {
    "none", "bad_args", "transient_error", "wrong_tool_first", "unrecoverable",
}
OUTCOME_CLASSES = {
    "completed", "honest_giveup", "timeout", "runner_crash", "gated_confirm",
    "sandbox_violation",
}
# outcome classes counted in the success denominator (frozen decision #1:
# gated_confirm and runner-level errors are excluded and reported separately)
DENOMINATOR_OUTCOMES = {"completed", "honest_giveup"}
DECISION_SOURCES = {"initial", "fastpath", "llm"}
PREDICATE_TYPES = {
    "tool_called", "tool_not_called", "fixture_state", "response_contains",
    "honest_failure",
}
FIXTURE_STATE_OPS = {"eq", "count_eq", "contains", "exists"}
EXCERPT_MAX = 500


# ── helpers ──────────────────────────────────────────────────────────────────

def _need(obj: dict, field: str, types: type | tuple, errs: list[str],
          where: str) -> Any:
    if field not in obj:
        errs.append(f"{where}: missing required field '{field}'")
        return None
    val = obj[field]
    if not isinstance(val, types):
        errs.append(f"{where}: field '{field}' must be {types}, got "
                    f"{type(val).__name__}")
        return None
    return val


def _validate_predicate(node: Any, errs: list[str], where: str) -> None:
    if not isinstance(node, dict):
        errs.append(f"{where}: predicate node must be an object")
        return
    combinators = [k for k in ("all", "any", "not") if k in node]
    if combinators:
        if len(node) != 1:
            errs.append(f"{where}: combinator node must have exactly one key, "
                        f"got {sorted(node)}")
            return
        key = combinators[0]
        if key == "not":
            _validate_predicate(node["not"], errs, f"{where}.not")
        else:
            children = node[key]
            if not isinstance(children, list) or not children:
                errs.append(f"{where}.{key}: must be a non-empty list")
                return
            for i, child in enumerate(children):
                _validate_predicate(child, errs, f"{where}.{key}[{i}]")
        return

    ptype = node.get("type")
    if ptype not in PREDICATE_TYPES:
        errs.append(f"{where}: unknown predicate type {ptype!r}")
        return
    if ptype in ("tool_called", "tool_not_called"):
        _need(node, "tool", str, errs, f"{where}({ptype})")
        if ptype == "tool_called":
            if "args_subset" in node and not isinstance(node["args_subset"], dict):
                errs.append(f"{where}: args_subset must be an object")
            if "min_times" in node and (not isinstance(node["min_times"], int)
                                        or node["min_times"] < 1):
                errs.append(f"{where}: min_times must be an int >= 1")
    elif ptype == "fixture_state":
        _need(node, "fixture", str, errs, f"{where}(fixture_state)")
        _need(node, "path", str, errs, f"{where}(fixture_state)")
        op = _need(node, "op", str, errs, f"{where}(fixture_state)")
        if op is not None and op not in FIXTURE_STATE_OPS:
            errs.append(f"{where}: op must be one of {sorted(FIXTURE_STATE_OPS)}")
        if op != "exists" and op is not None and "value" not in node:
            errs.append(f"{where}: fixture_state op {op!r} requires 'value'")
    elif ptype == "response_contains":
        has_any, has_all = "any_of" in node, "all_of" in node
        if has_any == has_all:  # neither or both
            errs.append(f"{where}: response_contains needs exactly one of "
                        f"any_of/all_of")
        for k in ("any_of", "all_of"):
            if k in node and (not isinstance(node[k], list) or not node[k]
                              or not all(isinstance(s, str) for s in node[k])):
                errs.append(f"{where}: {k} must be a non-empty list of strings")
    # honest_failure has no extra fields


# ── task spec ────────────────────────────────────────────────────────────────

def validate_task(task: dict) -> list[str]:
    """Return a list of contract violations (empty list = valid task spec)."""
    errs: list[str] = []
    if not isinstance(task, dict):
        return ["task: must be a JSON object"]
    tid = _need(task, "id", str, errs, "task")
    where = f"task[{tid or '?'}]"

    _need(task, "base_id", str, errs, where)
    cat = _need(task, "category", str, errs, where)
    if cat is not None and cat not in CATEGORIES:
        errs.append(f"{where}: category {cat!r} not in {sorted(CATEGORIES)}")
    prompt = _need(task, "prompt", str, errs, where)
    if prompt is not None and not prompt.strip():
        errs.append(f"{where}: prompt must be non-empty")
    _need(task, "expected_tool", str, errs, where)

    fi = _need(task, "failure_injection", dict, errs, where)
    fclass = None
    if fi is not None:
        fclass = fi.get("class")
        if fclass not in FAILURE_CLASSES:
            errs.append(f"{where}: failure_injection.class {fclass!r} not in "
                        f"{sorted(FAILURE_CLASSES)}")
        if "params" in fi and not isinstance(fi["params"], dict):
            errs.append(f"{where}: failure_injection.params must be an object")

    if "recoverable" not in task:
        errs.append(f"{where}: missing required field 'recoverable'")
    else:
        rec = task["recoverable"]
        if rec is not None and not isinstance(rec, bool):
            errs.append(f"{where}: recoverable must be bool or null")
        elif fclass in FAILURE_CLASSES:
            expected = (None if fclass == "none"
                        else fclass != "unrecoverable")
            if rec is not expected and rec != expected:
                errs.append(f"{where}: recoverable={rec!r} inconsistent with "
                            f"class {fclass!r} (expected {expected!r})")

    _need(task, "fixture_setup", dict, errs, where)
    sc = _need(task, "success_check", dict, errs, where)
    if sc is not None:
        _validate_predicate(sc, errs, f"{where}.success_check")

    if "tags" in task and (not isinstance(task["tags"], list)
                           or not all(isinstance(t, str) for t in task["tags"])):
        errs.append(f"{where}: tags must be a list of strings")
    if "timeout_s" in task and not isinstance(task["timeout_s"], (int, float)):
        errs.append(f"{where}: timeout_s must be a number")
    return errs


# ── result record ────────────────────────────────────────────────────────────

def validate_result(rec: dict) -> list[str]:
    """Return a list of contract violations (empty list = valid result)."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return ["result: must be a JSON object"]
    tid = rec.get("task_id", "?")
    where = f"result[{tid}]"

    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"{where}: schema_version must be {SCHEMA_VERSION!r}, got "
                    f"{rec.get('schema_version')!r}")
    for f in ("batch_id", "task_id", "base_id", "transcript_path", "ts"):
        _need(rec, f, str, errs, where)
    cat = _need(rec, "category", str, errs, where)
    if cat is not None and cat not in CATEGORIES:
        errs.append(f"{where}: category {cat!r} not in {sorted(CATEGORIES)}")
    fclass = _need(rec, "failure_class", str, errs, where)
    if fclass is not None and fclass not in FAILURE_CLASSES:
        errs.append(f"{where}: failure_class {fclass!r} invalid")
    if "recoverable" not in rec:
        errs.append(f"{where}: missing 'recoverable'")
    elif rec["recoverable"] is not None and not isinstance(rec["recoverable"], bool):
        errs.append(f"{where}: recoverable must be bool or null")

    cond = _need(rec, "condition", dict, errs, where)
    if cond is not None:
        _need(cond, "model", str, errs, f"{where}.condition")
        steps = _need(cond, "tool_loop_max_steps", int, errs, f"{where}.condition")
        if steps is not None and steps < 1:
            errs.append(f"{where}.condition: tool_loop_max_steps must be >= 1")
        _need(cond, "provider", str, errs, f"{where}.condition")

    success = _need(rec, "success", bool, errs, where)
    _need(rec, "selection_correct_first", bool, errs, where)
    loop_triggered = _need(rec, "loop_triggered", bool, errs, where)
    iters_used = _need(rec, "iterations_used", int, errs, where)
    recovery = _need(rec, "recovery", bool, errs, where)
    outcome = _need(rec, "outcome_class", str, errs, where)
    if outcome is not None and outcome not in OUTCOME_CLASSES:
        errs.append(f"{where}: outcome_class {outcome!r} not in "
                    f"{sorted(OUTCOME_CLASSES)}")

    # cross-field invariants (the fields analysis leans on hardest)
    if None not in (recovery, success, loop_triggered):
        if recovery != (success and loop_triggered):
            errs.append(f"{where}: recovery must equal (success AND "
                        f"loop_triggered); got recovery={recovery}, "
                        f"success={success}, loop_triggered={loop_triggered}")

    iters = _need(rec, "iterations", list, errs, where)
    if iters is not None:
        if not iters:
            errs.append(f"{where}: iterations must be non-empty")
        if iters_used is not None and iters_used != len(iters):
            errs.append(f"{where}: iterations_used={iters_used} != "
                        f"len(iterations)={len(iters)}")
        for i, it in enumerate(iters):
            iw = f"{where}.iterations[{i}]"
            if not isinstance(it, dict):
                errs.append(f"{iw}: must be an object")
                continue
            idx = _need(it, "index", int, errs, iw)
            if idx is not None and idx != i + 1:
                errs.append(f"{iw}: index must be {i + 1} (1-based, in order)")
            _need(it, "route", str, errs, iw)
            _need(it, "tool_name", str, errs, iw)
            _need(it, "tool_args", dict, errs, iw)
            src = _need(it, "decision_source", str, errs, iw)
            if src is not None and src not in DECISION_SOURCES:
                errs.append(f"{iw}: decision_source {src!r} not in "
                            f"{sorted(DECISION_SOURCES)}")
            res = _need(it, "result", dict, errs, iw)
            if res is not None:
                _need(res, "success", bool, errs, f"{iw}.result")
                if "error" not in res:
                    errs.append(f"{iw}.result: missing 'error' (may be null)")
                exc = res.get("output_excerpt")
                if not isinstance(exc, str):
                    errs.append(f"{iw}.result: output_excerpt must be a string")
                elif len(exc) > EXCERPT_MAX:
                    errs.append(f"{iw}.result: output_excerpt exceeds "
                                f"{EXCERPT_MAX} chars")
            if "exception" not in it:
                errs.append(f"{iw}: missing 'exception' (may be null)")
            if "gated" not in it:
                errs.append(f"{iw}: missing 'gated' (may be null)")
            elif it["gated"] not in (None, "needs_confirm"):
                errs.append(f"{iw}: gated must be null or 'needs_confirm'")
            for f in ("tokens_in", "tokens_out", "wall_ms"):
                v = _need(it, f, int, errs, iw)
                if v is not None and v < 0:
                    errs.append(f"{iw}: {f} must be >= 0")

    cost = _need(rec, "cost", dict, errs, where)
    if cost is not None:
        for f in ("llm_calls", "tokens_in", "tokens_out", "wall_ms",
                  "loop_overhead_tokens", "finalize_tokens"):
            v = _need(cost, f, int, errs, f"{where}.cost")
            if v is not None and v < 0:
                errs.append(f"{where}.cost: {f} must be >= 0")

    fre = rec.get("final_response_excerpt")
    if fre is not None and (not isinstance(fre, str) or len(fre) > EXCERPT_MAX):
        errs.append(f"{where}: final_response_excerpt must be a string <= "
                    f"{EXCERPT_MAX} chars")
    return errs


# ── worked examples (the SCHEMA.md examples, kept in sync by the self-check) ─

EXAMPLE_TASK: dict = {
    "id": "calendar_create_dinner.transient",
    "base_id": "calendar_create_dinner",
    "category": "calendar",
    "prompt": "add dinner with Defne tomorrow at 7pm to my calendar",
    "expected_tool": "calendar",
    "failure_injection": {
        "class": "transient_error",
        "params": {"fail_times": 1, "error": "API backend error"},
    },
    "recoverable": True,
    "fixture_setup": {"calendar": {"events": []}},
    "success_check": {"all": [
        {"type": "tool_called", "tool": "calendar",
         "args_subset": {"action": "create"}, "min_times": 2},
        {"type": "fixture_state", "fixture": "calendar", "path": "events",
         "op": "count_eq", "value": 1},
        {"type": "response_contains", "any_of": ["dinner", "19:00", "7"]},
    ]},
    "tags": ["pilot"],
}

EXAMPLE_RESULT: dict = {
    "schema_version": SCHEMA_VERSION,
    "batch_id": "b_example",
    "task_id": "calendar_create_dinner.transient",
    "base_id": "calendar_create_dinner",
    "category": "calendar",
    "failure_class": "transient_error",
    "recoverable": True,
    "condition": {"model": "gemma4:e4b-it-qat", "tool_loop_max_steps": 3,
                  "provider": "ollama"},
    "success": True,
    "selection_correct_first": True,
    "loop_triggered": True,
    "iterations_used": 2,
    "recovery": True,
    "outcome_class": "completed",
    "iterations": [
        {"index": 1, "route": "tool", "tool_name": "calendar",
         "tool_args": {"action": "create", "title": "dinner with Defne"},
         "decision_source": "llm",
         "result": {"success": False, "error": "API backend error",
                    "output_excerpt": ""},
         "exception": None, "gated": None,
         "tokens_in": 812, "tokens_out": 96, "wall_ms": 2350},
        {"index": 2, "route": "tool", "tool_name": "calendar",
         "tool_args": {"action": "create", "title": "dinner with Defne"},
         "decision_source": "llm",
         "result": {"success": True, "error": None,
                    "output_excerpt": "Created: dinner with Defne, 19:00"},
         "exception": None, "gated": None,
         "tokens_in": 1104, "tokens_out": 88, "wall_ms": 2710},
    ],
    "cost": {"llm_calls": 3, "tokens_in": 2400, "tokens_out": 310,
             "wall_ms": 9100, "loop_overhead_tokens": 1192,
             "finalize_tokens": 420},
    "final_response_excerpt": "Dinner with Defne is on the calendar for 19:00.",
    "transcript_path": "transcripts/b_example/calendar_create_dinner.transient.json",
    "ts": "2026-07-06T02:14:00Z",
}


# ── synthetic generator (for WS-D, deterministic) ────────────────────────────

def synthetic_results(n: int = 200, seed: int = 500) -> list[dict]:
    """Deterministic fake result records with plausible correlations, so the
    analysis pipeline (#514) can be built and tested before real data exists.
    """
    rng = random.Random(seed)
    cats = sorted(CATEGORIES)
    out: list[dict] = []
    for i in range(n):
        cat = cats[i % len(cats)]
        fclass = ["none", "bad_args", "transient_error", "wrong_tool_first",
                  "unrecoverable"][i % 5]
        recoverable = (None if fclass == "none"
                       else fclass != "unrecoverable")
        steps = 1 if i % 2 == 0 else 3
        base = f"{cat}_task_{i // 10:03d}"
        variant = {"none": "base", "bad_args": "bad_args",
                   "transient_error": "transient",
                   "wrong_tool_first": "wrong_tool",
                   "unrecoverable": "unrecoverable"}[fclass]

        selection_ok = rng.random() < (0.55 if fclass == "wrong_tool_first"
                                       else 0.95)
        if fclass == "none":
            first_ok = selection_ok and rng.random() < 0.9
        else:
            first_ok = False
        loop_triggered = (steps > 1 and not first_ok and rng.random() < 0.8)
        if first_ok:
            success = True
        elif loop_triggered and recoverable:
            success = rng.random() < 0.6
        elif fclass == "unrecoverable":
            success = rng.random() < 0.4  # honest give-up counts as success
        else:
            success = False
        iters_used = 1 if not loop_triggered else rng.choice([2, 3] if steps == 3 else [2])
        iters = []
        for j in range(iters_used):
            last = j == iters_used - 1
            iters.append({
                "index": j + 1, "route": "tool",
                "tool_name": cat if selection_ok or j > 0 else "web_search",
                "tool_args": {"action": "run"},
                "decision_source": "initial" if j == 0 else "llm",
                "result": {"success": bool(success and last),
                           "error": None if (success and last) else "synthetic error",
                           "output_excerpt": "synthetic"},
                "exception": None, "gated": None,
                "tokens_in": 800 + 40 * j, "tokens_out": 90, "wall_ms": 2500,
            })
        overhead = sum(it["tokens_in"] + it["tokens_out"] for it in iters[1:])
        out.append({
            "schema_version": SCHEMA_VERSION,
            "batch_id": "b_synthetic",
            "task_id": f"{base}.{variant}",
            "base_id": base,
            "category": cat,
            "failure_class": fclass,
            "recoverable": recoverable,
            "condition": {"model": "gemma4:e4b-it-qat",
                          "tool_loop_max_steps": steps, "provider": "ollama"},
            "success": success,
            "selection_correct_first": selection_ok,
            "loop_triggered": loop_triggered,
            "iterations_used": iters_used,
            "recovery": success and loop_triggered,
            "outcome_class": ("honest_giveup"
                              if fclass == "unrecoverable" and success
                              else "completed"),
            "iterations": iters,
            "cost": {"llm_calls": 1 + iters_used, "tokens_in": 900 * iters_used,
                     "tokens_out": 100 * iters_used, "wall_ms": 3000 * iters_used,
                     "loop_overhead_tokens": overhead, "finalize_tokens": 400},
            "transcript_path": f"transcripts/b_synthetic/{base}.{variant}.json",
            "ts": "2026-07-02T00:00:00Z",
        })
    return out


# ── self-check ───────────────────────────────────────────────────────────────

def _main() -> int:
    failures = 0
    errs = validate_task(EXAMPLE_TASK)
    if errs:
        failures += 1
        print("EXAMPLE_TASK invalid:\n  " + "\n  ".join(errs))
    errs = validate_result(EXAMPLE_RESULT)
    if errs:
        failures += 1
        print("EXAMPLE_RESULT invalid:\n  " + "\n  ".join(errs))
    synth = synthetic_results(200)
    bad = [(r["task_id"], e) for r in synth for e in validate_result(r)]
    if bad:
        failures += 1
        print(f"synthetic records invalid ({len(bad)} errors):")
        for tid, e in bad[:10]:
            print(f"  {tid}: {e}")
    # negative checks: the validators must actually reject broken input
    broken_task = dict(EXAMPLE_TASK, recoverable=False)  # contradicts class
    if not validate_task(broken_task):
        failures += 1
        print("validator failed to reject inconsistent recoverable/class")
    broken_res = json.loads(json.dumps(EXAMPLE_RESULT))
    broken_res["recovery"] = False  # contradicts success+loop_triggered
    if not validate_result(broken_res):
        failures += 1
        print("validator failed to reject inconsistent recovery invariant")
    if failures:
        print(f"SELF-CHECK FAILED ({failures} problem groups)")
        return 1
    print(f"loop_eval contract v{SCHEMA_VERSION} self-check OK: "
          f"examples valid, {len(synth)} synthetic records valid, "
          f"negative checks rejected as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
