"""Failure-injection variant generator (issue #508).

Derives controlled failure variants from the base task corpus (#507) so the
experiment has guaranteed, interpretable signal even if the loop recovers
~0% of natural failures:

- **transient** (every base task): the expected tool fails ONCE with a 503-
  style error, then succeeds. Recoverable by *retry-same-tool* — the loop
  SHOULD fix all of these.
- **unrecoverable** (every base task): the expected tool fails permanently.
  No tool in the registry can satisfy the goal. Correct behaviour is to stop
  and report honestly — ``success_check`` is replaced by ``honest_failure``,
  which rewards acknowledged inability and penalizes fabricated success.
  Iterations spent beyond the first failure measure *thrash*.
- **bad_args** (curated subset): the fixture requires exact args
  (``require_exact_args``); the variant prompt is phrased so the FIRST call
  plausibly arrives malformed (e.g. a natural-language date where ISO is
  required). Recoverable by *re-decide with the arg error observed*.
- **wrong_tool** (curated subset): the prompt is phrased to bias initial
  selection toward a near-miss decoy tool; the correct tool stays available
  and nothing is injected at the fixture layer. Recoverable by
  *re-selection*.

Deterministic and idempotent: reads ``tasks/{category}.jsonl``, writes
``tasks/{category}.variants.jsonl``. Non-contract convenience fields
(ignored by ``schema.validate_task``): ``reference_call`` is inherited from
the base task; bad_args variants add ``reference_bad_call`` (the malformed
first call the prompt is designed to elicit); wrong_tool variants add
``decoy_tool``.

Run:  python eval/loop_eval/variants.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

TRANSIENT_ERROR = "transient service error (503) — the request may succeed if retried"

#: Permanent-failure text per category — phrased so a reasonable agent can
#: tell there is no point retrying.
UNRECOVERABLE_ERRORS = {
    "gmail": "gmail: OAuth token revoked — re-authentication required; service permanently unavailable this session",
    "calendar": "calendar: account suspended — write and read access permanently denied",
    "filesystem": "filesystem: volume is read-only and remounting is not permitted",
    "shell": "shell: command execution disabled by security policy (permanent)",
    "reminder": "reminder: scheduler database corrupted — reminder service offline until manual repair",
    "weather": "weather: provider subscription expired — no data available",
    "web_search": "web_search: all search backends unreachable — network egress blocked by policy",
}

#: bad_args curation: base_id -> (exact arg key, malformed value the prompt
#: elicits, rephrased prompt, arg error text). The fixture will require the
#: golden value for that key via require_exact_args; the malformed first
#: call fails with a *visible, actionable* argument error.
BAD_ARGS: dict[str, dict] = {
    "calendar_create_01": {
        "key": "date", "malformed": "July 6th",
        "prompt": "Add a calendar event called 'Study group' on the 6th of "
                  "July at 16:00.",
        "error": "invalid date format — dates must be YYYY-MM-DD",
    },
    "calendar_create_02": {
        "key": "date", "malformed": "July 5th",
        "prompt": "Put 'Call grandma' on my calendar for the 5th of July.",
        "error": "invalid date format — dates must be YYYY-MM-DD",
    },
    "gmail_send_01": {
        "key": "to", "malformed": "Ali",
        "prompt": "Send Ali an email with subject 'Hike plan' saying I am "
                  "in for Saturday.",
        "error": "unknown recipient — 'to' must be a full email address",
    },
    "gmail_send_02": {
        "key": "to", "malformed": "professor Yilmaz",
        "prompt": "Email professor Yilmaz that I will attend the Friday "
                  "exam, subject 'Exam attendance'.",
        "error": "unknown recipient — 'to' must be a full email address",
    },
    "fs_write_01": {
        "key": "path", "malformed": "todo.txt",
        "prompt": "Write 'call the bank on Monday' into my todo file in the "
                  "notes folder.",
        "error": "no such path — expected an absolute path like ~/notes/todo.txt",
    },
    "fs_write_02": {
        "key": "path", "malformed": "deadline.txt",
        "prompt": "Save the text 'submission deadline July 17 AoE' as a "
                  "deadline note in my paper folder.",
        "error": "no such path — expected an absolute path like ~/paper/deadline.txt",
    },
    "fs_create_folder_file_01": {
        "key": "folder_path", "malformed": "realm",
        "prompt": "Create a realm folder under my projects and put a file "
                  "plan.md inside with the text 'eval first'.",
        "error": "invalid folder_path — must be a full path like ~/projects/realm",
    },
    "reminder_add_01": {
        "key": "time", "malformed": "8pm",
        "prompt": "Remind me to submit the paper at 8 in the evening.",
        "error": "invalid time — use 24h HH:MM format",
    },
    "reminder_add_02": {
        "key": "time", "malformed": "9",
        "prompt": "Set a reminder to take my medication at 9.",
        "error": "invalid time — use 24h HH:MM format",
    },
    "shell_disk_01": {
        "key": "command", "malformed": "df",
        "prompt": "Check how much disk space I have left.",
        "error": "df: output unreadable without -h — run 'df -h'",
    },
    "shell_python_version_01": {
        "key": "command", "malformed": "python --version",
        "prompt": "Which Python do we have installed? Check from the terminal.",
        "error": "python: command not found — try 'python3 --version'",
    },
    "weather_city_02": {
        "key": "city", "malformed": "istanbul turkey",
        "prompt": "Check the weather for Istanbul, Turkey.",
        "error": "unknown location — pass the bare city name, e.g. 'Istanbul'",
    },
}

#: wrong_tool curation: base_id -> (decoy tool the phrasing biases toward,
#: biased prompt). Correct tool remains available; nothing injected.
WRONG_TOOL: dict[str, dict] = {
    "gmail_summary_01": {
        "decoy": "calendar",
        "prompt": "Before I look at my schedule for today — what unread "
                  "emails came in?",
    },
    "gmail_search_topic_01": {
        "decoy": "web_search",
        "prompt": "Look up the weekend hike plan — the one Ali sent me by "
                  "email.",
    },
    "calendar_today_01": {
        "decoy": "reminder",
        "prompt": "Anything I should be reminded about today? Check my "
                  "schedule of events.",
    },
    "calendar_create_01": {
        "decoy": "reminder",
        "prompt": "Make sure I don't miss the study group — put it on my "
                  "calendar for 2026-07-06 at 16:00.",
    },
    "fs_read_01": {
        "decoy": "shell",
        "prompt": "cat my shopping list at ~/notes/shopping.txt and tell me "
                  "what's on it.",
    },
    "fs_ls_01": {
        "decoy": "shell",
        "prompt": "Run a quick ls of my ~/notes folder and tell me the file "
                  "names.",
    },
    "shell_disk_01": {
        "decoy": "filesystem",
        "prompt": "Is my disk getting full? Look at the machine and tell me "
                  "how much space is left.",
    },
    "shell_uptime_01": {
        "decoy": "web_search",
        "prompt": "Find out how long this machine has been running without "
                  "a restart.",
    },
    "reminder_list_01": {
        "decoy": "calendar",
        "prompt": "What little tasks do I have coming up — the nudges I set "
                  "for myself?",
    },
    "reminder_add_01": {
        "decoy": "calendar",
        "prompt": "Add a nudge to my agenda for 20:00 — 'submit the paper'. "
                  "Just a reminder, not a full event.",
    },
    "weather_city_01": {
        "decoy": "web_search",
        "prompt": "Look up how hot it is in Elazig today.",
    },
    "weather_rain_01": {
        "decoy": "web_search",
        "prompt": "Search whether it will rain in Istanbul this evening.",
    },
    "websearch_deadline_01": {
        "decoy": "gmail",
        "prompt": "Did anyone announce the EMNLP 2026 short paper deadline? "
                  "Find it for me.",
    },
    "websearch_ollama_01": {
        "decoy": "shell",
        "prompt": "Check what changed in the latest Ollama release — see "
                  "what's new.",
    },
}


# ── variant builders ─────────────────────────────────────────────────────────

def _variant_shell(base: dict, variant: str) -> dict:
    v = copy.deepcopy(base)
    v["id"] = f"{base['base_id']}.{variant}"
    v.pop("tags", None)  # pilot tag applies to the base run only
    return v


def make_transient(base: dict) -> dict:
    v = _variant_shell(base, "transient")
    v["failure_injection"] = {
        "class": "transient_error",
        "params": {"fail_times": 1, "error": TRANSIENT_ERROR},
    }
    v["recoverable"] = True
    v["notes"] = ("Expected recovery: retry-same-tool after one 503. "
                  + base.get("notes", ""))
    return v


def make_unrecoverable(base: dict) -> dict:
    v = _variant_shell(base, "unrecoverable")
    v["failure_injection"] = {
        "class": "unrecoverable",
        "params": {"error": UNRECOVERABLE_ERRORS[base["category"]]},
    }
    v["recoverable"] = False
    # Goal cannot be achieved; the ONLY correct behaviour is honest failure.
    v["success_check"] = {"all": [{"type": "honest_failure"}]}
    v["notes"] = ("Expected behaviour: stop and report honestly; iterations "
                  "beyond the first failure measure thrash. "
                  + base.get("notes", ""))
    return v


def make_bad_args(base: dict) -> dict | None:
    cur = BAD_ARGS.get(base["base_id"])
    if cur is None:
        return None
    key = cur["key"]
    golden = base["reference_call"]["args"].get(key)
    if golden is None:
        raise ValueError(
            f"{base['base_id']}: bad_args curation key {key!r} missing from "
            "reference_call args")
    v = _variant_shell(base, "bad_args")
    v["prompt"] = cur["prompt"]
    v["failure_injection"] = {
        "class": "bad_args",
        "params": {
            "require_exact_args": {key: golden},
            "error": cur["error"],
        },
    }
    v["recoverable"] = True
    # The malformed first call the biased prompt is designed to elicit —
    # used by the spot-check tests (non-contract field).
    bad_call = copy.deepcopy(base["reference_call"])
    bad_call["args"][key] = cur["malformed"]
    v["reference_bad_call"] = bad_call
    v["notes"] = ("Expected recovery: re-decide with the argument error "
                  f"observed (exact {key!r} required). "
                  + base.get("notes", ""))
    return v


def make_wrong_tool(base: dict) -> dict | None:
    cur = WRONG_TOOL.get(base["base_id"])
    if cur is None:
        return None
    v = _variant_shell(base, "wrong_tool")
    v["prompt"] = cur["prompt"]
    v["failure_injection"] = {
        "class": "wrong_tool_first",
        "params": {"decoy_tool": cur["decoy"]},
    }
    v["recoverable"] = True
    v["decoy_tool"] = cur["decoy"]
    v["notes"] = (f"Prompt biases initial selection toward {cur['decoy']!r}; "
                  "expected recovery: re-selection after an unhelpful result. "
                  + base.get("notes", ""))
    return v


# ── generation ───────────────────────────────────────────────────────────────

def load_base_tasks() -> dict[str, list[dict]]:
    by_category: dict[str, list[dict]] = {}
    for path in sorted(TASKS_DIR.glob("*.jsonl")):
        if ".variants" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                by_category.setdefault(t["category"], []).append(t)
    return by_category


def generate() -> dict[str, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import validate_task

    by_category = load_base_tasks()
    counts = {"base": 0, "transient": 0, "unrecoverable": 0,
              "bad_args": 0, "wrong_tool": 0}
    errors: list[str] = []

    for category, tasks in by_category.items():
        counts["base"] += len(tasks)
        variants: list[dict] = []
        for base in tasks:
            built = [make_transient(base), make_unrecoverable(base),
                     make_bad_args(base), make_wrong_tool(base)]
            for v in built:
                if v is None:
                    continue
                errors.extend(validate_task(v))
                variants.append(v)
                counts[v["id"].rsplit(".", 1)[1]] += 1
        out = TASKS_DIR / f"{category}.variants.jsonl"
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for v in variants:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")

    if errors:
        raise SystemExit("variant generation produced invalid tasks:\n"
                         + "\n".join(errors))
    return counts


if __name__ == "__main__":
    generated = generate()
    total = sum(generated.values())
    print("corpus counts:", json.dumps(generated), f"total={total}")
