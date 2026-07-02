# loop_eval task corpus (issue #507)

40 base tasks, one JSONL file per category, all conforming to the frozen
contract v1.0 (`../SCHEMA.md`, validated by `../schema.py`). Failure-injection
variants (`bad_args`, `transient`, `wrong_tool`, `unrecoverable`) are derived
from these base tasks in issue #508.

## Category distribution

| file | category | tasks | what they cover |
|---|---|---|---|
| `gmail.jsonl` | gmail | 8 | summary, count, sender/topic search, read, 2× send, empty-inbox honesty |
| `calendar.jsonl` | calendar | 8 | today/date/upcoming queries, 2× create, delete, conflicts, empty-calendar honesty |
| `filesystem.jsonl` | filesystem | 7 | 2× read, 2× write, ls, folder+file create, overwrite |
| `shell.jsonl` | shell | 5 | disk, uptime, memory, python version, ls (all canned, deterministic) |
| `reminder.jsonl` | reminder | 4 | 2× add, list, cancel |
| `weather.jsonl` | weather | 4 | 2× city query, rain question, local default |
| `web_search.jsonl` | web_search | 4 | 3× canned topical search, default fallback |

7 categories (contract requires ≥5). 10 tasks are tagged `pilot` — the
pre-flight subset exercised end-to-end by `tests/eval/test_corpus.py`.

## Design rules

- **Prompts are English-only** (frozen decision #3; MarianMT out of scope).
- **Mutations are checked against fixture state** (`fixture_state` on the
  virtual store — authoritative), with `tool_called` args as corroboration.
- **Reads are checked via `response_contains`** using data tokens that only
  exist in the fixture state (e.g. `"09:00"`, `"hazar"`), so a correct answer
  must have flowed through the tool — it cannot be guessed.
- **Empty-state tasks** (`gmail_empty_inbox_01`, `calendar_empty_01`) pin
  honesty: the assistant must report absence, not invent content.
- Every `success_check` is programmatic (no judge model) and every task is
  solvable by a human reading `fixture_setup` (sanity-pass instructions:
  read the prompt, read the state, check the predicate by hand).

## The `reference_call` field

Each task carries a `reference_call` — the golden `{tool, args}` invocation
that solves it. This is **not part of the frozen contract** (`validate_task`
ignores unknown fields); it exists for the harness:

- the solvability test executes it against a fresh `FixtureWorld` and asserts
  the task's `success_check` passes (and fails on an untouched world);
- the pilot test drives it through `brain.process` inside the sandbox with
  routing pinned, shaking out spec bugs without paying LLM routing variance.

Regeneration: the corpus is hand-maintained JSONL; edit lines directly and
re-run `python eval/loop_eval/schema.py` plus `pytest tests/eval/`.
