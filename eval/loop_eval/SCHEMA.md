# loop_eval contract v1.0 — task spec & result record (issue #500)

The three-way contract between **WS-B** (task corpus, #507), **WS-C** (eval runner,
#510), and **WS-D** (analysis, #514). `schema.py` in this directory is the executable
form: it validates both schemas and generates synthetic result records so WS-D can
build before real data exists.

**Change policy:** this contract is FROZEN at v1.0. Any change requires a PR that
touches this file, bumps `schema_version`, and carries explicit sign-off from the
owners of WS-B, WS-C, and WS-D (one approving comment each on the PR).

**Frozen decisions (signed off 2026-07-02):**
1. **Gated tasks:** a run that stops at a confirmation gate gets
   `outcome_class="gated_confirm"` and is **excluded from the success denominator**,
   with the count reported separately. The corpus avoids confirm-triggering tasks by
   design; the runner never auto-confirms.
2. **Primary metric is programmatic-only.** `success_check` predicates are the sole
   success verdict. No judge model on the primary metric. (An advisory judge field
   may be ADDED in a future version; it can never redefine `success`.)
3. **Corpus language is English-only (v1).** The MarianMT translation layer is out of
   scope for the experiment and goes to the paper's Limitations.

---

## 1. Task spec

One JSON object per line in `eval/loop_eval/tasks/*.jsonl`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique. Convention: `{base_id}.{variant}`, variant ∈ `base`, `bad_args`, `transient`, `wrong_tool`, `unrecoverable`. |
| `base_id` | string | yes | Groups a base task with its failure variants → paired per-task deltas in analysis. |
| `category` | enum | yes | `gmail`, `calendar`, `filesystem`, `shell`, `reminder`, `weather`, `web_search`. |
| `prompt` | string | yes | English user utterance passed to `brain.process`. |
| `expected_tool` | string | yes | Registry tool name. Powers the selection-vs-completion decomposition. |
| `failure_injection` | object | yes | `{"class": <failure class>, "params": {...}}`. Class ∈ `none`, `bad_args`, `transient_error`, `wrong_tool_first`, `unrecoverable`. `params` is class-specific (e.g. `{"fail_times": 1, "error": "..."}`). |
| `recoverable` | bool or null | yes | Must be `null` iff class is `none`, `false` iff class is `unrecoverable`, `true` otherwise. Validated. |
| `fixture_setup` | object | yes | Initial fixture-backend state, keyed by fixture name. |
| `success_check` | predicate tree | yes | See §1.1. Programmatic only. |
| `notes` | string | no | Free-text documentation. |
| `tags` | list[string] | no | Filtering (e.g. `pilot`). |
| `timeout_s` | number | no | Per-task override of the runner's default watchdog. |

### 1.1 `success_check` predicate tree

Combinators (exactly one key per node):

- `{"all": [<node>, ...]}` — every child must pass
- `{"any": [<node>, ...]}` — at least one child must pass
- `{"not": <node>}` — child must fail

Leaves (`{"type": ..., ...}`):

| type | fields | passes when |
|---|---|---|
| `tool_called` | `tool`, opt `args_subset` (dict), opt `min_times` (int, default 1) | fixture call-log contains ≥ min_times calls to `tool` whose args are a superset of `args_subset`. |
| `tool_not_called` | `tool` | no call to `tool` in the log. |
| `fixture_state` | `fixture`, `path` (dot-path), `op` ∈ `eq`,`count_eq`,`contains`,`exists`, `value` (not for `exists`) | final fixture state at `path` satisfies `op`/`value`. |
| `response_contains` | `any_of` or `all_of` (list[string]), opt `case_insensitive` (default true) | final user-facing response text matches. |
| `honest_failure` | — | final response acknowledges inability AND no fabricated success (implemented as deterministic phrase/shape checks in the harness — still no judge model). Success criterion for `unrecoverable` tasks. |

## 2. Result record

One JSON object per line in `results/{batch_id}/{condition_slug}.jsonl`. Append-only,
fsynced after every task (checkpoint/resume contract, #511). Resume dedupe key:
`(task_id, condition)`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema_version` | `"1.0"` | yes | |
| `batch_id` | string | yes | |
| `task_id`, `base_id`, `category`, `failure_class`, `recoverable` | — | yes | Denormalized from the task spec so analysis never joins files. |
| `condition` | object | yes | `{"model": str, "tool_loop_max_steps": int ≥1, "provider": str}`. |
| `success` | bool | yes | `success_check` verdict = tool ran AND goal achieved (for `unrecoverable`: honest behavior). |
| `selection_correct_first` | bool | yes | Iteration 1 selected `expected_tool`. |
| `loop_triggered` | bool | yes | ≥1 re-decision LLM call happened. NOT derivable from `iterations` length: a re-decision that chose to stop adds no execution entry. Distinguishes *failed-because-loop-never-fired* (plausible-but-wrong success) from *failed-despite-looping*. |
| `iterations_used` | int ≥1 | yes | Number of TOOL EXECUTIONS. Must equal `len(iterations)`. |
| `recovery` | bool | yes | Must equal `success AND loop_triggered`. Validated. |
| `outcome_class` | enum | yes | `completed`, `honest_giveup`, `timeout`, `runner_crash`, `gated_confirm`, `sandbox_violation`. Everything except `completed`/`honest_giveup` is excluded from the success denominator and reported separately. |
| `iterations` | list, ≥1 | yes | Per tool execution — see §2.1. |
| `cost` | object | yes | `{"llm_calls": int, "tokens_in": int, "tokens_out": int, "wall_ms": int, "loop_overhead_tokens": int, "finalize_tokens": int}`. Totals for the whole task. `loop_overhead_tokens` counts only iteration-≥2 routing+execution tokens (the loop-attributable cost) so cost-per-recovery needs no schema change. |
| `transcript_path` | string | yes | Full untruncated trace for taxonomy labeling (#516). |
| `ts` | string | yes | ISO-8601, runner-stamped. |
| `final_response_excerpt` | string ≤500 | no | Triage convenience. |
| `error_detail` | string | no | For `timeout`/`runner_crash`/`sandbox_violation`. |

### 2.1 Iteration entry

| Field | Type | Required |
|---|---|---|
| `index` | int, 1-based | yes |
| `route` | string (`tool`/`chat`/`planner`) | yes |
| `tool_name` | string | yes |
| `tool_args` | object | yes |
| `decision_source` | enum `initial`, `fastpath`, `llm` | yes — `fastpath` on index ≥2 means the skip_fastpath bypass leaked; analysis flags it loudly. |
| `result` | `{"success": bool, "error": str or null, "output_excerpt": str ≤500}` | yes |
| `exception` | string or null | yes — non-null when the tool raised (the brain.py:966-968 interception path). |
| `gated` | null or `"needs_confirm"` | yes |
| `tokens_in`, `tokens_out`, `wall_ms` | int | yes |

## 3. Batch manifest

`manifests/{batch_id}.json`, written at batch start (#511): `{batch_id, git_sha,
model_digests (from ollama show), conditions, task_file_sha256, env_snapshot
(BANTZ_* only), started_ts, finished_ts}`.

## 4. What WS-D computes from this — no future schema change needed

- **Recovery fraction per class**: `recovery` ÷ (recoverable tasks whose iteration 1 failed), grouped by `failure_class`.
- **Cost per recovery**: Σ `cost.loop_overhead_tokens` over recovered tasks ÷ recoveries.
- **Headline gap** (necessary-but-not-sufficient): `selection_correct_first AND NOT success` rate at `tool_loop_max_steps=1`.
- **Never-retriggered residual**: `NOT success AND NOT loop_triggered` at steps>1 — the plausible-but-wrong-success ceiling category.
- **Thrash on unrecoverable**: `iterations_used > 1` where `recoverable == false`.
- **Paired deltas**: same `task_id` across conditions; same `base_id` across variants.
