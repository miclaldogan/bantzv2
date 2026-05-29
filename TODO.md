# Bantz Paper Roadmap

Two Q1 journal papers from the Bantz codebase, on a 6-month plan committed 2026-05-29.

- **Paper 1** — Tool-grounded hallucination detection. Target submission **2026-08-29** (13 weeks).
- **Paper 2** — Budget-constrained hybrid memory recall (OmniMemory). Target submission **2026-11-29** (26 weeks).

Phase 0 (instrumentation) landed on branch `paper-1/finalizer-logging` (4 commits, 46 tests green). The branch is **not yet pushed** — push when you're ready to start collecting data.

## Legend

- **By:** who owns the task — `you` (manual work only you can do) / `claude` (I can do this code/analysis work) / `both` (paired)
- **Estimate:** wall-clock effort, not duration
- **Deliverable:** concrete artifact produced
- **Done when:** verifiable criterion you can check before moving on

---

# Paper 1 — Hallucination detection (13 weeks)

## Phase 1 — Live data collection (2026-05-30 → 2026-06-19, 3 weeks)

**Goal:** accumulate ≥500 turn rows in `hallucination_log` across the tool mix, so labeling has enough signal.

### 1.1 Push the branch
- **By:** you
- **Estimate:** 5 min
- **Deliverable:** `paper-1/finalizer-logging` on origin
- **Done when:** `git push -u origin paper-1/finalizer-logging` succeeds; `git ls-remote origin paper-1/finalizer-logging` returns a ref
- **Notes:** keep it as the daily-driver branch for the next 3 weeks — every turn writes (route, finalizer, message) triples

### 1.2 Verify instrumentation in the live system
- **By:** you
- **Estimate:** 15 min
- **Deliverable:** confirmation that all three tables are receiving writes
- **Done when:** after 10 real turns on the branch, this prints non-zero counts:
  ```bash
  .venv/bin/python -c "from bantz.core.memory import memory; memory.init('$HOME/.local/share/bantz/bantz.db'); memory.new_session(); from bantz.core import eval_view; print(eval_view.stats())"
  ```
- **Notes:** if `joined_to_finalizer` is 0 but `assistant_messages` is non-zero, the soft-join is failing — flag to me

### 1.3 Use Bantz heavily across the full tool mix
- **By:** you
- **Estimate:** spread over 21 days, ~20–30 turns/day → ~500–600 turns total
- **Deliverable:** `eval_view.stats()` shows `joined_to_finalizer ≥ 500` with non-trivial spread across tools
- **Done when:** at least 5 distinct `finalizer_tool` values appear with ≥ 30 rows each
- **Notes:** prioritize **gmail, calendar, weather, web_search, document, shell** — these have the richest hallucination surfaces. Avoid pure chat-route turns (they produce no finalizer row). Don't try to *trigger* hallucinations — use the tool naturally; let the data show what the model actually does.

### 1.4 Add detector v2 signal stubs (NOT enabled, just logged)
- **By:** claude
- **Estimate:** 1 day of code work
- **Deliverable:** new columns on `hallucination_log` for `date_fabrication_score`, `url_fabrication_score`, `negation_flip_score`, `tool_specific_score`; signals are computed and logged but **not** used to flag (existing `confidence` and `flagged` columns remain the ground truth for now)
- **Done when:** new tests pass; existing flagged set is unchanged; live system still writes correctly
- **Notes:** I'll do this in week 2 of Phase 1 so the back-half of your data collection captures v2 signals on the same turns

### 1.5 Mid-phase data check at day 14
- **By:** you
- **Estimate:** 5 min
- **Deliverable:** decision point — do we have enough or do we extend?
- **Done when:** you've run `python -m paper1_eval.metrics --db ~/.local/share/bantz/bantz.db` and inspected the row count + tool distribution
- **Notes:** if at day 14 you have < 250 rows, either accelerate usage or extend Phase 1 by a week and compress Phase 6

---

## Phase 2 — Labeling (2026-06-20 → 2026-07-03, 2 weeks)

**Goal:** 500+ human-labeled finalizer pairs, balanced enough to compute precision/recall.

### 2.1 Export a labeling batch
- **By:** you
- **Estimate:** 1 min
- **Deliverable:** `pairs_phase1.jsonl` snapshot
- **Done when:** file exists with ≥500 rows
- **Notes:** snapshot before labeling so we have a frozen reference; the SQLite table keeps growing as you keep using Bantz
  ```bash
  python -m paper1_eval.export_pairs --db ~/.local/share/bantz/bantz.db --out pairs_phase1.jsonl
  ```

### 2.2 Label 500 pairs
- **By:** you
- **Estimate:** 8–12 hours total, in 1-hour sessions over 10 days
- **Deliverable:** `paper1_labels` table populated with 500+ entries
- **Done when:** `python -m paper1_eval.metrics --db ~/.local/share/bantz/bantz.db` reports `Labeled rows: 500+`
- **Notes:**
  - Use `python -m paper1_eval.label_tui --db ~/.local/share/bantz/bantz.db --labeler memre`
  - Aim for roughly even mix of flagged vs. unflagged — alternate plain runs with `--flagged-only`
  - Stick to the rubric: **faithful** = every claim in response is supported by tool_output; **hallucinated** = ≥1 fabricated claim (made-up email, wrong number, invented name); **partial** = mostly grounded but adds unsupported colour; **unsure** when you genuinely can't tell from the data shown — don't overuse this

### 2.3 Inter-annotator agreement check (optional but recommended)
- **By:** you (find a second labeler)
- **Estimate:** 2 hours
- **Deliverable:** 50 pairs labeled by a second person (a friend, collaborator, or you-tomorrow-without-looking)
- **Done when:** Cohen's kappa computed; if κ < 0.6, refine the rubric and re-label
- **Notes:** I can write the kappa script. Even an n=50 sample carries weight in the methodology section. If solo, label 50 pairs twice with ≥2 days between sessions and compute self-agreement.

### 2.4 Snapshot labeled data
- **By:** you
- **Estimate:** 5 min
- **Deliverable:** `pairs_labeled.jsonl` with labels merged in
- **Done when:** file committed to a separate **data branch** (do not commit raw labels to the main paper branch — keep them out of the public history if Bantz becomes open-source)
- **Notes:**
  ```bash
  python -m paper1_eval.export_pairs --db ~/.local/share/bantz/bantz.db --out pairs_labeled.jsonl
  ```

---

## Phase 3 — Detector v2 + signal analysis (2026-07-04 → 2026-07-17, 2 weeks)

**Goal:** improve the regex detector based on what Phase 2 labels reveal it misses.

### 3.1 Error-analysis pass on labeled data
- **By:** claude
- **Estimate:** 1 day
- **Deliverable:** report listing the top miss patterns (FN cases the current detector didn't flag) and top false-alarm patterns (FP cases it flagged that were labeled faithful)
- **Done when:** report includes 10+ concrete miss examples with proposed signal rules
- **Notes:** I'll pull this from your labeled SQLite directly

### 3.2 Implement v2 signal rules
- **By:** claude
- **Estimate:** 3–4 days
- **Deliverable:** updated `hallucination_check` returning a richer score vector; configurable per-signal weights
- **Done when:** unit tests pass; running the v2 detector against the labeled set improves F1 by ≥0.05 vs. v1
- **Notes:** key candidates based on what's missing today — fabricated URLs, fabricated dates/times, negation flips ("did not" ↔ "did"), tool-specific rules (calendar event title match, gmail sender match, weather city match)

### 3.3 Lock detector versions
- **By:** you
- **Estimate:** 10 min
- **Deliverable:** commit tagging the v1 and v2 detector versions; eval pipeline can pick which to score
- **Done when:** `paper1_eval.metrics --detector v1` and `--detector v2` both run and report
- **Notes:** versioning is critical for the paper's reproducibility claim

---

## Phase 4 — Baselines (2026-07-18 → 2026-07-24, 1 week)

**Goal:** two comparison baselines so the paper has something to beat.

### 4.1 LLM-judge baseline
- **By:** claude
- **Estimate:** 2 days
- **Deliverable:** `paper1_eval/baselines/llm_judge.py` that asks Ollama (and optionally Gemini) "is this response faithful to this tool output? yes/no/partial" for each pair
- **Done when:** script runs against the labeled set, produces predictions, and `metrics.py` can score them; results table cached as JSON
- **Notes:** uses the model already configured in `.env` — no extra deps. Run on the **same** labeled set as the regex detector for direct comparison.

### 4.2 String-overlap baseline
- **By:** claude
- **Estimate:** 0.5 days
- **Deliverable:** `paper1_eval/baselines/string_overlap.py` — a trivial baseline that flags when response has < some-threshold token overlap with tool_output
- **Done when:** script runs, metrics scored, threshold swept for best F1
- **Notes:** this is the lower-bound baseline — the paper's contribution must clearly beat it

### 4.3 Optional — no-op detector
- **By:** claude
- **Estimate:** 1 hour
- **Deliverable:** baseline that always predicts "faithful" or always "hallucinated"
- **Done when:** scored
- **Notes:** establishes the prior (class imbalance) for context

---

## Phase 5 — Evaluation + analysis (2026-07-25 → 2026-08-03, 1.5 weeks)

**Goal:** the result tables and figures that will go in the paper.

### 5.1 Main results table
- **By:** claude
- **Estimate:** 1 day
- **Deliverable:** CSV/markdown table — precision / recall / F1 / accuracy for v1, v2, llm-judge, string-overlap, no-op detectors at chosen thresholds
- **Done when:** table file committed; results are reproducible from the labeled JSONL

### 5.2 ROC + threshold-sweep figures
- **By:** claude
- **Estimate:** 1 day
- **Deliverable:** ROC curves for each detector; AUC values; F1-vs-threshold plot
- **Done when:** PNG/SVG figures generated; reproducible from a script in `paper1_eval/figures/`
- **Notes:** matplotlib only; no fancy plotting deps

### 5.3 Per-tool breakdown
- **By:** claude
- **Estimate:** 0.5 days
- **Deliverable:** F1 per `finalizer_tool` for each detector
- **Done when:** table shows where v2 helps vs. hurts vs. v1 across tools (gmail, calendar, weather, etc.)

### 5.4 Ablation: signal contribution
- **By:** claude
- **Estimate:** 1 day
- **Deliverable:** table showing F1 with each v2 signal individually disabled
- **Done when:** all 8 ablation runs scored

### 5.5 Latent-bug case study writeup
- **By:** you (draft) + claude (polish)
- **Estimate:** 0.5 days
- **Deliverable:** 1-paragraph methodology note about the `result.tool` AttributeError that silenced the pre-paper detector for the entire codebase history — it motivates why retrospective evaluation needs careful re-instrumentation
- **Done when:** paragraph drafted; supporting commit (`f9df19d`) linked

### 5.6 Mid-eval decision point
- **By:** you
- **Estimate:** 30 min review
- **Deliverable:** go/no-go decision
- **Done when:** you've looked at the result table and decided: (a) results are publishable as-is, (b) v2 needs another iteration, or (c) we need more labels. If (b) or (c), compress Phase 6.

---

## Phase 6 — Paper write-up + submission (2026-08-04 → 2026-08-29, 3.5 weeks)

**Goal:** submission-ready Q1 paper.

### 6.1 Outline + venue lock-in
- **By:** you
- **Estimate:** 2 hours
- **Deliverable:** paper outline (8 sections) + target venue selected with author guidelines pulled
- **Done when:** outline committed to a `paper/` directory; venue's LaTeX template downloaded
- **Notes:** rank order — Knowledge-Based Systems, Expert Systems with Applications, Information Processing & Management. KBS has the friendliest scope for systems-flavoured contributions.

### 6.2 Related work section
- **By:** you (own this — reviewers check claim originality first)
- **Estimate:** 4 days
- **Deliverable:** ~30 cited papers across (a) hallucination detection in LLMs, (b) tool-augmented agents, (c) RAG faithfulness, (d) local-first AI assistants
- **Done when:** every claim in the introduction has a citation; key prior work (e.g., FactScore, SelfCheckGPT, RAGAS) explicitly compared
- **Notes:** I can help research but you should be the one judging which papers carry weight in your community

### 6.3 Method section
- **By:** both
- **Estimate:** 3 days
- **Deliverable:** the architecture figure + detector v2 algorithm description
- **Done when:** an unfamiliar reader could re-implement from your description alone; figure shows the pipeline cleanly

### 6.4 Evaluation section
- **By:** claude (drafts), you (judgment + revisions)
- **Estimate:** 3 days
- **Deliverable:** results presentation with tables/figures from Phase 5, error analysis, threats to validity
- **Done when:** every result tied to a script in `paper1_eval/`; reproducibility statement present

### 6.5 Intro + abstract
- **By:** you
- **Estimate:** 2 days
- **Deliverable:** intro that motivates tool-grounded fact consistency, plus a 250-word abstract
- **Done when:** intro narrative is tight, abstract hits problem/method/result/significance in that order
- **Notes:** write these LAST — they should reflect what the paper actually says

### 6.6 Discussion + limitations
- **By:** you
- **Estimate:** 2 days
- **Deliverable:** honest limitations (single user, single model family, single language pair, regex-only signals), broader-impact paragraph
- **Done when:** at least 4 named limitations; future-work bullet list

### 6.7 Internal review
- **By:** you (recruit 1-2 readers)
- **Estimate:** 1 week wall-clock for reviewers, 2 days revision
- **Deliverable:** annotated PDF + your response
- **Done when:** all major comments addressed or explicitly waived in a response note

### 6.8 Submission package
- **By:** you
- **Estimate:** 1 day
- **Deliverable:** uploaded manuscript + supplementary code release + cover letter
- **Done when:** submission portal confirms receipt
- **Notes:** for the code release: tag the commit, scrub the labeled-data branch for PII, write a `REPRODUCING.md`

---

# Paper 2 — OmniMemory hybrid recall (weeks 14-26, sketched)

Detail comes after Paper 1 submits. Skeleton only here.

## Phase 7 — Benchmark adaptation (2026-08-30 → 2026-09-26, 4 weeks)
- 7.1 Pull LongMemEval and/or LoCoMo, adapt to Bantz's MemPalace schema  *(claude)*
- 7.2 Generate synthetic long-horizon dialogues for Turkish so the benchmark reflects Bantz's actual usage  *(both)*
- 7.3 Lock dataset version, snapshot to a `paper2_eval/` dir  *(you)*

## Phase 8 — Baselines (2026-09-27 → 2026-10-10, 2 weeks)
- 8.1 Pure vector RAG baseline (no graph, no deep)  *(claude)*
- 8.2 Pure graph baseline (no vector)  *(claude)*
- 8.3 Random and recency baselines  *(claude)*

## Phase 9 — Eval + ablations (2026-10-11 → 2026-10-24, 2 weeks)
- 9.1 Main results — Recall@k, MRR, NDCG, end-task accuracy vs. baselines  *(claude)*
- 9.2 Ablation — 35/40/25 budget split vs. alternatives (uniform, learned)  *(claude)*
- 9.3 Ablation — entity re-ranking on/off  *(claude)*
- 9.4 Ablation — slack redistribution on/off  *(claude)*
- 9.5 Latency analysis — parallel vs. sequential recall  *(claude)*

## Phase 10 — Paper write-up (2026-10-25 → 2026-11-29, 5 weeks)
- Same structure as Phase 6, scaled to a more methods-heavy paper. Target venues: ACM TOIS, Information Processing & Management, Knowledge-Based Systems.

---

# What only you can do

These tasks bottleneck the timeline if delayed — none of them are things I can do for you:

- Phase 1.1, 1.3, 1.5 — using Bantz, generating real data
- Phase 2.2, 2.3 — manual labeling
- Phase 6.1, 6.2, 6.5, 6.6, 6.7, 6.8 — venue choice, related work, intro/abstract, limitations, internal review, submission

# What I can do without bottlenecking you

I can land all of these in parallel with your labeling, so they're ready when you need them:

- Phase 1.4 — detector v2 signal stubs
- Phase 3.1, 3.2 — error analysis + v2 rules
- Phase 4.1, 4.2, 4.3 — three baselines
- Phase 5.1–5.4 — result tables, figures, ablations
- Phase 6.3, 6.4 — method and evaluation section drafts

# Status tracking

When you finish a task, mark it `[x]` in this file and commit:

```bash
git commit -am "todo: phase N.M done — <one-line outcome>"
```

If a task slips by more than a week, re-evaluate the downstream phase timelines and adjust here rather than letting drift accumulate.
