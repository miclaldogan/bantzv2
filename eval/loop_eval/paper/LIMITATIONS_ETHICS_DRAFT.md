# Limitations + Ethics — draft for the REALM short paper (issue #522)

Drafted ahead of the paper skeleton (#517–#521) so the final gate is
mechanical. Paste into the LaTeX as `\section*{Limitations}` (required,
uncounted under ACL policy) and the ethics note; update every bracketed
`[n=…]` from `analyze.py` output before submission — no number in the
camera-ready may bypass that pipeline.

---

## Limitations (draft)

**Single system, single embodiment.** All results are measured inside one
open-source personal-assistant stack (Bantz) with its specific routing
prompt, butler persona and finalizer. The observe→re-decide loop is
implemented at that system's tool-execution boundary; portability of the
recovery fractions to other agent scaffolds is untested.

**Fixture backends, not live APIs.** Tools are deterministic fixtures with
injected failures (§Experimental Setup), which is what makes programmatic
success checks and controlled failure classes possible — but live Gmail /
Calendar / shell surfaces exhibit failure modes (latency spikes, partial
responses, auth drift) our four injected classes only approximate.

**Small local models only.** We evaluate 4B-class quantized local models
[list from the batch manifests]; the residual-failure taxonomy explicitly
measures that ceiling. Findings should not be extrapolated to frontier
models, where both first-shot accuracy and re-decision quality differ.

**Task distribution.** The corpus (~40 base tasks, [146] with variants) is
authored by the researchers around one user's assistant workload
(email/calendar/files/shell/reminders/weather/search), English-only (v1;
the system's Turkish translation layer is bypassed). It is not a sample
from real usage logs, and per-class n is small — we report bootstrap CIs
and withhold them below 5 pairs rather than imply precision we do not have
[n per class from analyze.py].

**Recovery is bounded by the step budget.** We test
`tool_loop_max_steps ∈ {1, 3}`; the budget-exhausted-mid-recovery category
in the taxonomy quantifies how often the cap, not the model, ended an
otherwise-live recovery.

## Ethics note (draft)

The evaluation corpus contains **no real user data**: every inbox, calendar
event, file, location and reminder in the fixture states is synthetic and
authored for this paper. Runs execute inside a sandbox that redirects all
persistence to per-task temporary directories and fails the run on any
write to live user data (verified per record; `sandbox_violation` is a
reported outcome class). The underlying assistant is a local-first system;
no third-party services are called during evaluation — web search and
weather are canned fixtures. The failure-injection framework is a
measurement instrument for robustness and does not create new dual-use
capability beyond the assistant's existing tools.

---

## Pre-submission gate (mechanical checklist, from #522)

- [ ] Limitations section present, uncounted; every bracketed number
      replaced from `analyze.py` output (grep for `[` before building)
- [ ] Ethics note present (above)
- [ ] Internal adversarial read by a non-WS-E team member; every claim
      traced to an `analyze.py` field or a code line reference
- [ ] 4pp content, A4, `pdffonts` shows all fonts embedded, no Type-3
      (figures.py already enforces this for our figures), ruler ON,
      anonymized, no live repo links (anonymized repo per #517 plan)
- [ ] Plain-text abstract ≤200 words, byte-identical to the PDF abstract,
      for OpenReview metadata
- [ ] Reviewer nomination selected
- [ ] Submitted on OpenReview ≥24h before Jul 17 AoE; confirmation
      email/screenshot archived next to the batch manifests
