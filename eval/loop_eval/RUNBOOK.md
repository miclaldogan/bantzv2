# loop_eval RUNBOOK — compute coordination (issue #509)

**The GPU is the one resource we cannot parallelize.** One Ollama instance
serves the daemon, everyone's dev smoke tests, and the eval batches — and it
stalls under concurrent load on this machine (known: web_research). Two
model-heavy runs at once = both runs garbage. These are the rules.

## Run queue rules

1. **Full batches run one at a time.** No exceptions. The runner enforces
   this with `eval/loop_eval/.run.lock` — a second batch start refuses with
   the holder's pid, batch id and start time.
2. **Announce before you start.** Say it in the team channel: batch id,
   condition set, expected duration. Preferably claim an overnight slot.
3. **Prefer overnight.** The 3am maintenance window is quiet; a full
   condition matrix belongs there, not at 2pm when someone is demoing.
4. **Dev smoke tests are capped at 5 tasks.** Anything bigger is a batch and
   takes the lock. (`--limit 5` in the runner; the pilot suite in
   `tests/eval/` uses mocked LLMs and does not touch Ollama at all.)
5. **Don't run `web_research` or other model-heavy Bantz features during a
   batch** — same Ollama, same stall.

## Pre-flight gate (MANDATORY before any full batch)

Full batches **require a green pre-flight on the current git SHA** (#512):

```bash
python eval/loop_eval/preflight.py            # real conditions, ~minutes
python eval/loop_eval/preflight.py --allow-no-loop   # baseline-only batches
```

Five checks in one run: mini-batch across steps={1,3} with contract-valid
records, hard-kill + resume with no duplicates, zero live writes, a real
recovery on a transient task at steps=3 (waivable with `--allow-no-loop`
ONLY for steps=1 batches until the C1 loop lands), and the watchdog firing
on a hung task. Output is archived to `eval/loop_eval/preflight_report.md`
— commit it alongside the batch manifest. RED verdict = the batch does not
start, whatever the schedule says.

## Overnight batch procedure

```bash
# 0. Green pre-flight on THIS SHA (see above) — no green, no batch
python eval/loop_eval/preflight.py

# 1. Check nobody is running
python eval/loop_eval/runlock.py status

# 2. Stop the daemon — it shares Ollama and its scheduled jobs (3am
#    maintenance, overnight polls) WILL collide with a long batch
systemctl --user stop bantz-daemon

# 3. Launch the batch (the runner takes the lock itself)
#    ... eval runner command per its own docs (#510) ...

# 4. MORNING, ALWAYS — even if the batch crashed:
systemctl --user start bantz-daemon
python eval/loop_eval/runlock.py status   # should be FREE
```

**The daemon MUST be restarted after the batch.** A stopped daemon means no
wake word, no Telegram, no scheduled jobs for the user — leaving it down is
a user-facing outage, not a dev inconvenience.

## Stall handling

Symptoms: task wall-clock way past the watchdog, Ollama not responding to
`curl localhost:11434/api/version`, load average pegged.

1. Kill the runner process (Ctrl-C / `kill <pid>`). Checkpointing (#511)
   makes this cheap: completed tasks are already fsynced to the results
   JSONL; resume skips them.
2. Restart Ollama if it's wedged: `systemctl restart ollama` (or however it
   is supervised on the box).
3. Resume the batch — same batch id, the runner dedupes on
   `(task_id, condition)`.
4. If the same task stalls twice, record it as `timeout` and move on —
   that's data, not a blocker.

## Lock mechanics & recovery

- Lock file: `eval/loop_eval/.run.lock` (JSON: pid, host, batch_id, start
  time). Created atomically (`O_CREAT|O_EXCL`) + advisory `flock` on Linux.
- **Stale lock** (holder pid dead, same host): the next acquire steals it
  automatically with a warning — a crashed runner never blocks the queue.
- **Lock held from another host** is never auto-stolen. Confirm with the
  holder, then: `python eval/loop_eval/runlock.py release --force`.
- Inspect any time: `python eval/loop_eval/runlock.py status`.
