"""
Bantz — Jury: event-driven system verifier (#557).

The jury "constantly monitors" without ever polling the LLM. It listens
to signals that already flow on the event bus — job failures, workflow
failures, delegation outcomes, observer errors, anomaly transitions —
keeps a 30-minute rolling window, and applies deterministic escalation
rules. Only when a rule fires AND the hourly LLM budget allows does it
delegate one analysis to the tiny read-only "jury" sub-agent; the verdict
lands in the InterventionQueue (whose accept/dismiss log doubles as RL
feedback on verdict usefulness) and on the bus as ``jury_verdict``.

Steady state (no signals) costs zero LLM calls by construction.

A daily self-check job (cron, BANTZ_JURY_SELFCHECK_HOUR) additionally
summarizes scheduler/agent/lane/intervention stats into one low-priority
briefing note — the single scheduled LLM call the jury is allowed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger("bantz.jury")

_WINDOW_S = 30 * 60
_RULE_COOLDOWN_S = 30 * 60          # a rule re-fires at most every 30 min
_CRITICAL_ANOMALY_PERSIST_S = 5 * 60
_KV_BUDGET_KEY = "jury:llm_calls"    # JSON list of epoch timestamps
_KV_SELFCHECK_KEY = "jury:selfcheck"


def _get_kv():
    from bantz.data.sqlite_store import SQLiteKVStore
    from bantz.config import config
    return SQLiteKVStore(config.db_path)


class Jury:
    def __init__(self) -> None:
        self._enabled = False
        # (monotonic_ts, kind, data) — rolling signal window
        self._signals: deque[tuple[float, str, dict]] = deque(maxlen=500)
        # anomaly id → first-seen monotonic ts (for the persistence rule)
        self._anomaly_seen: dict[str, float] = {}
        self._anomaly_critical: set[str] = set()
        self._rule_last_fired: dict[str, float] = {}
        self._llm_calls: list[float] = []  # epoch ts, persisted
        self._evaluating = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── lifecycle ─────────────────────────────────────────────────────────

    def init(self) -> None:
        """Subscribe to bus signals. Called from the daemon at startup."""
        from bantz.config import config
        self._enabled = getattr(config, "jury_enabled", False)
        if not self._enabled:
            return
        from bantz.core.event_bus import bus
        for name in ("job_failed", "workflow_failed", "observer_error",
                     "delegation_done", "health_alert"):
            bus.on(name, self._make_handler(name))
        bus.on("anomaly_detected", self._on_anomalies)
        self._load_budget()
        log.info(
            "Jury initialised — window %dm, LLM budget %d/h",
            _WINDOW_S // 60, config.jury_llm_budget_per_hour,
        )

    def _make_handler(self, kind: str):
        def _handler(event) -> None:
            self._signals.append((time.monotonic(), kind, dict(event.data)))
            self._kick_evaluate()
        _handler.__name__ = f"jury_on_{kind}"
        return _handler

    def _on_anomalies(self, event) -> None:
        """Track anomaly lifetimes from edge-triggered anomaly_detected."""
        now = time.monotonic()
        anomalies = event.data.get("anomalies") or []
        current = {str(a.get("id")) for a in anomalies}
        for a in anomalies:
            aid = str(a.get("id"))
            self._anomaly_seen.setdefault(aid, now)
            if a.get("severity") == "critical":
                self._anomaly_critical.add(aid)
        for gone in set(self._anomaly_seen) - current:
            self._anomaly_seen.pop(gone, None)
            self._anomaly_critical.discard(gone)
        self._signals.append((now, "anomaly_detected", {"ids": sorted(current)}))
        self._kick_evaluate()

    def _kick_evaluate(self) -> None:
        if self._evaluating:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._evaluating = True
        loop.create_task(self._evaluate())

    # ── rule tier (deterministic, free) ───────────────────────────────────

    def _window(self, kind: str) -> list[dict]:
        cutoff = time.monotonic() - _WINDOW_S
        return [d for ts, k, d in self._signals if k == kind and ts >= cutoff]

    def _fired_rules(self) -> list[tuple[str, str]]:
        """Return (rule_id, human reason) for every rule currently firing."""
        rules: list[tuple[str, str]] = []

        failed_jobs = self._window("job_failed")
        if len(failed_jobs) >= 3:
            ids = {d.get("job_id", "?") for d in failed_jobs}
            rules.append((
                "job_failures",
                f"{len(failed_jobs)} scheduled-job failures in 30m ({', '.join(sorted(ids))})",
            ))

        delegations = self._window("delegation_done")[-10:]
        failures = [d for d in delegations if not d.get("success")]
        if len(delegations) >= 4 and len(failures) * 2 > len(delegations):
            rules.append((
                "delegation_failures",
                f"{len(failures)}/{len(delegations)} recent agent delegations failed",
            ))

        by_source: dict[str, int] = {}
        for d in self._window("observer_error"):
            src = str(d.get("source") or d.get("category") or "unknown")
            by_source[src] = by_source.get(src, 0) + 1
        for src, n in by_source.items():
            if n >= 3:
                rules.append((
                    f"observer_{src}",
                    f"{n} repeated errors from '{src}' in 30m",
                ))

        now = time.monotonic()
        for aid in self._anomaly_critical:
            first = self._anomaly_seen.get(aid)
            if first is not None and now - first >= _CRITICAL_ANOMALY_PERSIST_S:
                rules.append((
                    f"anomaly_{aid}",
                    f"critical anomaly '{aid}' persisting >5m",
                ))

        if self._window("workflow_failed"):
            wf = self._window("workflow_failed")
            if len(wf) >= 2:
                names = {d.get("workflow", "?") for d in wf}
                rules.append((
                    "workflow_failures",
                    f"{len(wf)} workflow failures in 30m ({', '.join(sorted(names))})",
                ))

        return rules

    async def _evaluate(self) -> None:
        try:
            now = time.monotonic()
            for rule_id, reason in self._fired_rules():
                last = self._rule_last_fired.get(rule_id, 0.0)
                if now - last < _RULE_COOLDOWN_S:
                    continue
                self._rule_last_fired[rule_id] = now
                await self._escalate(rule_id, reason)
        except Exception:
            log.exception("jury evaluation failed")
        finally:
            self._evaluating = False

    # ── LLM tier (budgeted) ───────────────────────────────────────────────

    def _load_budget(self) -> None:
        try:
            raw = _get_kv().get(_KV_BUDGET_KEY, "")
            if raw:
                self._llm_calls = [float(x) for x in json.loads(raw)]
        except Exception:
            self._llm_calls = []

    def _budget_available(self) -> bool:
        from bantz.config import config
        cutoff = time.time() - 3600
        self._llm_calls = [t for t in self._llm_calls if t >= cutoff]
        return len(self._llm_calls) < config.jury_llm_budget_per_hour

    def _spend_budget(self) -> None:
        self._llm_calls.append(time.time())
        try:
            _get_kv().set(_KV_BUDGET_KEY, json.dumps(self._llm_calls))
        except Exception:
            pass

    async def _escalate(self, rule_id: str, reason: str) -> None:
        """Rule fired → produce a verdict (LLM-assisted when budget allows)."""
        verdict = "degraded"
        cause = reason
        suggestion = ""
        used_llm = False

        if self._budget_available():
            self._spend_budget()
            used_llm = True
            digest = self._incident_digest(rule_id, reason)
            try:
                from bantz.agent.agent_manager import agent_manager
                result = await agent_manager.delegate("jury", digest, internal=True)
                if result.success:
                    data = result.data or {}
                    verdict = str(data.get("verdict") or "degraded")
                    cause = str(data.get("cause") or result.summary or reason)[:300]
                    suggestion = str(data.get("suggestion") or "")[:300]
            except Exception as exc:
                log.warning("jury LLM escalation failed: %s", exc)

        self._push_verdict(rule_id, verdict, cause, suggestion, used_llm)

    def _incident_digest(self, rule_id: str, reason: str) -> str:
        cutoff = time.monotonic() - _WINDOW_S
        recent = [
            {"kind": k, **{key: str(v)[:120] for key, v in d.items()}}
            for ts, k, d in list(self._signals)[-15:] if ts >= cutoff
        ]
        return (
            f"System incident triggered rule '{rule_id}': {reason}.\n"
            f"Recent signals:\n```json\n{json.dumps(recent, ensure_ascii=False)}\n```\n"
            "Judge the system state. Respond with JSON: "
            '{"summary": "...", "data": {"verdict": "ok|degraded|broken", '
            '"cause": "...", "suggestion": "..."}}'
        )

    def _push_verdict(
        self, rule_id: str, verdict: str, cause: str,
        suggestion: str, used_llm: bool,
    ) -> None:
        from bantz.agent.interventions import (
            Priority, intervention_from_system, intervention_queue,
        )
        priority = Priority.HIGH if verdict == "broken" else Priority.MEDIUM
        reason = cause + (f" — {suggestion}" if suggestion else "")
        try:
            intervention_queue.push(intervention_from_system(
                f"Jury: {verdict} ({rule_id})", reason,
                priority=priority, ttl=3600,
            ))
        except Exception as exc:
            log.debug("jury intervention push failed: %s", exc)
        try:
            from bantz.core.event_bus import bus, new_corr_id
            bus.emit_threadsafe(
                "jury_verdict",
                corr_id=new_corr_id(), rule=rule_id, verdict=verdict,
                cause=cause, suggestion=suggestion, llm=used_llm,
            )
        except Exception:
            pass
        log.info("Jury verdict [%s]: %s — %s (llm=%s)", rule_id, verdict, cause, used_llm)

    # ── daily self-check ──────────────────────────────────────────────────

    async def selfcheck(self) -> str:
        """Gather subsystem stats → ONE LLM summary → briefing cache +
        low-priority intervention. Returns the summary text."""
        stats: dict[str, Any] = {}
        try:
            from bantz.agent.job_scheduler import job_scheduler
            stats["scheduler"] = job_scheduler.stats()
        except Exception:
            pass
        try:
            from bantz.agent.agent_manager import agent_manager
            stats["agents"] = agent_manager.stats()
        except Exception:
            pass
        try:
            from bantz.llm.lane import lane
            stats["llm_lane"] = lane.stats()
        except Exception:
            pass
        try:
            from bantz.agent.interventions import intervention_queue
            stats["interventions"] = intervention_queue.stats()
        except Exception:
            pass
        stats["open_anomalies"] = sorted(self._anomaly_seen)

        summary = ""
        try:
            from bantz.llm.lane import llm_call
            summary = await llm_call([
                {"role": "system", "content": (
                    "You are the daily self-check of a personal assistant "
                    "daemon. Given subsystem stats, answer in 2-3 plain "
                    "sentences: is everything working, and what (if anything) "
                    "needs attention? No preamble."
                )},
                {"role": "user", "content": json.dumps(stats, default=str)[:4000]},
            ], interactive=False)
        except Exception as exc:
            log.warning("jury selfcheck LLM summary failed: %s", exc)
            summary = f"Self-check ran; LLM summary unavailable ({exc})."

        try:
            _get_kv().set(_KV_SELFCHECK_KEY, json.dumps(
                {"ts": time.time(), "summary": summary}))
        except Exception:
            pass
        try:
            from bantz.agent.interventions import (
                Priority, intervention_from_system, intervention_queue,
            )
            intervention_queue.push(intervention_from_system(
                "Daily self-check", summary[:400],
                priority=Priority.LOW, ttl=6 * 3600,
            ))
        except Exception:
            pass
        log.info("Jury self-check: %s", summary[:200])
        return summary

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "signals_30m": len([1 for ts, _, _ in self._signals
                                if ts >= time.monotonic() - _WINDOW_S]),
            "llm_calls_last_hour": len(self._llm_calls),
            "open_anomalies": sorted(self._anomaly_seen),
        }


jury = Jury()


async def _job_jury_selfcheck() -> None:
    """APScheduler entry point (module-level for pickle-safety)."""
    if jury.enabled:
        await jury.selfcheck()
