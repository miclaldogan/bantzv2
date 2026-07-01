"""Google account-identity check (audit S5).

The calendar token in production authenticated as a *different* account than
the user's personal Gmail, so events were created successfully but landed on
an invisible calendar. Nothing in code surfaced the mismatch. This module
fetches the authenticated identity once per process per service and:

  - logs the active account at INFO (so the wrong-account case is diagnosable),
  - logs a WARNING when it differs from the configured ``BANTZ_GOOGLE_ACCOUNT``.

It never blocks — it makes a silent misconfiguration loud.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("bantz.google")

# Services already checked this process, so we make at most one extra API call
# per service per process rather than one per request.
_checked: set[str] = set()


def _expected_account() -> str:
    try:
        from bantz.config import config
        return (getattr(config, "google_account", "") or "").strip().lower()
    except Exception:
        return ""


def check_once(kind: str, service: Any, fetch_email: Callable[[Any], str]) -> None:
    """Verify the authenticated identity of *service* exactly once per process.

    ``kind`` is "gmail"/"calendar"; ``fetch_email`` extracts the account email
    from the built service (one cheap API call). All failures degrade to a
    debug log — identity checking must never break a real request.
    """
    if kind in _checked:
        return
    _checked.add(kind)
    try:
        email = (fetch_email(service) or "").strip()
    except Exception as exc:
        log.debug("Google %s identity check skipped: %s", kind, exc)
        return
    if not email:
        return
    expected = _expected_account()
    if expected and email.lower() != expected:
        log.warning(
            "Google %s authenticated as %r but BANTZ_GOOGLE_ACCOUNT=%r — "
            "reads/writes will hit the WRONG account (audit S5)",
            kind, email, expected,
        )
    else:
        log.info("Google %s authenticated as %s", kind, email)
