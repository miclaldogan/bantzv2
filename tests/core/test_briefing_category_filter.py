"""Morning briefing applies the Gmail category filter (issue #494).

The Gmail tool buckets mail into personal / institutional / services /
payments / notifications, but the briefing never used it — so low-value
notifications and services cluttered the morning summary. These tests pin
``Briefing._gmail_from_cache`` to the briefing filter: personal /
institutional are kept, everything else is dropped.
"""
from __future__ import annotations

from bantz.core.briefing import Briefing


def _cache(unread: int, urgent: list, normal: list) -> dict:
    return {"status": "ok", "data": {
        "unread": unread, "urgent_count": len(urgent),
        "urgent": urgent, "normal": normal,
    }}


def test_low_value_categories_excluded_high_value_kept():
    out = Briefing()._gmail_from_cache(_cache(
        unread=4,
        urgent=[{"from": "Erasmus Office <office@firat.edu.tr>",
                 "subject": "Placement decision"}],           # institutional
        normal=[
            {"from": "Defne <defne@gmail.com>",
             "subject": "dinner tonight?"},                    # personal
            {"from": "GitHub <noreply@github.com>",
             "subject": "[bantzv2] new pull request"},         # notifications
            {"from": "Apple <news@apple.com>",
             "subject": "Your weekly digest"},                 # services
        ],
    ))
    assert out is not None
    assert "4 unread emails" in out
    # kept
    assert "Placement decision" in out
    assert "dinner tonight?" in out
    # dropped
    assert "pull request" not in out.lower()
    assert "github" not in out.lower()
    assert "digest" not in out.lower()


def test_all_low_value_reports_nothing_important():
    out = Briefing()._gmail_from_cache(_cache(
        unread=3, urgent=[],
        normal=[
            {"from": "GitHub <noreply@github.com>", "subject": "PR merged"},
            {"from": "LinkedIn <noreply@linkedin.com>", "subject": "5 new jobs"},
        ],
    ))
    assert out is not None
    assert "3 unread emails" in out
    assert "nothing that needs your attention" in out
    assert "PR merged" not in out


def test_inbox_clean_unchanged():
    assert Briefing()._gmail_from_cache(
        {"status": "ok", "data": {"unread": 0}}) == "Inbox is clean"


def test_auth_error_passthrough():
    out = Briefing()._gmail_from_cache({"status": "auth_error"})
    assert out is not None and "re-authenticate" in out.lower()


def test_non_ok_status_returns_none():
    assert Briefing()._gmail_from_cache({"status": "error"}) is None
