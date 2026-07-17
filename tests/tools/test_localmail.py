"""Tests for the localmail tool (#552) — notmuch CLI mocked."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from bantz.config import config
from bantz.tools.localmail import LocalMailTool, _build_query, _extract_bodies

THREADS = [
    {"thread": "0001", "subject": "Project deadline", "authors": "Prof X",
     "date_relative": "2 hours ago", "tags": ["inbox", "unread"]},
    {"thread": "0002", "subject": "50% off everything!", "authors": "Shop",
     "date_relative": "3 hours ago", "tags": ["inbox", "unread"]},
]


@pytest.fixture
def tool():
    return LocalMailTool()


@pytest.fixture
def enabled():
    with patch.object(config, "localmail_enabled", True):
        yield


def _fake_notmuch(responses):
    """Return a _run_notmuch stub keyed by the notmuch subcommand."""
    def run(args, timeout=10.0):
        return responses.get(args[0], (False, "unexpected"))
    return run


async def test_disabled_flag_blocks(tool):
    with patch.object(config, "localmail_enabled", False):
        r = await tool.execute(action="count")
    assert not r.success and "disabled" in r.error


async def test_count(tool, enabled):
    with patch("bantz.tools.localmail._run_notmuch",
               _fake_notmuch({"count": (True, "7\n")})):
        r = await tool.execute(action="count")
    assert r.success and r.data["count"] == 7 and "7 unread" in r.output


async def test_search_formats_threads(tool, enabled):
    with patch("bantz.tools.localmail._run_notmuch",
               _fake_notmuch({"search": (True, json.dumps(THREADS))})):
        r = await tool.execute(action="search", query="deadline")
    assert r.success
    assert "Project deadline" in r.output
    assert len(r.data["threads"]) == 2


async def test_unread_summary_categorizes(tool, enabled):
    with patch("bantz.tools.localmail._run_notmuch",
               _fake_notmuch({"search": (True, json.dumps(THREADS))})):
        r = await tool.execute(action="unread_summary")
    assert r.success
    # The categorizer keeps personal/institutional and drops promo noise;
    # at minimum the output mentions the important thread.
    assert "Project deadline" in r.output


async def test_missing_notmuch_binary_is_graceful(tool, enabled):
    with patch("bantz.tools.localmail.shutil.which", return_value=None):
        r = await tool.execute(action="count")
    assert not r.success and "not installed" in r.error


def test_build_query_mapping():
    assert _build_query({}) == "tag:unread"
    q = _build_query({"from_sender": "prof@uni.edu", "days_ago": 2, "unread": True})
    assert "from:prof@uni.edu" in q and "date:2d.." in q and "tag:unread" in q
    assert _build_query({"query": "subject:x"}) == "subject:x"


def test_extract_bodies_walks_nested_show_json():
    show = [[[{
        "headers": {"From": "a@b.c", "Subject": "Hi", "Date": "today"},
        "body": [{"content-type": "text/plain", "content": "hello world"}],
    }, []]]]
    text = _extract_bodies(show)
    assert "From: a@b.c" in text and "hello world" in text
