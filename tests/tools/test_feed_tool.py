"""Tests for FeedTool (#289)."""
from __future__ import annotations

import textwrap
from datetime import datetime

import pytest

from bantz.tools.feed_tool import (
    FeedItem,
    FeedTool,
    FeedToolError,
    _parse_date,
    _strip_html,
    parse_feed,
)


# ── Sample feeds ──────────────────────────────────────────────────────────────

RSS_SAMPLE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>First Article</title>
          <link>https://example.com/1</link>
          <description>Summary of the first article.</description>
          <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
          <media:content url="https://example.com/img1.jpg" medium="image"/>
        </item>
        <item>
          <title>Second Article</title>
          <link>https://example.com/2</link>
          <description>&lt;p&gt;HTML in summary&lt;/p&gt;</description>
          <pubDate>Tue, 02 Jan 2024 08:00:00 GMT</pubDate>
          <enclosure url="https://example.com/img2.jpg" type="image/jpeg"/>
        </item>
        <item>
          <title>No Date Article</title>
          <link>https://example.com/3</link>
          <description></description>
        </item>
      </channel>
    </rss>
""")

ATOM_SAMPLE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Atom Test Feed</title>
      <entry>
        <title>Atom Entry One</title>
        <link rel="alternate" href="https://example.com/atom/1"/>
        <summary>First atom entry.</summary>
        <published>2024-03-15T10:00:00Z</published>
      </entry>
      <entry>
        <title>Atom Entry Two</title>
        <link href="https://example.com/atom/2"/>
        <content>Content of entry two.</content>
        <updated>2024-03-14T08:00:00Z</updated>
      </entry>
    </feed>
""")

INVALID_XML = "<not><valid xml"


# ── Tests: parse_feed ─────────────────────────────────────────────────────────

class TestParseFeed:
    def test_rss_basic(self):
        items = parse_feed(RSS_SAMPLE, source_name="TestFeed")
        assert len(items) == 3
        # Sorted by date descending: Second (Jan 2), First (Jan 1), No Date (min)
        assert items[0].title == "Second Article"
        assert items[1].title == "First Article"
        assert items[2].title == "No Date Article"

    def test_rss_fields(self):
        items = parse_feed(RSS_SAMPLE)
        first = next(i for i in items if i.title == "First Article")
        assert first.link == "https://example.com/1"
        assert "first article" in first.summary.lower()
        assert first.image_url == "https://example.com/img1.jpg"
        assert first.published_at is not None
        assert first.published_at.year == 2024

    def test_rss_enclosure_image(self):
        items = parse_feed(RSS_SAMPLE)
        second = next(i for i in items if i.title == "Second Article")
        assert second.image_url == "https://example.com/img2.jpg"

    def test_rss_html_stripped(self):
        items = parse_feed(RSS_SAMPLE)
        second = next(i for i in items if i.title == "Second Article")
        assert "<p>" not in second.summary
        assert "HTML in summary" in second.summary

    def test_rss_missing_description(self):
        items = parse_feed(RSS_SAMPLE)
        no_date = next(i for i in items if i.title == "No Date Article")
        assert no_date.summary == ""
        assert no_date.published_at is None
        assert no_date.image_url is None

    def test_rss_source_name(self):
        items = parse_feed(RSS_SAMPLE, source_name="MySource")
        assert all(i.source_name == "MySource" for i in items)

    def test_atom_basic(self):
        items = parse_feed(ATOM_SAMPLE)
        assert len(items) == 2
        # Sorted by date descending
        assert items[0].title == "Atom Entry One"
        assert items[1].title == "Atom Entry Two"

    def test_atom_fields(self):
        items = parse_feed(ATOM_SAMPLE)
        first = items[0]
        assert first.link == "https://example.com/atom/1"
        assert "first atom entry" in first.summary.lower()
        assert first.published_at is not None

    def test_atom_content_fallback(self):
        items = parse_feed(ATOM_SAMPLE)
        second = items[1]
        assert "Content of entry two" in second.summary

    def test_invalid_xml_raises(self):
        with pytest.raises(FeedToolError, match="Failed to parse"):
            parse_feed(INVALID_XML)

    def test_empty_feed(self):
        xml = '<?xml version="1.0"?><rss><channel></channel></rss>'
        items = parse_feed(xml)
        assert items == []

    def test_sorted_by_date_descending(self):
        items = parse_feed(RSS_SAMPLE)
        dates = [i.published_at for i in items if i.published_at]
        assert dates == sorted(dates, reverse=True)


# ── Tests: _parse_date ────────────────────────────────────────────────────────

class TestParseDate:
    def test_rfc2822(self):
        dt = _parse_date("Mon, 01 Jan 2024 12:00:00 GMT")
        assert dt is not None
        assert dt.year == 2024

    def test_iso8601(self):
        dt = _parse_date("2024-03-15T10:00:00Z")
        assert dt is not None
        assert dt.month == 3

    def test_date_only(self):
        dt = _parse_date("2024-01-15")
        assert dt is not None
        assert dt.day == 15

    def test_none_input(self):
        assert _parse_date(None) is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_garbage(self):
        assert _parse_date("not a date") is None


# ── Tests: _strip_html ───────────────────────────────────────────────────────

class TestStripHtml:
    def test_basic_tags(self):
        assert _strip_html("<p>Hello</p>") == "Hello"

    def test_entities(self):
        assert _strip_html("&amp; &lt; &gt;") == "& < >"

    def test_nested(self):
        assert _strip_html("<div><b>Bold</b> text</div>") == "Bold text"

    def test_plain_text(self):
        assert _strip_html("No HTML here") == "No HTML here"


# ── Tests: FeedTool actions ───────────────────────────────────────────────────

class TestFeedTool:
    def test_list_categories(self):
        tool = FeedTool()
        result = tool._list_categories()
        assert result.success
        assert "tech" in result.output.lower() or "Available" in result.output

    @pytest.mark.asyncio
    async def test_fetch_no_url(self):
        tool = FeedTool()
        result = await tool.execute(action="fetch")
        assert not result.success
        assert "url" in result.error.lower() or "category" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_category(self):
        tool = FeedTool()
        result = await tool.execute(action="category", category="nonexistent")
        assert not result.success
        assert "Unknown category" in result.error

    def test_format_items(self):
        items = [
            FeedItem(
                title="Test",
                link="https://example.com",
                summary="A test summary",
                published_at=datetime(2024, 1, 1, 12, 0),
                source_name="Source",
            ),
        ]
        result = FeedTool._format_items(items)
        assert result.success
        assert "Test" in result.output
        assert "Source" in result.output
        assert "items" in result.data

    def test_format_empty_items(self):
        result = FeedTool._format_items([])
        assert result.success
        assert "No items" in result.output
