"""Tests for UnifiedContactResolver (#254).

Covers:
- Tier 1: direct email passthrough
- Tier 2: local alias file hit
- Tier 3: Google People API hit (mocked)
- Cache-back: Google result persisted to contacts.json
- Multiple recipients: comma-separated resolution
- Unresolved contact → error tuple
- Google API failure → graceful fallback (no crash)
- resolve_alias (sync) for search contexts
- add_alias / all_aliases management
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from bantz.tools.contact_resolver import UnifiedContactResolver


@pytest.fixture
def resolver(tmp_path):
    """Fresh resolver with contacts.json in a temp directory."""
    contacts_file = tmp_path / "contacts.json"
    contacts_file.write_text(
        json.dumps({"hocam": "prof@uni.edu", "annem": "mom@gmail.com"}),
        encoding="utf-8",
    )
    r = UnifiedContactResolver()
    r._alias_cache = None  # force fresh load

    with patch("bantz.tools.contact_resolver.CONTACTS_PATH", contacts_file):
        yield r, contacts_file


class TestTier1DirectEmail:
    """Input with '@' is returned as-is."""

    @pytest.mark.asyncio
    async def test_email_passthrough(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("user@example.com")
        assert err is None
        assert resolved == "user@example.com"

    @pytest.mark.asyncio
    async def test_email_with_spaces(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("  user@example.com  ")
        assert err is None
        assert resolved == "user@example.com"


class TestTier2LocalAlias:
    """Names resolved from contacts.json alias file."""

    @pytest.mark.asyncio
    async def test_alias_hit(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("hocam")
        assert err is None
        assert resolved == "prof@uni.edu"

    @pytest.mark.asyncio
    async def test_alias_case_insensitive(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("Hocam")
        assert err is None
        assert resolved == "prof@uni.edu"


class TestTier3Google:
    """Google People API fallback when local alias misses."""

    @pytest.mark.asyncio
    async def test_google_hit(self, resolver):
        r, contacts_file = resolver
        with patch(
            "bantz.tools.contact_resolver.UnifiedContactResolver._google_lookup",
            new_callable=AsyncMock,
            return_value="ahmet@company.com",
        ):
            resolved, err = await r.resolve_addresses("ahmet")

        assert err is None
        assert resolved == "ahmet@company.com"

    @pytest.mark.asyncio
    async def test_google_caches_back(self, resolver):
        """Result from Google is persisted to contacts.json for next time."""
        r, contacts_file = resolver
        with patch(
            "bantz.tools.contact_resolver.UnifiedContactResolver._google_lookup",
            new_callable=AsyncMock,
            return_value="ahmet@company.com",
        ):
            await r.resolve_addresses("ahmet")

        # Verify contacts.json now contains the new alias
        data = json.loads(contacts_file.read_text(encoding="utf-8"))
        assert data["ahmet"] == "ahmet@company.com"

    @pytest.mark.asyncio
    async def test_google_failure_graceful(self, resolver):
        """Google API auth/network failure → None, doesn't crash."""
        r, _ = resolver
        with patch(
            "bantz.tools.contact_resolver.UnifiedContactResolver._google_lookup",
            new_callable=AsyncMock,
            side_effect=Exception("RefreshError: token expired"),
        ):
            # The _google_lookup itself is wrapped, but let's test the
            # actual wrapper by patching the contacts_tool import
            pass

        # Test via the full path — patch contacts_tool.resolve_email
        with patch(
            "bantz.tools.contacts.contacts_tool.resolve_email",
            new_callable=AsyncMock,
            side_effect=Exception("RefreshError: token expired"),
        ):
            resolved, err = await r.resolve_addresses("unknown_person")

        assert resolved is None
        assert "Could not find email address for: unknown_person" in err


class TestMultipleRecipients:
    """Comma-separated recipient lists."""

    @pytest.mark.asyncio
    async def test_mixed_email_and_alias(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("hocam, user@test.com")
        assert err is None
        assert resolved == "prof@uni.edu, user@test.com"

    @pytest.mark.asyncio
    async def test_all_aliases(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("hocam, annem")
        assert err is None
        assert resolved == "prof@uni.edu, mom@gmail.com"

    @pytest.mark.asyncio
    async def test_one_unresolved_blocks_all(self, resolver):
        """If ANY recipient is unresolvable, return error for ALL."""
        r, _ = resolver
        with patch(
            "bantz.tools.contacts.contacts_tool.resolve_email",
            new_callable=AsyncMock,
            return_value="",
        ):
            resolved, err = await r.resolve_addresses("hocam, ghost_person")

        assert resolved is None
        assert "ghost_person" in err

    @pytest.mark.asyncio
    async def test_multiple_unresolved(self, resolver):
        r, _ = resolver
        with patch(
            "bantz.tools.contacts.contacts_tool.resolve_email",
            new_callable=AsyncMock,
            return_value="",
        ):
            resolved, err = await r.resolve_addresses("ghost1, ghost2")

        assert resolved is None
        assert "ghost1" in err
        assert "ghost2" in err


class TestEdgeCases:
    """Empty input, whitespace, etc."""

    @pytest.mark.asyncio
    async def test_empty_string(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("")
        assert resolved is None
        assert "No recipient" in err

    @pytest.mark.asyncio
    async def test_whitespace_only(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("   ")
        assert resolved is None
        assert "No recipient" in err

    @pytest.mark.asyncio
    async def test_trailing_commas(self, resolver):
        r, _ = resolver
        resolved, err = await r.resolve_addresses("hocam,")
        assert err is None
        assert resolved == "prof@uni.edu"


class TestResolveAlias:
    """Synchronous tier-1+2 only for search contexts."""

    def test_email_passthrough(self, resolver):
        r, _ = resolver
        assert r.resolve_alias("user@x.com") == "user@x.com"

    def test_alias_hit(self, resolver):
        r, _ = resolver
        assert r.resolve_alias("hocam") == "prof@uni.edu"

    def test_unknown_returns_input(self, resolver):
        r, _ = resolver
        assert r.resolve_alias("unknown") == "unknown"


class TestAliasManagement:
    """add_alias / all_aliases for _manage_contacts."""

    def test_add_and_list(self, resolver):
        r, contacts_file = resolver
        r.add_alias("yeni", "new@email.com")

        # In-memory
        aliases = r.all_aliases()
        assert aliases["yeni"] == "new@email.com"

        # Persisted
        data = json.loads(contacts_file.read_text(encoding="utf-8"))
        assert data["yeni"] == "new@email.com"

    def test_all_aliases_returns_copy(self, resolver):
        r, _ = resolver
        a = r.all_aliases()
        a["mutated"] = "should_not_persist"
        assert "mutated" not in r.all_aliases()
