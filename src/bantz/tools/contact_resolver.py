"""
Bantz v3 — Unified Contact Resolver (#254)

Single source of truth for resolving human names → email addresses.
Eliminates the "Two-Headed Dragon" problem where gmail.py and contacts.py
maintained separate, incompatible resolution paths.

Resolution cascade:
  1. Contains '@' → already an email, keep as-is.
  2. Local alias file (~/.local/share/bantz/contacts.json).
  3. Google People API (via ContactsTool cache / live fetch).
  4. If found via Google, cache back to contacts.json for next time.

Supports comma-separated lists ("iclal, hocam@uni.edu").

Usage:
    from bantz.tools.contact_resolver import contact_resolver

    resolved, err = await contact_resolver.resolve_addresses("iclal, hocam")
    if err:
        return ToolResult(success=False, output=err)
    # resolved == "iclaldgn@gmail.com, prof@uni.edu"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("bantz.contact_resolver")

CONTACTS_PATH = Path.home() / ".local" / "share" / "bantz" / "contacts.json"


class UnifiedContactResolver:
    """Resolves human names to email addresses via a 3-tier cascade."""

    def __init__(self) -> None:
        self._alias_cache: dict[str, str] | None = None

    # ── Public API ────────────────────────────────────────────────────────

    async def resolve_addresses(
        self, names_or_emails: str,
    ) -> tuple[str | None, str | None]:
        """Resolve a comma-separated string of names/emails.

        Returns:
            (resolved_string, None) on success — all contacts found.
            (None, error_message) if ANY contact is unresolvable.
        """
        if not names_or_emails or not names_or_emails.strip():
            return None, "No recipient specified."

        items = [item.strip() for item in names_or_emails.split(",")]
        items = [item for item in items if item]  # drop empties

        if not items:
            return None, "No recipient specified."

        resolved: list[str] = []
        unresolved: list[str] = []

        for item in items:
            email = await self._resolve_single(item)
            if email:
                resolved.append(email)
            else:
                unresolved.append(item)

        if unresolved:
            names = ", ".join(unresolved)
            return None, f"Could not find email address for: {names}"

        return ", ".join(resolved), None

    async def resolve_single(self, name_or_email: str) -> str:
        """Convenience — resolve a single name/email.  Returns email or ''."""
        result = await self._resolve_single(name_or_email.strip())
        return result or ""

    def resolve_alias(self, name_or_email: str) -> str:
        """Synchronous tier-1+2 only: '@' check + local alias file.

        Used by ``build_query`` and ``_build_gmail_query`` in gmail.py
        where the full async Google cascade is unnecessary (search context,
        not sending).  Returns the input unchanged if not found.
        """
        item = name_or_email.strip()
        if not item:
            return item
        if "@" in item:
            return item
        aliases = self._load_aliases()
        return aliases.get(item.lower(), item)

    # ── Resolution cascade ────────────────────────────────────────────────

    async def _resolve_single(self, item: str) -> str | None:
        """Resolve one name-or-email through the 3-tier cascade.

        Returns the email address or None if unresolvable.
        """
        if not item:
            return None

        # Tier 1: already an email
        if "@" in item:
            log.debug("Tier 1 (email): %s", item)
            return item

        # Tier 2: local alias file
        aliases = self._load_aliases()
        key = item.lower()
        if key in aliases:
            log.debug("Tier 2 (alias): %s → %s", item, aliases[key])
            return aliases[key]

        # Tier 3: Google People API (cache + live)
        email = await self._google_lookup(item)
        if email:
            log.info("Tier 3 (Google): %s → %s", item, email)
            # Cache back to alias file for next time
            self._save_alias(key, email)
            return email

        log.warning("Contact unresolved after all tiers: %s", item)
        return None

    # ── Tier 2: Local alias file ──────────────────────────────────────────

    def _load_aliases(self) -> dict[str, str]:
        """Load contacts.json alias map (lazy, cached in memory)."""
        if self._alias_cache is not None:
            return self._alias_cache

        if CONTACTS_PATH.exists():
            try:
                self._alias_cache = json.loads(
                    CONTACTS_PATH.read_text(encoding="utf-8")
                )
            except Exception:
                self._alias_cache = {}
        else:
            self._alias_cache = {}

        return self._alias_cache

    def _save_alias(self, alias: str, email: str) -> None:
        """Persist a new alias to contacts.json and update in-memory cache."""
        aliases = self._load_aliases()
        aliases[alias.lower()] = email
        self._alias_cache = aliases
        try:
            CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONTACTS_PATH.write_text(
                json.dumps(aliases, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to write contacts.json: %s", exc)

    def invalidate_cache(self) -> None:
        """Force re-read of contacts.json on next resolve."""
        self._alias_cache = None

    def add_alias(self, alias: str, email: str) -> None:
        """Add an alias → email mapping and persist to contacts.json."""
        self._save_alias(alias.lower(), email)

    def all_aliases(self) -> dict[str, str]:
        """Return a copy of the local alias map."""
        return dict(self._load_aliases())

    # ── Tier 3: Google People API ─────────────────────────────────────────

    async def _google_lookup(self, name: str) -> str | None:
        """Look up a name via Google People API through ContactsTool.

        Wraps the call in try-except to handle auth/network failures
        gracefully (returns None instead of crashing).
        """
        try:
            # Lazy import to avoid circular dependencies (#254 edge case 3)
            from bantz.tools.contacts import contacts_tool

            email = await contacts_tool.resolve_email(name)
            return email if email else None
        except Exception as exc:
            log.warning(
                "Google Contacts API unavailable: %s — "
                "falling back to local-only resolution",
                exc,
            )
            return None


# ── Singleton ────────────────────────────────────────────────────────────────

contact_resolver = UnifiedContactResolver()
