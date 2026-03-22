"""
Bantz v3 — Google Contacts Tool
Read and search contacts via Google People API.

Actions:
  search     — search contacts by name, email, or phone
  list       — list all contacts (paginated)
  lookup     — find email for a name (for Gmail compose/reply)
  sync       — refresh local cache from Google

Uses People API with contacts.readonly scope.
Local cache in ~/.local/share/bantz/google_contacts_cache.json
for fast offline lookup. Cache refreshes every 24h or on-demand.

Usage by other tools:
    from bantz.tools.contacts import contacts_tool
    email = await contacts_tool.resolve_email("Ahmet")
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.tools import BaseTool, ToolResult, registry

logger = logging.getLogger(__name__)

CACHE_PATH = Path.home() / ".local" / "share" / "bantz" / "google_contacts_cache.json"
CACHE_TTL = 86400  # 24 hours


class ContactsTool(BaseTool):
    name = "contacts"
    description = (
        "Searches and lists Google contacts. "
        "Use for: find contact, who is, phone number, email address, "
        "contact list, search contacts, look up person."
    )
    risk_level = "safe"

    def __init__(self) -> None:
        self._cache: list[dict] | None = None
        self._cache_ts: float = 0.0

    async def execute(
        self,
        action: str = "search",
        # search | list | lookup | sync
        query: str = "",          # name, email, or phone to search
        name: str = "",           # alias for query
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        q = query or name

        if action == "sync":
            return await self._sync()
        elif action == "list":
            return await self._list_contacts(limit)
        elif action == "lookup":
            return await self._lookup_email(q)
        else:
            if not q:
                return await self._list_contacts(limit)
            return await self._search(q, limit)

    # ── Search ────────────────────────────────────────────────────────────

    async def _search(self, query: str, limit: int) -> ToolResult:
        contacts = await self._get_contacts()
        if contacts is None:
            return ToolResult(
                success=False, output="",
                error="Google Contacts not configured. Run: bantz --setup google gmail",
            )

        q_lower = query.lower()
        matches = []
        for c in contacts:
            searchable = f"{c.get('name', '')} {c.get('email', '')} {c.get('phone', '')}".lower()
            if q_lower in searchable:
                matches.append(c)
            if len(matches) >= limit:
                break

        if not matches:
            return ToolResult(
                success=True,
                output=f"No contacts found for '{query}'.",
            )

        lines = [self._format_contact(c) for c in matches]
        return ToolResult(
            success=True,
            output=f"Contacts matching '{query}' ({len(matches)}):\n" + "\n".join(lines),
            data={"count": len(matches), "contacts": matches},
        )

    # ── List ──────────────────────────────────────────────────────────────

    async def _list_contacts(self, limit: int) -> ToolResult:
        contacts = await self._get_contacts()
        if contacts is None:
            return ToolResult(
                success=False, output="",
                error="Google Contacts not configured.",
            )

        if not contacts:
            return ToolResult(success=True, output="No contacts found.")

        shown = contacts[:limit]
        lines = [self._format_contact(c) for c in shown]
        total = len(contacts)
        header = f"Contacts ({len(shown)} of {total}):"
        return ToolResult(
            success=True,
            output=header + "\n" + "\n".join(lines),
            data={"count": len(shown), "total": total, "contacts": shown},
        )

    # ── Lookup (for other tools) ──────────────────────────────────────────

    async def _lookup_email(self, name: str) -> ToolResult:
        """Find email address for a given name."""
        if not name:
            return ToolResult(
                success=False, output="", error="Name required for lookup.",
            )

        contacts = await self._get_contacts()
        if contacts is None:
            return ToolResult(
                success=False, output="",
                error="Google Contacts not configured.",
            )

        name_lower = name.lower()
        best = None
        for c in contacts:
            cname = c.get("name", "").lower()
            if name_lower == cname:
                best = c
                break
            if name_lower in cname and not best:
                best = c

        if best and best.get("email"):
            return ToolResult(
                success=True,
                output=f"{best['name']}: {best['email']}",
                data=best,
            )
        elif best:
            return ToolResult(
                success=True,
                output=f"Found {best['name']} but no email address on file.",
                data=best,
            )
        return ToolResult(
            success=True,
            output=f"No contact found named '{name}'.",
        )

    async def resolve_email(self, name: str) -> str:
        """Public API for other tools — returns email or empty string."""
        contacts = await self._get_contacts()
        if not contacts:
            return ""
        name_lower = name.lower()
        for c in contacts:
            if name_lower in c.get("name", "").lower() and c.get("email"):
                return c["email"]
        return ""

    # ── Sync ──────────────────────────────────────────────────────────────

    async def _sync(self) -> ToolResult:
        try:
            contacts = await self._fetch_from_google()
            self._save_cache(contacts)
            self._cache = contacts
            self._cache_ts = time.time()
            return ToolResult(
                success=True,
                output=f"Contacts synced ✓  {len(contacts)} contacts cached.",
                data={"count": len(contacts)},
            )
        except TokenNotFoundError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=f"Sync failed: {e}",
            )

    # ── Cache / fetch logic ───────────────────────────────────────────────

    async def _get_contacts(self) -> list[dict] | None:
        """Get contacts from cache or Google. Returns None if not configured."""
        # In-memory cache
        if self._cache is not None and (time.time() - self._cache_ts) < CACHE_TTL:
            return self._cache

        # Disk cache
        if CACHE_PATH.exists():
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                ts = data.get("timestamp", 0)
                if (time.time() - ts) < CACHE_TTL:
                    self._cache = data.get("contacts", [])
                    self._cache_ts = ts
                    return self._cache
            except Exception:
                pass

        # Fetch from Google
        try:
            contacts = await self._fetch_from_google()
            self._save_cache(contacts)
            self._cache = contacts
            self._cache_ts = time.time()
            return contacts
        except TokenNotFoundError:
            return None
        except Exception as e:
            logger.warning("Failed to fetch Google contacts: %s", e)
            # Return stale cache if available
            if self._cache is not None:
                return self._cache
            return None

    async def _fetch_from_google(self) -> list[dict]:
        """Fetch contacts from Google People API."""
        creds = token_store.get("gmail")  # shares Gmail OAuth token

        contacts = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_sync, creds
        )
        return contacts

    def _fetch_sync(self, creds) -> list[dict]:
        from googleapiclient.discovery import build

        svc = build("people", "v1", credentials=creds)
        contacts: list[dict] = []
        page_token = ""

        while True:
            result = svc.people().connections().list(
                resourceName="people/me",
                pageSize=200,
                personFields="names,emailAddresses,phoneNumbers,organizations",
                pageToken=page_token or None,
            ).execute()

            for person in result.get("connections", []):
                entry = self._parse_person(person)
                if entry.get("name") or entry.get("email"):
                    contacts.append(entry)

            page_token = result.get("nextPageToken", "")
            if not page_token:
                break

        # Sort by name
        contacts.sort(key=lambda c: c.get("name", "").lower())
        logger.info("Fetched %d contacts from Google People API", len(contacts))
        return contacts

    @staticmethod
    def _parse_person(person: dict) -> dict:
        """Parse a People API person resource into a simple dict."""
        names = person.get("names", [])
        name = names[0].get("displayName", "") if names else ""

        emails = person.get("emailAddresses", [])
        email = emails[0].get("value", "") if emails else ""

        phones = person.get("phoneNumbers", [])
        phone = phones[0].get("value", "") if phones else ""

        orgs = person.get("organizations", [])
        org = orgs[0].get("name", "") if orgs else ""

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "organization": org,
        }

    def _save_cache(self, contacts: list[dict]) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"timestamp": time.time(), "contacts": contacts}
        CACHE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Formatting ────────────────────────────────────────────────────────

    @staticmethod
    def _format_contact(c: dict) -> str:
        parts = [f"  {c.get('name', '(unnamed)')}"]
        if c.get("email"):
            parts.append(f"  {c['email']}")
        if c.get("phone"):
            parts.append(f"  {c['phone']}")
        if c.get("organization"):
            parts.append(f"  ({c['organization']})")
        return " —".join(parts)


contacts_tool = ContactsTool()
registry.register(contacts_tool)
