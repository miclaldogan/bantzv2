"""
Bantz v3 — Gmail Tool
Full OAuth v3: read, write, search, label, forward, thread view.

Features:
- contacts.json: alias → email mapping ("hocam" → "prof@uni.edu")
- Query builder: from, date, stars, labels, subject, full-text
- compose / reply / forward: LLM-assisted with draft confirmation
- Thread view: see full conversation in chronological order
- Label management: add/remove labels, star/unstar, mark read/unread
- Batch operations: modify multiple messages at once

Actions:
  summary      — LLM-powered unread summary
  count        — unread count
  read         — read single message content
  thread       — full thread view (all messages in conversation)
  search       — filtered search (from, date, stars, label, full-text)
  filter       — natural-language advanced filtering
  send         — direct send (to/subject/body known)
  compose      — LLM generates body, confirms before send
  reply        — reply to a thread
  forward      — forward email to another recipient
  star         — star a message
  unstar       — unstar a message
  mark_read    — mark message(s) as read
  mark_unread  — mark message(s) as unread
  add_label    — add label to message(s)
  remove_label — remove label from message(s)
  contacts     — manage contact aliases
"""
from __future__ import annotations

import asyncio
import base64
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.tools import BaseTool, ToolResult, registry
from bantz.tools.contact_resolver import contact_resolver

MAX_EMAILS = 10
CONTACTS_PATH = Path.home() / ".local" / "share" / "bantz" / "contacts.json"

GMAIL_SUMMARY_PROMPT = """\
You are Bantz. Summarize these unread emails.
Group by importance: urgent/action-required first, then FYI, then newsletters/promos last.
Write 3-5 plain sentences. Mention specific senders and subjects that stand out.
No bullet points. No markdown.\
"""

GMAIL_CONTENT_PROMPT = """\
You are Bantz. Summarize this email content in 2-4 sentences.
Include: who sent it, what they want or say, any action required.
Be direct. No bullet points. No markdown.\
"""

GMAIL_COMPOSE_PROMPT = """\
You are Bantz helping compose a professional email.
Write ONLY the email body — no subject line, no greeting header, no "Dear X" unless specified.
Be natural, concise, and appropriate for the context.
End with a polite closing if appropriate.\
"""


# ── Query builder ─────────────────────────────────────────────────────────────

def build_query(
    from_sender: str = "",
    subject_filter: str = "",
    days_ago: int = 0,           # 0 = no date filter
    starred: bool = False,
    label: str = "",             # "promotions", "social", "updates", "forums"
    unread_only: bool = True,
    full_text: str = "",         # raw Gmail query passthrough
) -> str:
    parts = []
    if unread_only:
        parts.append("label:unread")
    if from_sender:
        resolved = contact_resolver.resolve_alias(from_sender)
        parts.append(f"from:{resolved}")
    if subject_filter:
        parts.append(f"subject:{subject_filter}")
    if days_ago > 0:
        after_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y/%m/%d")
        parts.append(f"after:{after_date}")
    if starred:
        parts.append("is:starred")
    if label:
        _LABEL_MAP = {
            "social": "social", "promotions": "promotions", "promo": "promotions",
            "updates": "updates", "forums": "forums", "important": "important",
        }
        parts.append(f"label:{_LABEL_MAP.get(label.lower(), label)}")
    if full_text:
        parts.append(full_text)
    return " ".join(parts) if parts else "label:unread"


# ── Tool ──────────────────────────────────────────────────────────────────────

class GmailTool(BaseTool):
    name = "gmail"
    description = (
        "Full Gmail management: read inbox, compose, send, search, reply, forward, label, star. "
        "Params: action (unread|compose|compose_and_send|read|search|filter|send|reply|forward|contacts), "
        "to (str), intent/body (str), subject (str), id (str). "
        "compose_and_send: drafts AND sends in one step when recipient and intent are given. "
        "For follow-ups about a previously shown email, use action='read' with its Message ID."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "summary",
        # summary | count | read | thread | search | filter | send | compose
        # reply | forward | star | unstar | mark_read | mark_unread
        # add_label | remove_label | contacts
        message_id: str = "",
        thread_id: str = "",     # for thread view
        from_sender: str = "",
        subject_filter: str = "",
        days_ago: int = 0,
        starred: bool = False,
        label: str = "",
        label_name: str = "",    # for label management
        to: str = "",
        subject: str = "",
        body: str = "",
        intent: str = "",        # for compose: "tell them I can't submit tomorrow"
        raw_query: str = "",     # for filter/compose: full original query text
        alias: str = "",         # for contacts: add alias
        email: str = "",         # for contacts: add email
        message_ids: str = "",   # comma-separated for batch ops
        full_text: str = "",     # raw Gmail query passthrough
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            creds = token_store.get("gmail")
        except TokenNotFoundError as e:
            return ToolResult(success=False, output="", error=str(e))

        if action == "count":
            return await self._count(creds)
        elif action == "read":
            return await self._read_message(creds, message_id)
        elif action == "thread":
            return await self._thread_view(creds, thread_id or message_id)
        elif action == "search":
            return await self._search(creds, from_sender, subject_filter,
                                      days_ago, starred, label, limit, full_text)
        elif action == "filter":
            return await self._filter(creds, raw_query or intent, limit)
        elif action == "send":
            return await self._send(creds, to, subject, body)
        elif action == "compose":
            return await self._compose(creds, to, subject, intent or raw_query)
        elif action == "compose_and_send":
            return await self._compose_and_send(
                creds, to, subject, body, intent or raw_query,
            )
        elif action == "reply":
            return await self._reply(creds, message_id, intent or body)
        elif action == "forward":
            return await self._forward(creds, message_id, to, intent or body)
        elif action == "star":
            return await self._modify_labels(creds, message_id, message_ids,
                                              add_labels=["STARRED"])
        elif action == "unstar":
            return await self._modify_labels(creds, message_id, message_ids,
                                              remove_labels=["STARRED"])
        elif action == "mark_read":
            return await self._modify_labels(creds, message_id, message_ids,
                                              remove_labels=["UNREAD"])
        elif action == "mark_unread":
            return await self._modify_labels(creds, message_id, message_ids,
                                              add_labels=["UNREAD"])
        elif action == "add_label":
            return await self._add_remove_label(creds, message_id, message_ids,
                                                 label_name, add=True)
        elif action == "remove_label":
            return await self._add_remove_label(creds, message_id, message_ids,
                                                 label_name, add=False)
        elif action == "contacts":
            return self._manage_contacts(alias, email)
        else:
            return await self._summary(creds, limit)

    # ── Count ─────────────────────────────────────────────────────────────

    async def _count(self, creds) -> ToolResult:
        count = await asyncio.get_event_loop().run_in_executor(
            None, self._count_sync, creds
        )
        return ToolResult(
            success=True,
            output=f"You have {count} unread emails in your inbox.",
            data={"count": count},
        )

    # ── Summary ───────────────────────────────────────────────────────────

    async def _summary(self, creds, limit: int = 10) -> ToolResult:
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, build_query(), limit
        )
        if not messages:
            return ToolResult(
                success=True,
                output="Your inbox is clean — no unread emails. ✓",
                data={"count": 0},
            )
        lines = [
            f"From: {m['from']}  Subject: {m['subject']}  Snippet: {m['snippet'][:100]}"
            for m in messages
        ]
        summary = await self._llm_summarize("\n".join(lines), GMAIL_SUMMARY_PROMPT)
        return ToolResult(
            success=True,
            output=summary,
            data={"count": len(messages), "messages": messages},
        )

    # ── Read ──────────────────────────────────────────────────────────────

    async def _read_message(self, creds, message_id: str) -> ToolResult:
        if not message_id:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, build_query(), 1
            )
            if not messages:
                return ToolResult(success=True, output="No unread emails.")
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )
        if not content:
            return ToolResult(success=False, output="", error="Could not read email content.")

        summary = await self._llm_summarize(
            f"From: {content['from']}\nSubject: {content['subject']}\n\n{content['body'][:3000]}",
            GMAIL_CONTENT_PROMPT,
        )
        output = summary or (
            f"From: {content['from']}\n"
            f"Subject: {content['subject']}\n\n"
            f"{content['body'][:800]}"
        )
        return ToolResult(
            success=True,
            output=output,
            data={
                "message_id": message_id,
                "from": content["from"],
                "subject": content["subject"],
                "thread_id": content.get("thread_id", ""),
            },
        )

    # ── Search ────────────────────────────────────────────────────────────

    async def _search(
        self, creds,
        from_sender: str, subject_filter: str,
        days_ago: int, starred: bool, label: str, limit: int,
        full_text: str = "",
    ) -> ToolResult:
        query = build_query(
            from_sender=from_sender,
            subject_filter=subject_filter,
            days_ago=days_ago,
            starred=starred,
            label=label,
            full_text=full_text,
        )
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, query, limit
        )

        filter_parts = []
        if from_sender:
            filter_parts.append(f"from: {contact_resolver.resolve_alias(from_sender)}")
        if days_ago:
            filter_parts.append(f"last {days_ago} days")
        if starred:
            filter_parts.append("starred")
        if label:
            filter_parts.append(f"label: {label}")
        filter_desc = ", ".join(filter_parts) or "these criteria"

        if not messages:
            return ToolResult(
                success=True,
                output=f"No emails found for '{filter_desc}'.",
            )

        lines = [
            f"From: {m['from']}  Subject: {m['subject']}  Snippet: {m['snippet'][:100]}"
            for m in messages
        ]
        summary = await self._llm_summarize("\n".join(lines), GMAIL_SUMMARY_PROMPT)
        return ToolResult(
            success=True,
            output=f"🔍 {filter_desc}:\n{summary}",
            data={"count": len(messages), "messages": messages, "query": query},
        )

    # ── Filter (natural language → Gmail query) ──────────────────────────

    async def _filter(self, creds, raw_query: str, limit: int = 10) -> ToolResult:
        """Parse natural language into Gmail query and search."""
        q = self._build_gmail_query(raw_query)
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, q, limit, False
        )
        if not messages:
            return ToolResult(
                success=True,
                output=f"No emails found for '{raw_query}'.",
                data={"query": q},
            )
        lines = [
            f"From: {m['from']}  Subject: {m['subject']}  Date: {m['date']}  "
            f"Snippet: {m['snippet'][:100]}"
            for m in messages
        ]
        summary = await self._llm_summarize("\n".join(lines), GMAIL_SUMMARY_PROMPT)
        return ToolResult(
            success=True,
            output=f"🔍 Filter ({q}):\n{summary}",
            data={"count": len(messages), "messages": messages, "query": q},
        )

    def _build_gmail_query(self, text: str) -> str:
        """
        Convert natural language to Gmail query syntax.

        Examples:
          "emails from github"           → "from:noreply@github.com"
          "emails from my professor"     → "from:prof@uni.edu"
          "starred emails"               → "is:starred"
          "important emails"             → "is:important"
          "emails with attachments"      → "has:attachment"
          "unread emails"                → "is:unread"
          "yesterday's emails"           → "after:2025/06/18 before:2025/06/19"
          "this week's emails"           → "after:2025/06/16"
          "unread emails from ahmet"     → "from:ahmet is:unread"
          "social emails"                → "category:social"
          "promotional emails"           → "category:promotions"
        """
        q_parts: list[str] = []

        # Sender: "emails from X" / "from X"
        m = re.search(
            r"(?:emails?|mails?)\s+from\s+(\S+)",
            text, re.IGNORECASE,
        )
        if not m:
            m = re.search(r"from\s+(\S+?)(?:\s+emails?|\s+mails?|$)", text, re.IGNORECASE)
        if m:
            sender = m.group(1)
            email = contact_resolver.resolve_alias(sender)
            q_parts.append(f"from:{email}")

        # Stars / importance / attachment / unread
        if re.search(r"starred?", text, re.IGNORECASE):
            q_parts.append("is:starred")
        if re.search(r"important", text, re.IGNORECASE):
            q_parts.append("is:important")
        if re.search(r"attach|with\s+attachment", text, re.IGNORECASE):
            q_parts.append("has:attachment")
        if re.search(r"unread", text, re.IGNORECASE):
            q_parts.append("is:unread")

        # Labels / categories
        _CATEGORY_MAP = {
            "social": "social", "promotions": "promotions", "promotional": "promotions",
            "updates": "updates", "forums": "forums",
        }
        for label, gmail_cat in _CATEGORY_MAP.items():
            if label in text.lower():
                q_parts.append(f"category:{gmail_cat}")
                break

        # Label by name ("labeled X" pattern)
        lm = re.search(r"label(?:ed|:)?\s+(\S+)", text, re.IGNORECASE)
        if lm:
            q_parts.append(f"label:{lm.group(1)}")

        # Date ranges using date_parser
        from bantz.core.date_parser import resolve_date
        dt = resolve_date(text)
        if dt:
            after = dt.strftime("%Y/%m/%d")
            before = (dt + timedelta(days=1)).strftime("%Y/%m/%d")
            q_parts.append(f"after:{after} before:{before}")
        else:
            # "this week" / "last N days"
            if "this week" in text.lower():
                from datetime import datetime as _dt
                now = _dt.now()
                monday = now - timedelta(days=now.weekday())
                q_parts.append(f"after:{monday.strftime('%Y/%m/%d')}")
            else:
                dm = re.search(r"last\s+(\d+)\s*days?", text, re.IGNORECASE)
                if dm:
                    days = int(dm.group(1))
                    after = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
                    q_parts.append(f"after:{after}")

        return " ".join(q_parts) if q_parts else "is:unread"

    # ── Send ──────────────────────────────────────────────────────────────

    async def _send(self, creds, to: str, subject: str, body: str) -> ToolResult:
        if not to or not subject or not body:
            return ToolResult(
                success=False, output="",
                error="To send: to, subject and body are required."
            )
        to_resolved, resolve_err = await contact_resolver.resolve_addresses(to)
        if resolve_err:
            return ToolResult(success=False, output="", error=resolve_err)
        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._send_sync, creds, to_resolved, subject, body, None
        )
        if ok:
            return ToolResult(
                success=True,
                output=f"Email sent → {to_resolved}  [{subject}] ✓",
            )
        return ToolResult(success=False, output="", error="Failed to send email.")

    # ── Compose (LLM generates body) ──────────────────────────────────────

    async def _compose(self, creds, to: str, subject: str, intent: str) -> ToolResult:
        """LLM generates email body from intent, returns draft for confirmation."""
        if not to:
            return ToolResult(success=False, output="", error="Recipient not specified.")

        to_resolved, resolve_err = await contact_resolver.resolve_addresses(to)
        if resolve_err:
            return ToolResult(success=False, output="", error=resolve_err)

        # LLM generates body from user intent
        body = await self._llm_compose(to_resolved, subject, intent)
        if not body:
            return ToolResult(success=False, output="", error="Could not generate email content.")

        # Auto-generate subject if not provided
        if not subject:
            subject = await self._generate_subject(body)

        # Return draft — brain will show confirmation
        preview = (
            f"📧 Draft email:\n"
            f"  To: {to_resolved}\n"
            f"  Subject: {subject or '(no subject)'}\n"
            f"  ─────\n"
            f"{body}\n"
            f"  ─────\n"
            f"Shall I send it? (yes/no)"
        )
        return ToolResult(
            success=True,
            output=preview,
            data={
                "draft": True,
                "to": to_resolved,
                "subject": subject,
                "body": body,
            },
        )

    # ── Compose-and-Send (atomic) ─────────────────────────────────────────

    async def _compose_and_send(
        self, creds, to: str, subject: str, body: str, intent: str,
    ) -> ToolResult:
        """Atomic compose + send for complete email requests.

        Required: to (recipient).  At least one of body or intent.
        Optional: subject (auto-generated if missing).
        """
        if not to:
            return ToolResult(
                success=False, output="",
                error="Recipient not specified.",
            )

        to_resolved, resolve_err = await contact_resolver.resolve_addresses(to)
        if resolve_err:
            return ToolResult(success=False, output="", error=resolve_err)

        # Generate body via LLM if only intent provided
        if not body and intent:
            body = await self._llm_compose(to_resolved, subject, intent)
        if not body:
            return ToolResult(
                success=False, output="",
                error="No email content provided.",
            )

        # Auto-generate subject with fallback
        if not subject:
            subject = await self._auto_subject(body)

        # Send directly
        try:
            ok = await asyncio.get_event_loop().run_in_executor(
                None, self._send_sync, creds, to_resolved, subject, body, None,
            )
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Send failed: {exc}",
            )

        if ok:
            return ToolResult(
                success=True,
                output=(
                    f"Email dispatched to {to} ({to_resolved}).\n"
                    f"Subject: {subject}"
                ),
                data={"sent": True, "to": to_resolved, "subject": subject},
            )
        return ToolResult(success=False, output="", error="Failed to send email.")

    async def _auto_subject(self, body: str) -> str:
        """Generate subject via LLM, with a safe fallback."""
        subj = await self._generate_subject(body)
        if subj:
            return subj
        # Fallback: first 5 words of body, capped
        words = body.split()[:5]
        return " ".join(words).rstrip(".,;:!?") if words else "Message from Bantz"

    # ── Reply ─────────────────────────────────────────────────────────────

    async def _reply(self, creds, message_id: str, intent: str) -> ToolResult:
        """Reply to a thread."""
        # Get original message for context
        if not message_id:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, build_query(), 1
            )
            if not messages:
                return ToolResult(success=True, output="No email found to reply to.")
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )
        if not content:
            return ToolResult(success=False, output="", error="Could not read email.")

        # Generate reply body
        context = f"Original email from {content['from']}:\nSubject: {content['subject']}\n\n{content['body'][:1000]}"
        _reply_prompt = f"{GMAIL_COMPOSE_PROMPT}\n\nThis is a REPLY. Context:\n{context}\n\nIntent: {intent}"

        body = await self._llm_compose(content["from"], content["subject"], intent, context=context)
        if not body:
            return ToolResult(success=False, output="", error="Could not generate reply.")

        preview = (
            f"📧 Reply draft:\n"
            f"  To: {content['from']}\n"
            f"  Subject: Re: {content['subject']}\n"
            f"  ─────\n"
            f"{body}\n"
            f"  ─────\n"
            f"Shall I send it? (yes/no)"
        )
        return ToolResult(
            success=True,
            output=preview,
            data={
                "draft": True,
                "to": content["from"],
                "subject": f"Re: {content['subject']}",
                "body": body,
                "thread_id": content.get("thread_id", ""),
                "message_id": message_id,
            },
        )

    # ── Contacts ──────────────────────────────────────────────────────────

    def _manage_contacts(self, alias: str, email: str) -> ToolResult:
        if alias and email:
            contact_resolver.add_alias(alias, email)
            return ToolResult(
                success=True,
                output=f"Contact added: '{alias}' → {email} ✓",
            )
        # List contacts
        all_contacts = contact_resolver.all_aliases()
        if not all_contacts:
            return ToolResult(
                success=True,
                output="No saved contacts.\nTo add: 'save my professor: prof@uni.edu'",
            )
        lines = [f"  {alias}: {email}" for alias, email in all_contacts.items()]
        return ToolResult(
            success=True,
            output="Saved contacts:\n" + "\n".join(lines),
        )

    # ── Thread view ───────────────────────────────────────────────────────

    async def _thread_view(self, creds, thread_id: str) -> ToolResult:
        """Fetch full thread — all messages in a conversation."""
        if not thread_id:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, build_query(), 1
            )
            if not messages:
                return ToolResult(success=True, output="No unread threads.")
            thread_id = messages[0]["thread_id"]

        thread_msgs = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_thread_sync, creds, thread_id
        )
        if not thread_msgs:
            return ToolResult(
                success=False, output="", error="Could not fetch thread."
            )

        lines = []
        for i, m in enumerate(thread_msgs, 1):
            lines.append(
                f"[{i}] From: {m['from']}  ({m['date']})\n"
                f"    Subject: {m['subject']}\n"
                f"    {m['body'][:500]}\n"
            )

        summary = await self._llm_summarize(
            "\n".join(lines),
            "You are Bantz. Summarize this email thread as a conversation. "
            "Note who said what in chronological order and any actions needed. "
            "Be concise. No markdown.",
        )

        return ToolResult(
            success=True,
            output=summary or "\n".join(lines),
            data={
                "thread_id": thread_id,
                "message_count": len(thread_msgs),
                "messages": [
                    {
                        "id": m["id"],
                        "from": m["from"],
                        "subject": m["subject"],
                        "date": m["date"],
                    }
                    for m in thread_msgs
                ],
            },
        )

    # ── Forward ───────────────────────────────────────────────────────────

    async def _forward(
        self, creds, message_id: str, to: str, intent: str
    ) -> ToolResult:
        """Forward an email to another recipient."""
        if not to:
            return ToolResult(
                success=False, output="",
                error="Recipient required for forwarding.",
            )

        if not message_id:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, build_query(), 1
            )
            if not messages:
                return ToolResult(
                    success=True, output="No email found to forward."
                )
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )
        if not content:
            return ToolResult(
                success=False, output="", error="Could not read email."
            )

        to_resolved, resolve_err = await contact_resolver.resolve_addresses(to)
        if resolve_err:
            return ToolResult(success=False, output="", error=resolve_err)
        fwd_subject = f"Fwd: {content['subject']}"

        quoted = (
            f"---------- Forwarded message ----------\n"
            f"From: {content['from']}\n"
            f"Subject: {content['subject']}\n\n"
            f"{content['body'][:3000]}"
        )

        if intent:
            note = await self._llm_compose(
                to_resolved, fwd_subject, intent,
                context=f"Forwarding this email:\n{quoted[:1000]}",
            )
            fwd_body = f"{note}\n\n{quoted}" if note else quoted
        else:
            fwd_body = quoted

        preview = (
            f"Forward draft:\n"
            f"  To: {to_resolved}\n"
            f"  Subject: {fwd_subject}\n"
            f"  ---\n"
            f"{fwd_body[:800]}\n"
            f"  ---\n"
            f"Shall I send it? (yes/no)"
        )
        return ToolResult(
            success=True,
            output=preview,
            data={
                "draft": True,
                "to": to_resolved,
                "subject": fwd_subject,
                "body": fwd_body,
                "original_id": message_id,
            },
        )

    # ── Label management ─────────────────────────────────────────────────

    async def _modify_labels(
        self, creds, message_id: str, message_ids: str,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
    ) -> ToolResult:
        """Add/remove system labels (STARRED, UNREAD, etc.)."""
        ids = self._resolve_message_ids(message_id, message_ids)
        if not ids:
            return ToolResult(
                success=False, output="",
                error="No message ID(s) specified.",
            )

        count = await asyncio.get_event_loop().run_in_executor(
            None, self._batch_modify_sync, creds, ids,
            add_labels or [], remove_labels or [],
        )

        _names = {
            "STARRED": "starred", "UNREAD": "unread",
            "IMPORTANT": "important", "INBOX": "inbox",
        }
        actions = []
        for lbl in (add_labels or []):
            actions.append(f"added {_names.get(lbl, lbl)}")
        for lbl in (remove_labels or []):
            actions.append(f"removed {_names.get(lbl, lbl)}")

        desc = " & ".join(actions)
        return ToolResult(
            success=True,
            output=f"Done: {desc} on {count} message(s).",
            data={"modified_count": count, "message_ids": ids},
        )

    async def _add_remove_label(
        self, creds, message_id: str, message_ids: str,
        label_name: str, add: bool = True,
    ) -> ToolResult:
        """Add or remove a label by name."""
        if not label_name:
            return ToolResult(
                success=False, output="", error="Label name required.",
            )

        ids = self._resolve_message_ids(message_id, message_ids)
        if not ids:
            return ToolResult(
                success=False, output="",
                error="No message ID(s) specified.",
            )

        label_id = await asyncio.get_event_loop().run_in_executor(
            None, self._resolve_label_sync, creds, label_name
        )
        if not label_id:
            return ToolResult(
                success=False, output="",
                error=f"Label '{label_name}' not found.",
            )

        add_labels = [label_id] if add else []
        remove_labels = [label_id] if not add else []
        count = await asyncio.get_event_loop().run_in_executor(
            None, self._batch_modify_sync, creds, ids,
            add_labels, remove_labels,
        )

        verb = "added" if add else "removed"
        return ToolResult(
            success=True,
            output=f"Label '{label_name}' {verb} on {count} message(s).",
            data={
                "modified_count": count,
                "label": label_name,
                "message_ids": ids,
            },
        )

    @staticmethod
    def _resolve_message_ids(message_id: str, message_ids: str) -> list[str]:
        """Parse single ID or comma-separated IDs into a list."""
        ids: list[str] = []
        if message_ids:
            ids = [mid.strip() for mid in message_ids.split(",") if mid.strip()]
        if message_id and message_id not in ids:
            ids.insert(0, message_id)
        return ids

    # ── Sync helpers ──────────────────────────────────────────────────────

    def _build_service(self, creds):
        from googleapiclient.discovery import build
        return build("gmail", "v1", credentials=creds)

    def _count_sync(self, creds) -> int:
        svc = self._build_service(creds)
        result = svc.users().messages().list(
            userId="me", labelIds=["INBOX", "UNREAD"], maxResults=1
        ).execute()
        return result.get("resultSizeEstimate", 0)

    def _fetch_messages_sync(self, creds, query: str, limit: int,
                             inbox_only: bool = True) -> list[dict]:
        svc = self._build_service(creds)
        kwargs: dict[str, Any] = {
            "userId": "me",
            "q": query,
            "maxResults": limit,
        }
        if inbox_only:
            kwargs["labelIds"] = ["INBOX"]
        result = svc.users().messages().list(**kwargs).execute()

        messages = []
        for ref in result.get("messages", []):
            msg = svc.users().messages().get(
                userId="me", id=ref["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            messages.append({
                "id": ref["id"],
                "thread_id": msg.get("threadId", ""),
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            })
        return messages

    def _fetch_content_sync(self, creds, message_id: str) -> Optional[dict]:
        svc = self._build_service(creds)
        msg = svc.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        body = self._extract_body(msg["payload"])
        return {
            "id": message_id,
            "thread_id": msg.get("threadId", ""),
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "body": body,
        }

    def _extract_body(self, payload: dict) -> str:
        mime_type = payload.get("mimeType", "")
        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace").strip()
        elif "parts" in payload:
            for part in payload["parts"]:
                body = self._extract_body(part)
                if body:
                    return body
        return ""

    def _send_sync(
        self, creds, to: str, subject: str, body: str,
        thread_id: Optional[str] = None,
    ) -> bool:
        from email.mime.text import MIMEText
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds)
        msg = MIMEText(body, "plain", "utf-8")
        msg["to"] = to
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        message_body: dict = {"raw": raw}
        if thread_id:
            message_body["threadId"] = thread_id
        svc.users().messages().send(userId="me", body=message_body).execute()
        return True

    def _fetch_thread_sync(self, creds, thread_id: str) -> list[dict]:
        """Fetch all messages in a thread."""
        svc = self._build_service(creds)
        thread = svc.users().threads().get(
            userId="me", id=thread_id, format="full"
        ).execute()
        messages = []
        for msg in thread.get("messages", []):
            headers = {
                h["name"]: h["value"] for h in msg["payload"]["headers"]
            }
            body = self._extract_body(msg["payload"])
            messages.append({
                "id": msg["id"],
                "thread_id": thread_id,
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "body": body,
            })
        return messages

    def _batch_modify_sync(
        self, creds, message_ids: list[str],
        add_labels: list[str], remove_labels: list[str],
    ) -> int:
        """Modify labels on multiple messages. Returns count modified."""
        svc = self._build_service(creds)
        body: dict[str, Any] = {}
        if add_labels:
            body["addLabelIds"] = add_labels
        if remove_labels:
            body["removeLabelIds"] = remove_labels
        count = 0
        for mid in message_ids:
            try:
                svc.users().messages().modify(
                    userId="me", id=mid, body=body
                ).execute()
                count += 1
            except Exception:
                continue
        return count

    def _resolve_label_sync(self, creds, label_name: str) -> str:
        """Resolve a label name to its Gmail label ID."""
        _SYSTEM = {
            "inbox": "INBOX", "starred": "STARRED",
            "important": "IMPORTANT", "unread": "UNREAD",
            "sent": "SENT", "draft": "DRAFT",
            "spam": "SPAM", "trash": "TRASH",
            "social": "CATEGORY_SOCIAL",
            "promotions": "CATEGORY_PROMOTIONS",
            "updates": "CATEGORY_UPDATES",
            "forums": "CATEGORY_FORUMS",
        }
        sys_id = _SYSTEM.get(label_name.lower())
        if sys_id:
            return sys_id
        try:
            svc = self._build_service(creds)
            result = svc.users().labels().list(userId="me").execute()
            for lbl in result.get("labels", []):
                if lbl["name"].lower() == label_name.lower():
                    return lbl["id"]
        except Exception:
            pass
        return ""

    # ── LLM helpers ───────────────────────────────────────────────────────

    async def _generate_subject(self, body: str) -> str:
        """Use LLM to generate a concise email subject from the body."""
        try:
            from bantz.llm.ollama import ollama
            raw = await ollama.chat([
                {"role": "system", "content":
                 "Generate a short, natural email subject line (max 8 words). "
                 "Return ONLY the subject. No quotes, no prefix."},
                {"role": "user", "content": body[:300]},
            ])
            return raw.strip().strip('"').strip("'")
        except Exception:
            return ""

    async def _llm_summarize(self, text: str, system_prompt: str) -> str:
        try:
            from bantz.llm.ollama import ollama
            raw = await ollama.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ])
            return re.sub(r"\*\*(.+?)\*\*", r"\1", raw).strip()
        except Exception:
            return ""

    async def _llm_compose(
        self, to: str, subject: str, intent: str,
        context: str = "",
    ) -> str:
        try:
            from bantz.llm.ollama import ollama
            user_msg = f"To: {to}\nSubject: {subject}\nIntent: {intent}"
            if context:
                user_msg += f"\n\nContext:\n{context}"
            raw = await ollama.chat([
                {"role": "system", "content": GMAIL_COMPOSE_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            return re.sub(r"\*\*(.+?)\*\*", r"\1", raw).strip()
        except Exception:
            return ""


registry.register(GmailTool())


# ── Digest helpers ────────────────────────────────────────────────────────────

def _count_query_sync(creds, query: str) -> int:
    """Count messages matching a Gmail query."""
    from googleapiclient.discovery import build
    svc = build("gmail", "v1", credentials=creds)
    result = svc.users().messages().list(
        userId="me", q=query, maxResults=1,
    ).execute()
    return result.get("resultSizeEstimate", 0)


async def email_stats_today() -> dict:
    """Return email stats for today: received, sent, unread counts."""
    try:
        tool = GmailTool()
        creds = token_store.get("gmail")
        loop = asyncio.get_event_loop()

        unread = await loop.run_in_executor(None, tool._count_sync, creds)
        received = await loop.run_in_executor(
            None, _count_query_sync, creds, "newer_than:1d in:inbox",
        )
        sent = await loop.run_in_executor(
            None, _count_query_sync, creds, "newer_than:1d in:sent",
        )
        return {"received": received, "sent": sent, "unread": unread}
    except Exception:
        return {"received": 0, "sent": 0, "unread": 0}


async def email_stats_week() -> dict:
    """Return email stats for this week."""
    try:
        tool = GmailTool()
        creds = token_store.get("gmail")
        loop = asyncio.get_event_loop()

        unread = await loop.run_in_executor(None, tool._count_sync, creds)
        received = await loop.run_in_executor(
            None, _count_query_sync, creds, "newer_than:7d in:inbox",
        )
        sent = await loop.run_in_executor(
            None, _count_query_sync, creds, "newer_than:7d in:sent",
        )
        return {"received": received, "sent": sent, "unread": unread}
    except Exception:
        return {"received": 0, "sent": 0, "unread": 0}