"""
Bantz v2 — Gmail Tool (v2)
Advanced filtering, contacts, smart compose, reply.

New features:
- contacts.json: takma ad → email mapping ("hocam" → "prof@uni.edu")
- Query builder: from, date, stars, labels, subject
- compose: LLM generates body from intent + confirmation step
- reply: thread reply to last/specific message

Actions:
  summary   — LLM-powered unread summary
  count     — unread count
  read      — read single message content
  search    — filtered search (from, date, stars, label)  filter    — natural-language advanced filtering (sender, date, stars, labels)  send      — direct send (to/subject/body known)
  compose   — LLM generates body, confirms before send
  reply     — reply to a thread
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.tools import BaseTool, ToolResult, registry

MAX_EMAILS = 10
CONTACTS_PATH = Path.home() / ".local" / "share" / "bantz" / "contacts.json"

GMAIL_SUMMARY_PROMPT = """\
You are Bantz — a sharp personal assistant. Summarize these emails the way a smart host would.
- Open with the count: "X emails" or "X unread"
- For each notable one: one line — who sent it and what they want
- Flag urgent/action-required ones first; skip LinkedIn invites and newsletters unless notable
- End with "Which one do you want to read?" if there are multiple
Plain English. No markdown. Max 5 sentences.\
"""

GMAIL_CONTENT_PROMPT = """\
You are Bantz. Read this email and tell the user what it says in 2-3 direct sentences.
Who sent it, what they want or say, any action required. Be specific — no vague summaries.
No filler phrases. No markdown.\
"""

GMAIL_COMPOSE_PROMPT = """\
You are Bantz helping compose a professional email in English.
Write ONLY the email body — no subject line, no greeting header, no "Dear X" unless specified.
Be natural, concise, and appropriate for the context.
End with a polite closing if appropriate.\
"""


# ── Contacts ──────────────────────────────────────────────────────────────────

class Contacts:
    """
    Simple alias → email resolver.
    ~/.local/share/bantz/contacts.json:
    {
      "prof": "professor@university.edu",
      "github": "noreply@github.com",
      "mom": "mom@gmail.com"
    }
    """
    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if CONTACTS_PATH.exists():
            try:
                self._data = json.loads(CONTACTS_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    def resolve(self, name_or_email: str) -> str:
        """Return email for alias, or input if already an email."""
        self._load()
        key = name_or_email.lower().strip()
        return self._data.get(key, name_or_email)

    def add(self, alias: str, email: str) -> None:
        self._load()
        CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._data[alias.lower()] = email
        CONTACTS_PATH.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def all(self) -> dict[str, str]:
        self._load()
        return dict(self._data)


contacts = Contacts()


# ── Query builder ─────────────────────────────────────────────────────────────

def build_query(
    from_sender: str = "",
    subject_filter: str = "",
    days_ago: int = 0,           # 0 = no date filter
    starred: bool = False,
    label: str = "",             # "promotions", "social", "updates", "forums"
    unread_only: bool = True,
) -> str:
    parts = []
    if unread_only:
        parts.append("label:unread")
    if from_sender:
        resolved = contacts.resolve(from_sender)
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
            "social": "social", "promotions": "promotions",
            "updates": "updates", "forums": "forums", "important": "important",
        }
        parts.append(f"label:{_LABEL_MAP.get(label.lower(), label)}")
    return " ".join(parts) if parts else "label:unread"


# ── Tool ──────────────────────────────────────────────────────────────────────

class GmailTool(BaseTool):
    name = "gmail"
    description = (
        "Reads, filters, composes and sends Gmail messages. "
        "Use for: mail, gmail, inbox, summarize emails, "
        "emails from X, starred emails, this week's emails, "
        "send email, compose email, reply to mail, unread count."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "summary",
        # summary | count | read | search | filter | send | compose | reply | contacts
        message_id: str = "",
        from_sender: str = "",
        subject_filter: str = "",
        days_ago: int = 0,
        starred: bool = False,
        label: str = "",
        to: str = "",
        subject: str = "",
        body: str = "",
        intent: str = "",        # for compose: "tell them I cannot submit tomorrow"
        raw_query: str = "",     # for filter/compose: full original query text
        alias: str = "",         # for contacts: add alias
        email: str = "",         # for contacts: add email
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
        elif action == "search":
            return await self._search(creds, from_sender, subject_filter,
                                      days_ago, starred, label, limit)
        elif action == "filter":
            return await self._filter(creds, raw_query or intent, limit)
        elif action == "send":
            return await self._send(creds, to, subject, body)
        elif action == "compose":
            return await self._compose(creds, to, subject, intent or raw_query)
        elif action == "reply":
            return await self._reply(creds, message_id, intent or body)
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
            output=f"{count} unread mail(s) in your inbox.",
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
                output="Inbox clear — no unread mail. ✓",
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
                return ToolResult(success=True, output="No unread mail.")
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )
        if not content:
            return ToolResult(success=False, output="", error="Could not read mail content.")

        summary = await self._llm_summarize(
            f"From: {content['from']}\nSubject: {content['subject']}\n\n{content['body'][:3000]}",
            GMAIL_CONTENT_PROMPT,
        )
        output = summary or (
            f"From: {content['from']}\nSubject: {content['subject']}\n\n{content['body'][:800]}"
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
    ) -> ToolResult:
        query = build_query(
            from_sender=from_sender,
            subject_filter=subject_filter,
            days_ago=days_ago,
            starred=starred,
            label=label,
        )
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, query, limit
        )

        filter_parts = []
        if from_sender:
            filter_parts.append(f"from: {contacts.resolve(from_sender)}")
        if days_ago:
            filter_parts.append(f"last {days_ago} days")
        if starred:
            filter_parts.append("starred")
        if label:
            filter_parts.append(f"label: {label}")
        filter_desc = ", ".join(filter_parts) or "filter"

        if not messages:
            return ToolResult(
                success=True,
                output=f"No mail found for '{filter_desc}'.",
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
        """Parse Turkish natural language into Gmail query and search."""
        q = self._build_gmail_query(raw_query)
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, q, limit, False
        )
        if not messages:
            return ToolResult(
                success=True,
                output=f"No mail found for '{raw_query}'.",
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
            output=f"🔍 Filtre ({q}):\n{summary}",
            data={"count": len(messages), "messages": messages, "query": q},
        )

    def _build_gmail_query(self, text: str) -> str:
        """
        Convert natural language to Gmail query syntax.

        Examples:
          "emails from github"     → "from:noreply@github.com"
          "starred emails"         → "is:starred"
          "important emails"       → "is:important"
          "emails with attachments" → "has:attachment"
          "unread emails"          → "is:unread"
          "emails from this week"  → "after:2025/06/16"
          "social emails"          → "category:social"
          "promotional emails"     → "category:promotions"
        """
        q_parts: list[str] = []

        # Sender — English: "emails from X" / "mail from X" / "from X"
        m = re.search(
            r"(?:mails?|emails?|messages?)\s+from\s+([\w\s\u00C0-\u024F]{2,40}?)(?:\?|$|\.|please)",
            text, re.IGNORECASE,
        )
        if not m:
            m = re.search(r"\bfrom[:\s]+([\w.\-\u00C0-\u024F]{2,40})", text, re.IGNORECASE)
        if m:
            sender = m.group(1).strip().rstrip("?.,;:")
            email = contacts.resolve(sender)
            q_parts.append(f"from:{email}")

        # Sender — Turkish: "X'den gelen mailler"
        elif not m:
            m_tr = re.search(
                r"(\S+?)[''\u2019]?(?:den|dan|tan|ten|ndan|nden)\s+(?:mailler?|mail|gelen)",
                text, re.IGNORECASE,
            )
            if m_tr:
                sender = m_tr.group(1)
                email = contacts.resolve(sender)
                q_parts.append(f"from:{email}")

        # Stars / importance / attachment / unread
        if re.search(r"starred|star", text, re.IGNORECASE):
            q_parts.append("is:starred")
        if re.search(r"important", text, re.IGNORECASE):
            q_parts.append("is:important")
        if re.search(r"attachment|attached", text, re.IGNORECASE):
            q_parts.append("has:attachment")
        if re.search(r"unread", text, re.IGNORECASE):
            q_parts.append("is:unread")

        # Labels / categories
        _CATEGORY_MAP = {
            "social": "social", "promotions": "promotions",
            "updates": "updates", "forums": "forums",
        }
        for tr_label, gmail_cat in _CATEGORY_MAP.items():
            if tr_label in text.lower():
                q_parts.append(f"category:{gmail_cat}")
                break

        # Label by name (etiketli pattern)
        lm = re.search(r"(\S+)\s+label", text, re.IGNORECASE)
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
            # "bu hafta" / "son X gün"
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
        to_resolved = contacts.resolve(to)
        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._send_sync, creds, to_resolved, subject, body, None
        )
        if ok:
            return ToolResult(
                success=True,
                output=f"Mail sent → {to_resolved}  [{subject}] ✓",
            )
        return ToolResult(success=False, output="", error="Failed to send mail.")

    # ── Compose (LLM generates body) ──────────────────────────────────────

    async def _compose(self, creds, to: str, subject: str, intent: str) -> ToolResult:
        """LLM generates email body from intent, returns draft for confirmation."""
        if not to:
            return ToolResult(success=False, output="", error="Recipient not specified.")

        to_resolved = contacts.resolve(to)

        # LLM generates body from user intent
        body = await self._llm_compose(to_resolved, subject, intent)
        if not body:
            return ToolResult(success=False, output="", error="Could not generate mail content.")

        # Auto-generate subject if not provided
        if not subject:
            subject = await self._generate_subject(body)

        # Return draft — brain will show confirmation
        preview = (
            f"📧 Draft mail:\n  To: {to_resolved}\n  Subject: {subject or '(no subject)'}\n  ─────\n{body}\n  ─────\nShall I send it? (yes/no)"
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

    # ── Reply ─────────────────────────────────────────────────────────────

    async def _reply(self, creds, message_id: str, intent: str) -> ToolResult:
        """Reply to a thread."""
        # Get original message for context
        if not message_id:
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, build_query(), 1
            )
            if not messages:
                return ToolResult(success=True, output="No mail to reply to.")
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )
        if not content:
            return ToolResult(success=False, output="", error="Could not read mail.")

        # Generate reply body
        context = f"Original email from {content['from']}:\nSubject: {content['subject']}\n\n{content['body'][:1000]}"
        reply_prompt = f"{GMAIL_COMPOSE_PROMPT}\n\nThis is a REPLY. Context:\n{context}\n\nIntent: {intent}"

        body = await self._llm_compose(content["from"], content["subject"], intent, context=context)
        if not body:
            return ToolResult(success=False, output="", error="Could not generate reply.")

        preview = (
            f"📧 Reply draft:\n  To: {content['from']}\n  Subject: Re: {content['subject']}\n  ─────\n{body}\n  ─────\nShall I send it? (yes/no)"
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
            contacts.add(alias, email)
            return ToolResult(
                success=True,
                output=f"Contact saved: '{alias}' → {email} ✓",
            )
        # List contacts
        all_contacts = contacts.all()
        if not all_contacts:
            return ToolResult(
                success=True,
                output="No saved contacts.\nTo add: 'save prof@uni.edu as prof'",
            )
        lines = [f"  {alias}: {email}" for alias, email in all_contacts.items()]
        return ToolResult(
            success=True,
            output="Saved contacts:\n" + "\n".join(lines),
        )

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

    # ── LLM helpers ───────────────────────────────────────────────────────

    async def _generate_subject(self, body: str) -> str:
        """Use LLM to generate a concise email subject from the body."""
        try:
            from bantz.llm.ollama import ollama
            raw = await ollama.chat([
                {"role": "system", "content":
                 "Generate a short, natural English email subject line (max 8 words). "
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