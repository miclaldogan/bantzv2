"""
Bantz v2 — Gmail Tool
Unread summary, full content reading, sender-filtered search, send.
Uses gmail_token.json (personal account).

Triggers: mail, gmail, gelen kutusu, mailleri özetle, X'ten mailler, belirli mail oku
"""
from __future__ import annotations

import asyncio
import base64
import re
from email import message_from_bytes
from typing import Any

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.tools import BaseTool, ToolResult, registry

MAX_EMAILS = 10

GMAIL_SUMMARY_PROMPT = """\
You are Bantz. Summarize these unread emails in Turkish.
Group by importance: urgent/action-required first, then FYI, then newsletters/promos last.
Write 3-5 plain sentences. Mention specific senders and subjects that stand out.
No bullet points. No markdown.\
"""

GMAIL_CONTENT_PROMPT = """\
You are Bantz. Summarize this email content in Turkish in 2-4 sentences.
Include: who sent it, what they want or say, any action required.
Be direct. No bullet points. No markdown.\
"""


class GmailTool(BaseTool):
    name = "gmail"
    description = (
        "Reads, searches, summarizes Gmail messages and sends email. "
        "Use for: mail, gmail, gelen kutusu, maillerimi göster, mailleri özetle, "
        "X'ten mailler, belirli maili oku, mail gönder, kaç mail var."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "summary",
        # "summary" | "count" | "read" | "search" | "send"
        message_id: str = "",       # for action="read"
        from_sender: str = "",      # for action="search"
        subject_filter: str = "",   # for action="search"
        to: str = "",
        subject: str = "",
        body: str = "",
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
            return await self._search(creds, from_sender, subject_filter, limit)
        elif action == "send":
            return await self._send(creds, to, subject, body)
        else:
            return await self._summary(creds, limit)

    # ── Count ─────────────────────────────────────────────────────────────

    async def _count(self, creds) -> ToolResult:
        count = await asyncio.get_event_loop().run_in_executor(
            None, self._count_sync, creds
        )
        return ToolResult(
            success=True,
            output=f"Gelen kutusunda {count} okunmamış mail var.",
            data={"count": count},
        )

    # ── Summary ───────────────────────────────────────────────────────────

    async def _summary(self, creds, limit: int = 10) -> ToolResult:
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, None, None, limit
        )
        if not messages:
            return ToolResult(
                success=True,
                output="Gelen kutun temiz — okunmamış mail yok. ✓",
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

    # ── Read single message ───────────────────────────────────────────────

    async def _read_message(self, creds, message_id: str) -> ToolResult:
        if not message_id:
            # No ID given — read the most recent unread
            messages = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_messages_sync, creds, None, None, 1
            )
            if not messages:
                return ToolResult(success=True, output="Okunmamış mail yok.")
            message_id = messages[0]["id"]

        content = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_content_sync, creds, message_id
        )

        if not content:
            return ToolResult(success=False, output="", error="Mail içeriği okunamadı.")

        summary = await self._llm_summarize(
            f"From: {content['from']}\nSubject: {content['subject']}\n\n{content['body'][:3000]}",
            GMAIL_CONTENT_PROMPT,
        )

        output = summary if summary else (
            f"Gönderen: {content['from']}\n"
            f"Konu: {content['subject']}\n\n"
            f"{content['body'][:800]}"
        )
        return ToolResult(
            success=True,
            output=output,
            data={"message_id": message_id, "from": content["from"], "subject": content["subject"]},
        )

    # ── Search / filter ───────────────────────────────────────────────────

    async def _search(self, creds, from_sender: str, subject_filter: str, limit: int) -> ToolResult:
        messages = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_messages_sync, creds, from_sender, subject_filter, limit
        )
        if not messages:
            filter_desc = from_sender or subject_filter or "bu kriter"
            return ToolResult(
                success=True,
                output=f"'{filter_desc}' için mail bulunamadı.",
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

    # ── Send ──────────────────────────────────────────────────────────────

    async def _send(self, creds, to: str, subject: str, body: str) -> ToolResult:
        if not to or not subject or not body:
            return ToolResult(
                success=False, output="",
                error="Göndermek için: to, subject ve body gerekli."
            )
        ok = await asyncio.get_event_loop().run_in_executor(
            None, self._send_sync, creds, to, subject, body
        )
        if ok:
            return ToolResult(success=True, output=f"Mail gönderildi → {to}  [{subject}] ✓")
        return ToolResult(success=False, output="", error="Mail gönderilemedi.")

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

    def _fetch_messages_sync(
        self, creds,
        from_sender: str | None,
        subject_filter: str | None,
        limit: int,
    ) -> list[dict]:
        svc = self._build_service(creds)

        # Build query string
        query_parts = ["label:unread"]
        if from_sender:
            query_parts.append(f"from:{from_sender}")
        if subject_filter:
            query_parts.append(f"subject:{subject_filter}")
        query = " ".join(query_parts)

        result = svc.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            q=query,
            maxResults=limit,
        ).execute()

        messages = []
        for ref in result.get("messages", []):
            msg = svc.users().messages().get(
                userId="me", id=ref["id"], format="metadata",
                metadataHeaders=["From", "Subject"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            messages.append({
                "id": ref["id"],
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "snippet": msg.get("snippet", ""),
            })
        return messages

    def _fetch_content_sync(self, creds, message_id: str) -> dict | None:
        """Fetch full message body."""
        svc = self._build_service(creds)
        msg = svc.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        body = self._extract_body(msg["payload"])

        return {
            "id": message_id,
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "body": body,
        }

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract plain text body from Gmail payload."""
        body = ""
        mime_type = payload.get("mimeType", "")

        if mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        elif "parts" in payload:
            for part in payload["parts"]:
                body = self._extract_body(part)
                if body:
                    break

        return body.strip()

    def _send_sync(self, creds, to: str, subject: str, body: str) -> bool:
        from email.mime.text import MIMEText
        from googleapiclient.discovery import build
        svc = build("gmail", "v1", credentials=creds)
        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True

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


registry.register(GmailTool())