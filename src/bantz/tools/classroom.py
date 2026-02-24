"""
Bantz v2 — Google Classroom Tool
Courses, assignments, announcements.
Uses classroom_token.json (separate school account).

Triggers: assignment, announcement, classroom, deadline, courses, due today
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from bantz.auth.token_store import token_store, TokenNotFoundError
from bantz.tools import BaseTool, ToolResult, registry

CLASSROOM_SUMMARY_PROMPT = """\
You are Bantz. Summarize these assignments in English.
Focus on urgency — due today or tomorrow first, then upcoming.
Write 2-4 plain sentences. No bullet points. No markdown.
If something is overdue, mention it clearly.\
"""


class ClassroomTool(BaseTool):
    name = "classroom"
    description = (
        "Fetches Google Classroom courses, assignments and announcements. "
        "Use for: assignments, homework, announcements, classroom, deadline, "
        "my courses, course list, due today."
    )
    risk_level = "safe"

    async def execute(
        self,
        action: str = "assignments",
        # "assignments" | "announcements" | "due_today" | "courses"
        **kwargs: Any,
    ) -> ToolResult:
        try:
            creds = token_store.get("classroom")
        except TokenNotFoundError as e:
            return ToolResult(success=False, output="", error=str(e))

        if action == "courses":
            return await self._get_courses(creds)
        elif action == "announcements":
            return await self._get_announcements(creds)
        elif action == "due_today":
            return await self._get_due_today(creds)
        else:
            return await self._get_assignments(creds)

    # ── Courses ───────────────────────────────────────────────────────────

    async def _get_courses(self, creds) -> ToolResult:
        courses = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_courses_sync, creds
        )
        if not courses:
            return ToolResult(success=True, output="No active courses found.")

        lines = []
        for c in courses:
            section = f" — {c['section']}" if c.get("section") else ""
            teacher = f"  👤 {c['teacher']}" if c.get("teacher") else ""
            lines.append(f"  📚 {c['name']}{section}{teacher}")

        return ToolResult(
            success=True,
            output=f"You are enrolled in {len(courses)} course(s):\n" + "\n".join(lines),
            data={"count": len(courses), "courses": courses},
        )

    # ── Assignments ───────────────────────────────────────────────────────

    async def _get_assignments(self, creds) -> ToolResult:
        courses, assignments = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_assignments_sync, creds
        )
        if not assignments:
            return ToolResult(success=True, output="No active assignments. 🎉", data={"count": 0})

        now = datetime.now(timezone.utc)
        overdue, urgent, upcoming = [], [], []

        for a in assignments:
            due = a.get("due_dt")
            course = a.get("course", "")
            title = a.get("title", "(untitled)")
            if due:
                delta = (due - now).days
                if delta < 0:
                    overdue.append(f"  ⚠️  OVERDUE: {title}  [{course}]")
                elif delta == 0:
                    urgent.append(f"  🔴 Today: {title}  [{course}]")
                elif delta == 1:
                    urgent.append(f"  🟡 Tomorrow: {title}  [{course}]")
                else:
                    upcoming.append(f"  🟢 {due.strftime('%d %b')}: {title}  [{course}]")
            else:
                upcoming.append(f"  ⬜ No deadline: {title}  [{course}]")

        all_lines = overdue + urgent + upcoming
        summary = await self._llm_summarize(all_lines)
        return ToolResult(
            success=True,
            output=summary if summary else "\n".join(all_lines),
            data={"count": len(assignments), "overdue": len(overdue)},
        )

    async def _get_due_today(self, creds) -> ToolResult:
        _, assignments = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_assignments_sync, creds
        )
        now = datetime.now(timezone.utc)
        today = [a for a in assignments if a.get("due_dt") and a["due_dt"].date() == now.date()]
        if not today:
            return ToolResult(success=True, output="No assignments due today. ✓")
        lines = [f"  🔴 {a['title']}  [{a.get('course','')}]" for a in today]
        return ToolResult(
            success=True,
            output="Due today:\n" + "\n".join(lines),
            data={"count": len(today)},
        )

    # ── Announcements ─────────────────────────────────────────────────────

    async def _get_announcements(self, creds) -> ToolResult:
        announcements = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_announcements_sync, creds
        )
        if not announcements:
            return ToolResult(success=True, output="No new announcements.")
        lines = [f"  [{a['course']}]  {a['text'][:120]}" for a in announcements[:5]]
        return ToolResult(
            success=True,
            output="Announcements:\n" + "\n".join(lines),
            data={"count": len(announcements)},
        )

    # ── Sync helpers ──────────────────────────────────────────────────────

    def _fetch_courses_sync(self, creds) -> list[dict]:
        from googleapiclient.discovery import build
        svc = build("classroom", "v1", credentials=creds)
        result = svc.courses().list(courseStates=["ACTIVE"]).execute()
        courses = []
        for c in result.get("courses", []):
            teacher = ""
            if owners := c.get("ownerId"):
                try:
                    profile = svc.userProfiles().get(userId=owners).execute()
                    teacher = profile.get("name", {}).get("fullName", "")
                except Exception:
                    pass
            courses.append({
                "id": c["id"],
                "name": c.get("name", ""),
                "section": c.get("section", ""),
                "teacher": teacher,
            })
        return courses

    def _fetch_assignments_sync(self, creds) -> tuple[list, list]:
        from googleapiclient.discovery import build
        svc = build("classroom", "v1", credentials=creds)
        courses = svc.courses().list(courseStates=["ACTIVE"]).execute().get("courses", [])
        assignments = []
        for course in courses:
            try:
                cw = svc.courses().courseWork().list(
                    courseId=course["id"],
                    courseWorkStates=["PUBLISHED"],
                    orderBy="dueDate asc",
                    maxResults=10,
                ).execute()
                for item in cw.get("courseWork", []):
                    due_dt = None
                    if d := item.get("dueDate"):
                        try:
                            due_dt = datetime(
                                d.get("year", 2026), d.get("month", 1), d.get("day", 1),
                                tzinfo=timezone.utc,
                            )
                        except Exception:
                            pass
                    assignments.append({
                        "title": item.get("title", ""),
                        "course": course.get("name", ""),
                        "due_dt": due_dt,
                    })
            except Exception:
                continue
        return courses, assignments

    def _fetch_announcements_sync(self, creds) -> list[dict]:
        from googleapiclient.discovery import build
        svc = build("classroom", "v1", credentials=creds)
        courses = svc.courses().list(courseStates=["ACTIVE"]).execute().get("courses", [])
        announcements = []
        for course in courses:
            try:
                anns = svc.courses().announcements().list(
                    courseId=course["id"],
                    announcementStates=["PUBLISHED"],
                    orderBy="updateTime desc",
                    maxResults=3,
                ).execute()
                for ann in anns.get("announcements", []):
                    announcements.append({
                        "course": course.get("name", ""),
                        "text": ann.get("text", ""),
                    })
            except Exception:
                continue
        return announcements

    async def _llm_summarize(self, lines: list[str]) -> str:
        try:
            from bantz.llm.ollama import ollama
            raw = await ollama.chat([
                {"role": "system", "content": CLASSROOM_SUMMARY_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ])
            return re.sub(r"\*\*(.+?)\*\*", r"\1", raw).strip()
        except Exception:
            return ""


registry.register(ClassroomTool())