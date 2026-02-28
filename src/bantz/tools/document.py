"""
Bantz v2 — Document Tool

Reads and summarizes documents: PDF, TXT, MD, DOCX.
Sends extracted text to Ollama for summarization or Q&A.

Usage:
    result = await tool.execute(path="~/report.pdf", action="summarize")
    result = await tool.execute(path="~/notes.md", action="read")
    result = await tool.execute(path="~/paper.pdf", action="ask", question="What is the conclusion?")
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bantz.tools import BaseTool, ToolResult, registry

# Max chars to send to LLM — avoids context overflow
_MAX_TEXT = 4000


class DocumentTool(BaseTool):
    name = "document"
    description = (
        "Reads and summarizes documents: PDF, TXT, MD, DOCX. "
        "Use for: summarize file, read document, what's in this file, explain this PDF."
    )
    risk_level = "safe"

    async def execute(
        self,
        path: str = "",
        action: str = "summarize",
        question: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="", error="No file path provided.")

        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return ToolResult(
                success=False, output="",
                error=f"File not found: {resolved}",
            )
        if not resolved.is_file():
            return ToolResult(
                success=False, output="",
                error=f"Not a file: {resolved}",
            )

        # Extract text based on file type
        suffix = resolved.suffix.lower()
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_text, resolved, suffix,
            )
        except Exception as exc:
            return ToolResult(
                success=False, output="",
                error=f"Failed to read {resolved.name}: {exc}",
            )

        if not text or not text.strip():
            return ToolResult(
                success=False, output="",
                error=f"No readable text found in {resolved.name}.",
            )

        char_count = len(text)
        truncated = text[:_MAX_TEXT]
        trunc_note = f" (showing first {_MAX_TEXT} of {char_count} chars)" if char_count > _MAX_TEXT else ""

        # ── Action dispatch ───────────────────────────────────────────────

        if action == "read":
            # Just return the raw text
            return ToolResult(
                success=True,
                output=f"📄 {resolved.name}{trunc_note}\n\n{truncated}",
                data={"path": str(resolved), "chars": char_count},
            )

        if action == "ask" and question:
            summary = await self._ask_llm(truncated, question, resolved.name)
            return ToolResult(
                success=True,
                output=summary,
                data={"path": str(resolved), "chars": char_count, "action": "ask"},
            )

        # Default: summarize
        summary = await self._summarize_llm(truncated, resolved.name)
        return ToolResult(
            success=True,
            output=summary,
            data={"path": str(resolved), "chars": char_count, "action": "summarize"},
        )

    # ── Text extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_text(path: Path, suffix: str) -> str:
        if suffix == ".pdf":
            return DocumentTool._read_pdf(path)
        elif suffix == ".docx":
            return DocumentTool._read_docx(path)
        elif suffix in (".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".py", ".js", ".ts", ".html", ".xml"):
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            # Try reading as text
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from PDF. Tries pymupdf first, then pdfplumber."""
        # Try PyMuPDF (fitz)
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except ImportError:
            pass

        # No PDF library available
        raise ImportError(
            "No PDF library found. Install one: pip install pymupdf  OR  pip install pdfplumber"
        )

    @staticmethod
    def _read_docx(path: Path) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError(
                "python-docx not found. Install: pip install python-docx"
            )

    # ── LLM interaction ───────────────────────────────────────────────────

    @staticmethod
    async def _summarize_llm(text: str, filename: str) -> str:
        from bantz.llm.ollama import ollama
        messages = [
            {"role": "system", "content": (
                "You are a document summarizer. Summarize the following document text "
                "clearly and concisely in 3-5 bullet points. Include key facts, numbers, "
                "and conclusions. English only, plain text."
            )},
            {"role": "user", "content": f"Document: {filename}\n\n{text}"},
        ]

        # Try Gemini first if available
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages, temperature=0.2)
                if raw and len(raw) > 10:
                    return f"📄 Summary of {filename}:\n\n{raw}"
        except Exception:
            pass

        raw = await ollama.chat(messages)
        return f"📄 Summary of {filename}:\n\n{raw}"

    @staticmethod
    async def _ask_llm(text: str, question: str, filename: str) -> str:
        from bantz.llm.ollama import ollama
        messages = [
            {"role": "system", "content": (
                "You are a document Q&A assistant. Answer the user's question based ONLY "
                "on the provided document text. If the answer isn't in the document, say so. "
                "Be concise. English only, plain text."
            )},
            {"role": "user", "content": (
                f"Document: {filename}\n\n{text}\n\nQuestion: {question}"
            )},
        ]

        # Try Gemini first if available
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages, temperature=0.2)
                if raw and len(raw) > 10:
                    return raw
        except Exception:
            pass

        return await ollama.chat(messages)


# Register
registry.register(DocumentTool())
