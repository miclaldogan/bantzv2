"""
Bantz v3 — Document Tool
Reads, summarizes, and answers questions about documents.
Supports PDF (text + tables), DOCX, TXT, MD, CSV, and more.

Actions:
  read       — raw text extraction (with optional page range)
  summarize  — LLM-powered summary (chunked for large docs)
  ask        — Q&A against document content
  tables     — extract tables from PDF (returns structured data)
  info       — file metadata (pages, size, word count)

Features:
  - Smart chunking: splits large docs into overlapping chunks for LLM
  - Table extraction: pdfplumber tables → structured rows
  - Page-specific reads: read pages 3-5 of a PDF
  - CSV awareness: structured column/row reading
"""
from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
from typing import Any, Optional

from bantz.tools import BaseTool, ToolResult, registry

# Max chars to send to LLM per chunk
_CHUNK_SIZE = 3500
_CHUNK_OVERLAP = 300


class DocumentTool(BaseTool):
    name = "document"
    description = (
        "Reads and summarizes documents: PDF, DOCX, TXT, MD, CSV. "
        "Extracts tables from PDFs, answers questions about content. "
        "Use for: summarize file, read document, what's in this file, "
        "explain PDF, extract table, page 3 of report."
    )
    risk_level = "safe"

    async def execute(
        self,
        path: str = "",
        action: str = "summarize",
        # read | summarize | ask | tables | info
        question: str = "",       # for ask action
        page: int = 0,            # specific page (1-indexed, 0=all)
        page_start: int = 0,      # page range start
        page_end: int = 0,        # page range end
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

        suffix = resolved.suffix.lower()

        # ── Info (no text extraction needed) ──────────────────────────────
        if action == "info":
            return await self._file_info(resolved, suffix)

        # ── Tables (PDF only) ────────────────────────────────────────────
        if action == "tables":
            if suffix != ".pdf":
                return ToolResult(
                    success=False, output="",
                    error="Table extraction is only supported for PDF files.",
                )
            return await self._extract_tables(resolved, page)

        # ── Extract text ──────────────────────────────────────────────────
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_text, resolved, suffix, page, page_start, page_end,
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

        # ── Action dispatch ───────────────────────────────────────────────

        if action == "read":
            truncated = text[:_CHUNK_SIZE * 2]
            trunc_note = (
                f" (showing first {len(truncated)} of {char_count} chars)"
                if char_count > len(truncated) else ""
            )
            return ToolResult(
                success=True,
                output=f"{resolved.name}{trunc_note}\n\n{truncated}",
                data={"path": str(resolved), "chars": char_count},
            )

        if action == "ask" and question:
            answer = await self._ask_chunked(text, question, resolved.name)
            return ToolResult(
                success=True,
                output=answer,
                data={"path": str(resolved), "chars": char_count, "action": "ask"},
            )

        # Default: summarize
        summary = await self._summarize_chunked(text, resolved.name)
        return ToolResult(
            success=True,
            output=summary,
            data={"path": str(resolved), "chars": char_count, "action": "summarize"},
        )

    # ── Text extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_text(
        path: Path, suffix: str,
        page: int = 0, page_start: int = 0, page_end: int = 0,
    ) -> str:
        if suffix == ".pdf":
            return DocumentTool._read_pdf(path, page, page_start, page_end)
        elif suffix == ".docx":
            return DocumentTool._read_docx(path)
        elif suffix == ".csv":
            return DocumentTool._read_csv(path)
        elif suffix in (
            ".txt", ".md", ".markdown", ".rst", ".log",
            ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
            ".py", ".js", ".ts", ".html", ".xml",
        ):
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_pdf(
        path: Path,
        page: int = 0, page_start: int = 0, page_end: int = 0,
    ) -> str:
        """Extract text from PDF with optional page range."""
        # Determine page range (0-indexed internally)
        p_start = 0
        p_end = None  # all pages

        if page > 0:
            p_start = page - 1
            p_end = page
        elif page_start > 0:
            p_start = page_start - 1
            p_end = page_end if page_end > 0 else None

        # Try PyMuPDF (fitz)
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            pages = []
            for i, pg in enumerate(doc):
                if i < p_start:
                    continue
                if p_end is not None and i >= p_end:
                    break
                pages.append(pg.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                pages = []
                for i, pg in enumerate(pdf.pages):
                    if i < p_start:
                        continue
                    if p_end is not None and i >= p_end:
                        break
                    pages.append(pg.extract_text() or "")
            return "\n\n".join(pages)
        except ImportError:
            pass

        raise ImportError(
            "No PDF library found. Install one: pip install pymupdf  OR  pip install pdfplumber"
        )

    @staticmethod
    def _read_docx(path: Path) -> str:
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError(
                "python-docx not found. Install: pip install python-docx"
            )

    @staticmethod
    def _read_csv(path: Path) -> str:
        """Read CSV with structure awareness."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            return ""

        header = rows[0]
        lines = [" | ".join(header), "-" * (len(" | ".join(header)))]
        for row in rows[1:]:
            lines.append(" | ".join(row))

        return "\n".join(lines)

    # ── Table extraction ──────────────────────────────────────────────────

    async def _extract_tables(self, path: Path, page: int = 0) -> ToolResult:
        tables = await asyncio.get_event_loop().run_in_executor(
            None, self._extract_tables_sync, path, page
        )
        if not tables:
            return ToolResult(
                success=True,
                output=f"No tables found in {path.name}.",
            )

        output_parts = []
        for i, table in enumerate(tables, 1):
            rows = table.get("rows", [])
            if not rows:
                continue
            header = "Table {i} (page {pg}, {n} rows):".format(
                i=i, pg=table.get("page", "?"), n=len(rows),
            )
            lines = []
            for row in rows[:20]:  # limit preview
                lines.append(" | ".join(str(c) if c else "" for c in row))
            if len(rows) > 20:
                lines.append(f"  ... ({len(rows) - 20} more rows)")
            output_parts.append(header + "\n" + "\n".join(lines))

        return ToolResult(
            success=True,
            output="\n\n".join(output_parts),
            data={"table_count": len(tables), "tables": tables},
        )

    @staticmethod
    def _extract_tables_sync(path: Path, page: int = 0) -> list[dict]:
        """Extract tables from PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            return []

        tables: list[dict] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, pg in enumerate(pdf.pages):
                if page > 0 and i != page - 1:
                    continue
                for tbl in pg.extract_tables():
                    if tbl:
                        tables.append({
                            "page": i + 1,
                            "rows": tbl,
                        })
        return tables

    # ── File info ─────────────────────────────────────────────────────────

    async def _file_info(self, path: Path, suffix: str) -> ToolResult:
        info = await asyncio.get_event_loop().run_in_executor(
            None, self._file_info_sync, path, suffix
        )
        lines = [f"File: {path.name}"]
        lines.append(f"  Size: {info['size_kb']:.1f} KB")
        lines.append(f"  Type: {info['type']}")
        if info.get("pages"):
            lines.append(f"  Pages: {info['pages']}")
        if info.get("word_count"):
            lines.append(f"  Words: ~{info['word_count']}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=info,
        )

    @staticmethod
    def _file_info_sync(path: Path, suffix: str) -> dict:
        info: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "size_kb": path.stat().st_size / 1024,
            "type": suffix.lstrip(".").upper(),
        }

        # Page count for PDF
        if suffix == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                info["pages"] = len(doc)
                doc.close()
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as pdf:
                        info["pages"] = len(pdf.pages)
                except ImportError:
                    pass

        # Word count (approximate)
        try:
            text = DocumentTool._extract_text(path, suffix)
            info["word_count"] = len(text.split())
        except Exception:
            pass

        return info

    # ── Chunking ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text into overlapping chunks for LLM processing."""
        if len(text) <= _CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - _CHUNK_OVERLAP
        return chunks

    # ── LLM — summarize with chunking ─────────────────────────────────────

    async def _summarize_chunked(self, text: str, filename: str) -> str:
        """Summarize with chunk-and-merge for long documents."""
        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            return await self._summarize_llm(chunks[0], filename)

        # Multi-chunk: summarize each, then merge
        chunk_summaries = []
        for i, chunk in enumerate(chunks[:6]):  # max 6 chunks
            s = await self._summarize_llm(
                chunk, f"{filename} (part {i+1}/{min(len(chunks), 6)})"
            )
            chunk_summaries.append(s)

        merged = "\n\n".join(chunk_summaries)
        if len(chunks) > 6:
            merged += f"\n\n(Document has {len(chunks)} sections — showing first 6)"

        # Final merge summary
        final = await self._merge_summaries(merged, filename)
        return final

    async def _ask_chunked(self, text: str, question: str, filename: str) -> str:
        """Answer question using the most relevant chunk(s)."""
        chunks = self._chunk_text(text)

        if len(chunks) == 1:
            return await self._ask_llm(chunks[0], question, filename)

        # Find most relevant chunks by keyword overlap
        q_words = set(question.lower().split())
        scored = []
        for i, chunk in enumerate(chunks):
            c_words = set(chunk.lower().split())
            overlap = len(q_words & c_words)
            scored.append((overlap, i, chunk))
        scored.sort(reverse=True)

        # Use top 2 most relevant chunks
        best_chunks = [c for _, _, c in scored[:2]]
        combined = "\n\n---\n\n".join(best_chunks)
        return await self._ask_llm(combined[:_CHUNK_SIZE * 2], question, filename)

    # ── LLM interaction ───────────────────────────────────────────────────

    @staticmethod
    async def _llm_call(messages: list[dict]) -> str:
        """Call LLM — tries Gemini first, falls back to Ollama."""
        try:
            from bantz.llm.gemini import gemini
            if gemini.is_enabled():
                raw = await gemini.chat(messages, temperature=0.2)
                if raw and len(raw) > 10:
                    return raw
        except Exception:
            pass

        from bantz.llm.ollama import ollama
        return await ollama.chat(messages)

    @staticmethod
    async def _summarize_llm(text: str, filename: str) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a document summarizer. Summarize the following document text "
                "clearly and concisely in 3-5 bullet points. Include key facts, numbers, "
                "and conclusions. English only, plain text."
            )},
            {"role": "user", "content": f"Document: {filename}\n\n{text}"},
        ]
        raw = await DocumentTool._llm_call(messages)
        return f"Summary of {filename}:\n\n{raw}"

    @staticmethod
    async def _ask_llm(text: str, question: str, filename: str) -> str:
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
        return await DocumentTool._llm_call(messages)

    @staticmethod
    async def _merge_summaries(summaries: str, filename: str) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a document summarizer. The following are summaries of different "
                "sections of a document. Merge them into one coherent 5-7 bullet-point "
                "summary. Remove duplicates. English only, plain text."
            )},
            {"role": "user", "content": f"Document: {filename}\n\n{summaries}"},
        ]
        raw = await DocumentTool._llm_call(messages)
        return f"Summary of {filename}:\n\n{raw}"


# Register
registry.register(DocumentTool())
