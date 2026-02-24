"""
Bantz v3 — Context Builder

Converts Neo4j graph data into LLM-ready context strings.
Used by the Gemini finalizer for memory-enhanced responses.
"""
from __future__ import annotations

from bantz.memory.graph import graph_memory


class ContextBuilder:
    """Builds LLM context from Neo4j graph memory."""

    def for_query(self, user_input: str) -> str:
        """Extract relevant context for a user query."""
        if not graph_memory.is_available:
            return ""

        context_parts: list[str] = []

        # Extract person names (simple heuristic: capitalized words)
        import re
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', user_input)
        for word in words[:2]:
            ctx = graph_memory.build_context(person=word)
            if ctx:
                context_parts.append(ctx)

        # Check for topic keywords
        topics = re.findall(r'\b(?:project|bantz|meeting|assignment)\w*\b', user_input, re.I)
        for topic in topics[:1]:
            ctx = graph_memory.build_context(topic=topic.lower())
            if ctx:
                context_parts.append(ctx)

        if not context_parts:
            # Default: recent decisions
            ctx = graph_memory.build_context(n=5)
            if ctx:
                context_parts.append(ctx)

        return "\n\n".join(context_parts)

    def format_for_llm(self, raw_context: str) -> str:
        """Wrap context for injection into LLM prompt."""
        if not raw_context.strip():
            return ""
        return f"\n[Memory context from graph]\n{raw_context}\n[End context]\n"


context_builder = ContextBuilder()
