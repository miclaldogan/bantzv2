"""
Bantz v3 — CoT Intent Parser

Uses Qwen2.5 7B Instruct via Ollama to perform Chain of Thought routing.

Step 1: What does the user want?
Step 2: Which tool(s) are needed?
Step 3: What parameters?
Step 4: Execute.

This replaces the simple JSON-only router with a reasoning-first approach.
"""
from __future__ import annotations

import json
import re

from bantz.llm.ollama import ollama


INTENT_SYSTEM = """\
You are an intent classifier for Bantz, an operations director AI.
Think step by step, then return a JSON routing decision.

Available tools: {tool_list}

THINK THROUGH:
1. What is the user asking for?
2. Which tool handles this? (be exact)
3. What parameters does the tool need?
4. What is the risk level? (safe / moderate / destructive)

RESPOND with this exact format:
Thought: [your reasoning in one sentence]
Action: [JSON routing decision]

JSON format:
{{"route": "tool", "tool_name": "<name>", "tool_args": {{...}}, "risk_level": "safe|moderate|destructive"}}
or
{{"route": "chat", "tool_name": null, "tool_args": {{}}, "risk_level": "safe"}}

Rules:
- shell: bash commands, file listing, system info
- system: CPU/RAM/disk metrics
- weather: weather, temperature, forecast
- news: headlines, news, hacker news
- web_search: internet search, look up online
- gmail: email, inbox, compose, send
- calendar: events, meetings, schedule management
- classroom: assignments, homework, deadlines
- filesystem: read/write files
- chat: ONLY if nothing else applies
- NEVER refuse system queries\
"""


async def parse_intent(
    user_input: str,
    tool_schemas: list[dict],
) -> dict | None:
    """
    CoT intent parsing: think → route.
    Returns routing dict or None on failure.
    """
    tool_list = ", ".join(t["name"] for t in tool_schemas)

    try:
        raw = await ollama.chat([
            {"role": "system", "content": INTENT_SYSTEM.format(tool_list=tool_list)},
            {"role": "user", "content": user_input},
        ])

        # Extract JSON from "Action: {...}" pattern
        action_match = re.search(r"Action:\s*(\{.*\})", raw, re.DOTALL)
        if action_match:
            return json.loads(action_match.group(1))

        # Fallback: try extracting any JSON object
        json_match = re.search(r"\{[^{}]*\"route\"[^{}]*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

    except Exception:
        pass

    return None
