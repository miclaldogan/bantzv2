"""
Bantz MemPalace Onboarding — First-run user knowledge seeding.

When Bantz starts and no onboarding has been completed, this module
runs a natural LLM-powered conversation to learn about the user:

  - Name / how to address them
  - Profession / occupation
  - Preferred language
  - Interests / work areas
  - Tools / technologies

Instead of a rigid Q&A form, Bantz asks conversational questions and
the LLM extracts structured facts from the user's natural responses.

Answers are stored in three places:
  - L0 identity file  (~/.mempalace/identity.txt)
  - KnowledgeGraph     (entities + triples)
  - EntityRegistry     (people seed)

A flag file (~/.mempalace/.bantz_onboarding_done) prevents re-asking.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.entity_registry import EntityRegistry

log = logging.getLogger("bantz.memory.onboarding")

# ── Flag file ────────────────────────────────────────────────────────────

_FLAG_NAME = ".bantz_onboarding_done"


def _flag_path(palace_parent: str | None = None) -> Path:
    if palace_parent:
        return Path(palace_parent) / _FLAG_NAME
    return Path.home() / ".mempalace" / _FLAG_NAME


def is_onboarding_done(palace_parent: str | None = None) -> bool:
    """Check whether onboarding has already been completed."""
    return _flag_path(palace_parent).exists()


def _set_done(palace_parent: str | None = None) -> None:
    p = _flag_path(palace_parent)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(datetime.now().isoformat())


# ── Conversational questions ─────────────────────────────────────────────

CONVERSATION_STEPS: list[dict] = [
    {
        "key": "name",
        "prompt_tr": "Size nasıl hitap etmeliyim?",
        "prompt_en": "How should I address you?",
        "required": True,
    },
    {
        "key": "profession",
        "prompt_tr": "Ne iş yapıyorsunuz, nelerle uğraşıyorsunuz?",
        "prompt_en": "What do you do? What's your line of work?",
        "required": False,
    },
    {
        "key": "language",
        "prompt_tr": "Benimle hangi dilde konuşmayı tercih edersiniz?",
        "prompt_en": "Which language do you prefer to chat in?",
        "required": False,
        "default": "tr",
    },
    {
        "key": "interests",
        "prompt_tr": "Nelere ilgi duyuyorsunuz, hangi konularda çalışıyorsunuz?",
        "prompt_en": "What are you interested in? What topics do you work on?",
        "required": False,
    },
    {
        "key": "tools",
        "prompt_tr": "Günlük işlerinizde hangi araçları kullanıyorsunuz?",
        "prompt_en": "What tools or technologies do you use day-to-day?",
        "required": False,
    },
]

# ── LLM extraction ──────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You extract structured information from a user's natural-language answer.
Given the question key and the user's response, return ONLY a JSON object
with the extracted value. Be concise — extract the core fact, not the full sentence.

Rules:
- key="name": extract just the name/nickname. e.g. "İclal diyebilirsin" → "İclal"
- key="profession": extract role/title. e.g. "yazılımla uğraşıyorum" → "yazılımcı"
- key="language": extract language code. e.g. "Türkçe konuşalım" → "tr", "English" → "en"
- key="interests": extract comma-separated topics. e.g. "AI ve backend ile ilgileniyorum" → "AI, backend"
- key="tools": extract comma-separated tools. e.g. "Python ve Docker kullanıyorum" → "Python, Docker"
- If the response is empty or meaningless, return {"value": ""}

Return ONLY: {"value": "<extracted>"}
No explanation, no markdown.
"""


async def _extract_with_llm(key: str, user_response: str) -> str:
    """Use LLM to extract structured data from a natural response."""
    try:
        from bantz.llm.ollama import ollama

        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": f'key="{key}"\nresponse: {user_response}'},
        ]
        raw = await ollama.chat(messages)
        data = json.loads(raw.strip())
        return data.get("value", "").strip()
    except Exception as exc:
        log.debug("LLM extraction failed for key=%s: %s — using raw input", key, exc)
        # Graceful fallback: return the raw input as-is
        return user_response.strip()


def _extract_without_llm(key: str, user_response: str) -> str:
    """Simple heuristic extraction — no LLM needed (for sync/offline)."""
    text = user_response.strip()
    if not text:
        return ""

    if key == "name":
        # "İclal diyebilirsin" → "İclal", "call me Bob" → "Bob"
        import re
        for pat in [
            r"(?:bana|beni)\s+(\S+)", r"(\S+)\s+(?:de|diye)", r"(?:call me|i'?m)\s+(\S+)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip(".,!?")
        # If short (1-3 words), it's probably just the name
        words = text.split()
        if len(words) <= 3:
            return words[0].strip(".,!?")
        return text

    if key == "language":
        low = text.lower()
        if any(w in low for w in ("türkçe", "turkish", "tr")):
            return "tr"
        if any(w in low for w in ("ingilizce", "english", "en")):
            return "en"
        if len(text) <= 3:
            return text.lower()
        return text

    # For profession, interests, tools — return as-is
    return text


# ── Conversational Q&A ──────────────────────────────────────────────────

def _ask_conversational(*, locale: str = "tr") -> dict[str, str]:
    """Ask natural conversational questions, return answers dict."""
    answers: dict[str, str] = {}
    lang_key = "prompt_tr" if locale.startswith("tr") else "prompt_en"

    print()
    print("=" * 50)
    if locale.startswith("tr"):
        print("  🧠 Bantz — Tanışma")
        print("=" * 50)
        print("  Birkaç soru soracağım, sizi daha iyi tanıyayım.")
    else:
        print("  🧠 Bantz — Getting to Know You")
        print("=" * 50)
        print("  Let me ask a few questions so I can get to know you.")
    print()

    for step in CONVERSATION_STEPS:
        prompt_text = step.get(lang_key, step["prompt_en"])
        default = step.get("default", "")
        suffix = f" [{default}]" if default else ""

        while True:
            raw = input(f"  {prompt_text}{suffix} ").strip()
            if not raw and default:
                raw = default
            if not raw and step.get("required"):
                hint = "Bunu bilmem lazım :)" if locale.startswith("tr") else "I need to know this one :)"
                print(f"    → {hint}")
                continue
            break

        if raw:
            # Use heuristic extraction for sync context
            extracted = _extract_without_llm(step["key"], raw)
            answers[step["key"]] = extracted or raw

    return answers


async def _ask_conversational_async(*, locale: str = "tr") -> dict[str, str]:
    """Ask natural questions and use LLM to extract structured answers."""
    answers: dict[str, str] = {}
    lang_key = "prompt_tr" if locale.startswith("tr") else "prompt_en"

    print()
    print("=" * 50)
    if locale.startswith("tr"):
        print("  🧠 Bantz — Tanışma")
        print("=" * 50)
        print("  Birkaç soru soracağım, sizi daha iyi tanıyayım.")
    else:
        print("  🧠 Bantz — Getting to Know You")
        print("=" * 50)
        print("  Let me ask a few questions so I can get to know you.")
    print()

    for step in CONVERSATION_STEPS:
        prompt_text = step.get(lang_key, step["prompt_en"])
        default = step.get("default", "")
        suffix = f" [{default}]" if default else ""

        while True:
            raw = input(f"  {prompt_text}{suffix} ").strip()
            if not raw and default:
                raw = default
            if not raw and step.get("required"):
                hint = "Bunu bilmem lazım :)" if locale.startswith("tr") else "I need to know this one :)"
                print(f"    → {hint}")
                continue
            break

        if raw:
            extracted = await _extract_with_llm(step["key"], raw)
            answers[step["key"]] = extracted or raw

    return answers


def seed_identity(
    answers: dict[str, str],
    identity_path: str,
) -> None:
    """Write / update the L0 identity file from onboarding answers."""
    name = answers.get("name", "User")
    profession = answers.get("profession", "")
    language = answers.get("language", "tr")
    interests = answers.get("interests", "")
    tools = answers.get("tools", "")

    lines = [
        f"## L0 — IDENTITY",
        f"Name: {name}",
    ]
    if profession:
        lines.append(f"Profession: {profession}")
    lines.append(f"Language: {language}")
    if interests:
        lines.append(f"Interests: {interests}")
    if tools:
        lines.append(f"Tools: {tools}")

    path = Path(identity_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Identity file written: %s", path)


def seed_knowledge_graph(
    answers: dict[str, str],
    kg: "KnowledgeGraph",
) -> int:
    """Seed the KG with onboarding facts. Returns the number of triples added."""
    today = datetime.now().strftime("%Y-%m-%d")
    src = "bantz_onboarding"
    count = 0

    name = answers.get("name")
    if name:
        kg.add_entity(name, entity_type="person")
        kg.add_triple("user", "name_is", name, valid_from=today, source_file=src)
        count += 1

    profession = answers.get("profession")
    if profession:
        kg.add_triple("user", "works_as", profession, valid_from=today, source_file=src)
        count += 1

    language = answers.get("language")
    if language:
        kg.add_triple("user", "speaks", language, valid_from=today, source_file=src)
        count += 1

    interests = answers.get("interests", "")
    for item in (i.strip() for i in interests.split(",") if i.strip()):
        kg.add_triple("user", "interested_in", item, valid_from=today, source_file=src)
        count += 1

    tools = answers.get("tools", "")
    for item in (i.strip() for i in tools.split(",") if i.strip()):
        kg.add_triple("user", "uses_tool", item, valid_from=today, source_file=src)
        count += 1

    return count


def seed_entity_registry(
    answers: dict[str, str],
    registry: "EntityRegistry",
) -> None:
    """Seed the EntityRegistry with the user's name."""
    name = answers.get("name")
    if not name:
        return

    registry.seed(
        mode="personal",
        people=[{
            "name": name,
            "relationship": "self",
            "context": "personal",
        }],
        projects=[],
    )


def run_onboarding(
    *,
    identity_path: str,
    kg: "KnowledgeGraph",
    registry: "EntityRegistry",
    palace_parent: str | None = None,
    locale: str = "tr",
) -> dict[str, str]:
    """Full interactive onboarding flow (sync — heuristic extraction).

    Returns the answers dict (empty if user cancels).
    """
    answers = _ask_conversational(locale=locale)
    if not answers.get("name"):
        log.info("Onboarding cancelled — no name provided")
        return {}

    seed_identity(answers, identity_path)
    triple_count = seed_knowledge_graph(answers, kg)
    seed_entity_registry(answers, registry)
    _set_done(palace_parent)

    log.info(
        "Onboarding complete: %d triples, identity written, flag set",
        triple_count,
    )
    return answers


async def run_onboarding_async(
    *,
    identity_path: str,
    kg: "KnowledgeGraph",
    registry: "EntityRegistry",
    palace_parent: str | None = None,
    locale: str = "tr",
) -> dict[str, str]:
    """Full interactive onboarding with LLM-powered answer extraction.

    Returns the answers dict (empty if user cancels).
    """
    answers = await _ask_conversational_async(locale=locale)
    if not answers.get("name"):
        log.info("Onboarding cancelled — no name provided")
        return {}

    seed_identity(answers, identity_path)
    triple_count = seed_knowledge_graph(answers, kg)
    seed_entity_registry(answers, registry)
    _set_done(palace_parent)

    log.info(
        "Onboarding complete (LLM): %d triples, identity written, flag set",
        triple_count,
    )
    return answers


def run_onboarding_noninteractive(
    *,
    answers: dict[str, str],
    identity_path: str,
    kg: "KnowledgeGraph",
    registry: "EntityRegistry",
    palace_parent: str | None = None,
) -> int:
    """Programmatic onboarding (for tests / CI). Returns triple count."""
    if not answers.get("name"):
        return 0

    seed_identity(answers, identity_path)
    count = seed_knowledge_graph(answers, kg)
    seed_entity_registry(answers, registry)
    _set_done(palace_parent)
    return count
