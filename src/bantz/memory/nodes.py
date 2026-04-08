"""
Bantz v3 — Graph Node / Relationship Schema + Entity Extraction

Defines the node labels, relationship types, and rule-based entity
extraction logic for the Neo4j knowledge graph.

Usage:
    from bantz.memory.nodes import NODE_LABELS, REL_TYPES, extract_entities
"""
from __future__ import annotations

import re
from datetime import datetime

# ── Schema constants ───────────────────────────────────────────────────────

NODE_LABELS = (
    "Person", "Topic", "Decision", "Task", "Event",
    "Location", "Document", "Reminder", "Commitment",
    "Project", "Fact",
)

REL_TYPES = {
    "KNOWS":        ("Person", "Person"),
    "ASSIGNED_TO":  ("Task", "Person"),
    "RELATED_TO":   (None, None),          # flexible
    "DECIDED_IN":   ("Decision", "Event"),
    "WORKS_ON":     ("Person", "Topic"),
    "LOCATED_AT":   (None, "Location"),
    "REFERENCES":   (None, "Document"),
    "COMMITTED_TO": ("Person", "Task"),
    "FOLLOWS_UP":   ("Task", "Decision"),
    # #293 additions
    "ABOUT":        (None, "Topic"),
    "DECIDED_BY":   ("Decision", "Person"),
    "DEPENDS_ON":   ("Task", "Task"),
    "HAPPENED_AT":  ("Event", "Location"),
}

# Words that look like names but aren't
_SKIP_NAMES = frozenset({
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March",
    "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Today", "Tomorrow",
    "Error", "Done", "Event", "Calendar", "Gmail", "News",
    "Desktop", "Downloads", "Documents", "Home", "Bantz",
    "Linux", "Python", "English", "Turkish",
})

_SKIP_LOCATIONS = frozenset({
    *_SKIP_NAMES, "Neo", "Ollama", "Google",
})

_TOOL_TOPIC_MAP = {
    "calendar": "calendar",
    "gmail": "email",
    "news": "news",
    "weather": "weather",
    "classroom": "university",
    "web_search": "research",
    "document": "documents",
    "shell": "system",
}


# ── Entity extraction ──────────────────────────────────────────────────────

def extract_entities(
    user_msg: str,
    assistant_msg: str,
    tool_used: str | None,
    tool_data: dict | None,
) -> list[dict]:
    """
    Rule-based entity extraction from a conversation exchange.

    Returns list of dicts:
        {"label": ..., "key": ..., "props": {...}, "rels": [...]}
    """
    entities: list[dict] = []
    combined = f"{user_msg} {assistant_msg}".lower()
    tool_data = tool_data or {}

    # ── People mentioned ──
    people_patterns = [
        r"(?:from|with|to|by|about|ask|tell|call|email|meet)"
        r"\s+([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)",
    ]
    found_people: set[str] = set()
    for pat in people_patterns:
        for m in re.finditer(pat, f"{user_msg} {assistant_msg}"):
            name = m.group(1).strip()
            if name not in _SKIP_NAMES and len(name) > 2:
                found_people.add(name)

    for name in found_people:
        entities.append({
            "label": "Person",
            "key": "name",
            "props": {"name": name, "last_seen": datetime.now().isoformat()},
            "rels": [],
        })

    # Track extracted names/labels for relationship building later
    _people_names = list(found_people)

    # ── Topics from tool usage ──
    if tool_used and tool_used in _TOOL_TOPIC_MAP:
        topic_name = _TOOL_TOPIC_MAP[tool_used]
        entities.append({
            "label": "Topic",
            "key": "name",
            "props": {"name": topic_name, "last_accessed": datetime.now().isoformat()},
            "rels": [],
        })

    # ── Calendar events → Event nodes ──
    if tool_used == "calendar" and tool_data:
        events = tool_data.get("events", [])
        if isinstance(events, list):
            for ev in events[:5]:
                title = ev.get("summary") or ev.get("title", "")
                if title:
                    start = ev.get("start", "")
                    entities.append({
                        "label": "Event",
                        "key": "title",
                        "props": {"title": title, "date": start,
                                  "updated_at": datetime.now().isoformat()},
                        "rels": [],
                    })
        if "added" in assistant_msg.lower() or "created" in assistant_msg.lower():
            title_m = re.search(
                r"[\"'](.+?)[\"']|Event:\s*(.+?)(?:\n|$)", assistant_msg)
            if title_m:
                t = (title_m.group(1) or title_m.group(2) or "").strip()
                if t:
                    entities.append({
                        "label": "Event",
                        "key": "title",
                        "props": {"title": t, "updated_at": datetime.now().isoformat()},
                        "rels": [],
                    })

    # ── Gmail → potential person nodes ──
    if tool_used == "gmail" and tool_data:
        messages = tool_data.get("messages", [])
        for msg in messages[:5]:
            sender = msg.get("from", "")
            if sender:
                name_match = re.match(r"([^<]+)", sender)
                if name_match:
                    name = name_match.group(1).strip().strip('"')
                    if name and len(name) > 2 and "@" not in name:
                        entities.append({
                            "label": "Person",
                            "key": "name",
                            "props": {"name": name, "email": sender.strip(),
                                      "last_seen": datetime.now().isoformat()},
                            "rels": [],
                        })

    # ── Decisions — "let's use X", "we'll go with X" ──
    decision_patterns = [
        r"(?:let'?s|we(?:'ll)?\s+(?:go\s+with|use|pick|choose|decided?\s+to))"
        r"\s+(.+?)(?:\.|$|!)",
        r"(?:i'?ll|going to|plan to)\s+(.+?)(?:\.|$|!)",
    ]
    for pat in decision_patterns:
        m = re.search(pat, combined)
        if m:
            what = m.group(1).strip()[:100]
            if len(what) > 5:
                entities.append({
                    "label": "Decision",
                    "key": "what",
                    "props": {"what": what,
                              "date": datetime.now().isoformat(),
                              "context": user_msg[:200]},
                    "rels": [],
                })
            break

    # ── Tasks — "remind me to", "i need to", "add task" ──
    task_patterns = [
        r"(?:remind\s+me\s+to|i\s+need\s+to|todo|add\s+task|task:)\s+(.+?)(?:\.|$|!)",
        r"(?:don'?t\s+forget\s+to)\s+(.+?)(?:\.|$|!)",
    ]
    for pat in task_patterns:
        m = re.search(pat, combined)
        if m:
            desc = m.group(1).strip()[:150]
            if len(desc) > 3:
                entities.append({
                    "label": "Task",
                    "key": "description",
                    "props": {"description": desc, "status": "open",
                              "priority": "medium",
                              "created_at": datetime.now().isoformat()},
                    "rels": [],
                })
            break

    # ── Documents ──
    if tool_used == "document":
        path = tool_data.get("path", "")
        if path:
            entities.append({
                "label": "Document",
                "key": "path",
                "props": {"path": path,
                          "accessed_at": datetime.now().isoformat()},
                "rels": [],
            })

    # ── Locations ──
    _location_names: list[str] = []
    loc_patterns = [
        r"(?:in|at|from|going to|travel to)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        r"(?:\s|,|\.|\?|!|$)",
    ]
    for pat in loc_patterns:
        for m in re.finditer(pat, f"{user_msg} {assistant_msg}"):
            loc = m.group(1).strip()
            if loc not in _SKIP_LOCATIONS and len(loc) > 2:
                _location_names.append(loc)
                entities.append({
                    "label": "Location",
                    "key": "name",
                    "props": {"name": loc,
                              "last_mentioned": datetime.now().isoformat()},
                    "rels": [],
                })

    # ── Reminders (from tool data) ──
    if tool_used == "reminder" and tool_data:
        title = tool_data.get("title", "")
        if title:
            props = {
                "title": title,
                "status": "active",
                "created_at": datetime.now().isoformat(),
            }
            if tool_data.get("fire_at"):
                props["fire_at"] = tool_data["fire_at"]
                props["trigger_type"] = "time"
            if tool_data.get("trigger_place"):
                props["trigger_place"] = tool_data["trigger_place"]
                props["trigger_type"] = "location"
            entities.append({
                "label": "Reminder",
                "key": "title",
                "props": props,
                "rels": [],
            })

    # ── Commitments — "I promised", "I committed", "I guarantee" ──
    commitment_patterns = [
        r"(?:i\s+promise[d]?\s+(?:to\s+)?|i\s+commit(?:ted)?\s+to\s+|"
        r"i\s+guarantee\s+|i\s+swear\s+(?:to\s+)?|i'?ll\s+make\s+sure\s+(?:to\s+)?)"
        r"(.+?)(?:\.|$|!)",
        r"(?:promise[d]?\s+(?:you|him|her|them)\s+(?:to\s+)?)"
        r"(.+?)(?:\.|$|!)",
    ]
    for pat in commitment_patterns:
        m = re.search(pat, combined)
        if m:
            what = m.group(1).strip()[:150]
            if len(what) > 3:
                rels: list[dict] = []
                # Link commitment to people mentioned
                for pname in _people_names:
                    rels.append({
                        "type": "COMMITTED_TO",
                        "target_label": "Person",
                        "target_key": "name",
                        "target_val": pname,
                    })
                entities.append({
                    "label": "Commitment",
                    "key": "what",
                    "props": {
                        "what": what,
                        "status": "active",
                        "date": datetime.now().isoformat(),
                        "context": user_msg[:200],
                    },
                    "rels": rels,
                })
            break

    # ── Projects — "project X", "working on X", "building X" ──
    project_patterns = [
        r"(?:project|working\s+on|building|developing)\s+[\"']?([A-Za-z][\w\s\-]{2,30})[\"']?",
    ]
    for pat in project_patterns:
        m = re.search(pat, combined)
        if m:
            name = m.group(1).strip()
            if len(name) > 2:
                entities.append({
                    "label": "Project",
                    "key": "name",
                    "props": {"name": name,
                              "updated_at": datetime.now().isoformat()},
                    "rels": [],
                })
            break

    # ── Facts — "X is Y", "did you know" assertions ──
    fact_patterns = [
        r"(?:note\s+that|fact:|remember\s+that|fyi[,:]?\s+)(.+?)(?:\.|$|!)",
    ]
    for pat in fact_patterns:
        m = re.search(pat, combined)
        if m:
            text = m.group(1).strip()[:200]
            if len(text) > 5:
                entities.append({
                    "label": "Fact",
                    "key": "text",
                    "props": {"text": text,
                              "created_at": datetime.now().isoformat()},
                    "rels": [],
                })
            break

    # ── Build relationships between co-occurring entities ──
    entities = _build_relationships(entities, _people_names, _location_names, tool_used)

    return entities


def _build_relationships(
    entities: list[dict],
    people: list[str],
    locations: list[str],
    tool_used: str | None,
) -> list[dict]:
    """Enrich entities with cross-references based on co-occurrence."""

    # Collect labels of entities we extracted this turn
    any(e["label"] == "Task" for e in entities)
    has_event = any(e["label"] == "Event" for e in entities)
    has_decision = any(e["label"] == "Decision" for e in entities)
    has_document = any(e["label"] == "Document" for e in entities)
    has_topic = any(e["label"] == "Topic" for e in entities)

    for ent in entities:
        label = ent["label"]

        # Person → Task: ASSIGNED_TO
        if label == "Task" and people:
            for pname in people[:2]:
                ent["rels"].append({
                    "type": "ASSIGNED_TO",
                    "target_label": "Person",
                    "target_key": "name",
                    "target_val": pname,
                })

        # Person → Topic: WORKS_ON
        if label == "Person" and has_topic:
            topic_ent = next((e for e in entities if e["label"] == "Topic"), None)
            if topic_ent:
                ent["rels"].append({
                    "type": "WORKS_ON",
                    "target_label": "Topic",
                    "target_key": "name",
                    "target_val": topic_ent["props"]["name"],
                })

        # Decision → Event: DECIDED_IN
        if label == "Decision" and has_event:
            event_ent = next((e for e in entities if e["label"] == "Event"), None)
            if event_ent:
                ent["rels"].append({
                    "type": "DECIDED_IN",
                    "target_label": "Event",
                    "target_key": "title",
                    "target_val": event_ent["props"]["title"],
                })

        # Event/Person → Location: LOCATED_AT
        if label in ("Event", "Person") and locations:
            ent["rels"].append({
                "type": "LOCATED_AT",
                "target_label": "Location",
                "target_key": "name",
                "target_val": locations[0],
            })

        # Task/Decision → Document: REFERENCES
        if label in ("Task", "Decision") and has_document:
            doc_ent = next((e for e in entities if e["label"] == "Document"), None)
            if doc_ent:
                ent["rels"].append({
                    "type": "REFERENCES",
                    "target_label": "Document",
                    "target_key": "path",
                    "target_val": doc_ent["props"]["path"],
                })

        # Topic ↔ Topic / Event / Person: RELATED_TO
        if label == "Topic" and has_event:
            event_ent = next((e for e in entities if e["label"] == "Event"), None)
            if event_ent:
                ent["rels"].append({
                    "type": "RELATED_TO",
                    "target_label": "Event",
                    "target_key": "title",
                    "target_val": event_ent["props"]["title"],
                })

        # Person → Person: KNOWS (if multiple people mentioned together)
        if label == "Person" and len(people) > 1:
            my_name = ent["props"]["name"]
            for other in people:
                if other != my_name:
                    ent["rels"].append({
                        "type": "KNOWS",
                        "target_label": "Person",
                        "target_key": "name",
                        "target_val": other,
                    })

        # Task → Decision: FOLLOWS_UP
        if label == "Task" and has_decision:
            dec_ent = next((e for e in entities if e["label"] == "Decision"), None)
            if dec_ent:
                ent["rels"].append({
                    "type": "FOLLOWS_UP",
                    "target_label": "Decision",
                    "target_key": "what",
                    "target_val": dec_ent["props"]["what"],
                })

    return entities
