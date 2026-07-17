"""
Bantz v2 — Chain-of-Thought Intent Parser (#78, #273)

Replaces the one-shot router with structured multi-step reasoning:

  Step 1 [Intent]:  What does the user want?
  Step 2 [Tool]:    Which tool handles this?
  Step 3 [Params]:  Extract parameters from the input

The reasoning chain is logged for debugging and returned alongside the
routing decision so downstream code can inspect *why* a route was chosen.

#273: cot_route now uses ``ollama.chat_stream()`` to stream the LLM's
``<thinking>`` process in real-time to the TUI via EventBus events.

Usage:
    from bantz.core.intent import cot_route
    plan = await cot_route(en_input, registry.all_schemas())
"""
from __future__ import annotations

import json
import logging
import re

from bantz.llm.ollama import ollama
from bantz.core.event_bus import bus

log = logging.getLogger("bantz.intent")

# ── CoT system prompt ─────────────────────────────────────────────────────────

# Compact routing hints — one line per tool keeps the prompt small for 8b models.
# Full descriptions live on the tool classes; these are routing-only summaries.
_ROUTING_HINTS: dict[str, str] = {
    "web_search": "Search the internet for SPECIFIC facts, prices, named entities, or the current status of a specific thing. 'search X', 'find X', 'look up X', 'ara' (search), 'bul' (find) → web_search. NOT for general news/headlines/'what happened today' — those go to the news tool.",
    "web_research": "Deep, multi-source research that produces a long structured report (slow, minutes). 'research X', 'deep dive into Y', 'investigate Z', 'araştır' (research), 'write me a report on X' → web_research. Use web_search for quick lookups, NOT this.",
    "web_news": "Live web news/headlines for a topic via bantz-web. 'news about X', 'latest on Y', 'what happened with Z', 'haberler' (news), 'gündem' (agenda/current affairs) → web_news.",
    "calendar": "Google Calendar: view, create, update, delete events and meetings. action=create for 'add/schedule/create event/dinner/meeting'. action=today for 'what's on my calendar'. action=delete for 'remove/cancel event'.",
    "gmail": "Gmail: read inbox, compose, send, search, reply, forward emails. Use for ANY mention of email/mail/inbox/mails. action=summary for 'check my mail'. NEVER answer email questions from memory.",
    "weather": "Current weather AND forecast — handles 'tomorrow', 'this week', 'will it rain', forecast queries. Single tool call, no planning needed. city param optional (auto-detects location).",
    "reminder": "Create, list, cancel, snooze reminders and timers. action=add for new reminders. action=cancel with id=N for 'delete/cancel/remove reminder #N'. action=list for 'show my reminders'.",
    "shell": "Run a bash command (ls, df, top, apt, etc.)",
    "system": (
        "Live system metrics: CPU, RAM, disk usage, uptime. "
        "Raw Turkish that may pass through translation, all meaning a system-metric query: "
        "'cpu kullanımı' (cpu usage), 'ram kullanımı' (ram usage), 'bellek kullanımı' (memory usage), "
        "'disk kullanımı' (disk usage), 'sistem durumu' (system status), 'işlemci' (processor). "
        "Translation artifact: 'what is the use of cpu/ram/disk' = system metric query, NOT web_search. "
        "NOT for windows, apps, or UI elements — 'what windows are open', 'list interactive elements' go to accessibility/desktop."
    ),
    "lantern": (
        "Desktop ambiance control (Lantern). Turkish/English triggers: "
        "'odak modu' (focus mode) → action=focus; 'müzik ışığı/glow' → action=glow "
        "(mode=waves|pulse|hybrid via action=glow_mode); 'albüm teması' → action=album_theme; "
        "'masaüstü sesleri' → action=sounds; 'gece modu' (night mode) → action=night; "
        "'ortam/ambiyans durumu', 'batarya sağlığı', 'gpu durumu' → action=status. "
        "state=on|off|toggle. NOT for system CPU/RAM metrics (that is 'system')."
    ),
    "filesystem": "Read, write, list files and folders under home directory",
    "browser_control": "Launch apps AND control the web browser ONLY (clicking links, filling web forms, navigating URLs, web page elements). Actions: open, navigate, find_and_click, type_in_element, hotkey, type, scroll, screenshot, wait_for_load. Web services (YouTube, Spotify, Netflix…) = action=open. NOT for: clicking desktop buttons/dialogs (visual_click), native app windows (desktop), bare keyboard/mouse input (input_control), listing windows (accessibility). IMPORTANT: 'play/search/find X on YouTube/YT Music' needs MULTIPLE steps (search → wait → click) → route=planner.",
    "visual_click": "Click any visible UI element on screen by describing it ('click the submit button', 'click OK in the dialog'). Works on desktop and native app windows — first choice for clicking things that are NOT inside a web page.",
    "screenshot": "Capture screen image (action=capture) OR analyze what's visible on screen (action=analyze). 'what do you see/what's on my screen' = action=analyze. 'take a screenshot/ss al' = action=capture.",
    "news": "News headlines and digests: 'what's in the news', 'today's headlines', 'tech news', 'what happened today'. ALWAYS news for headline browsing, never web_search.",
    "document": "Read and summarize documents: PDF, DOCX, TXT, MD, CSV",
    "read_url": "READ the full text content of a specific URL: 'read <url>', 'fetch the text of <url>'. NOT browser_control (that is for interacting with pages, not reading them).",
    "contacts": "Look up a person's phone number or email address in Google contacts. \"X's phone number\", 'find Y's email', 'do I have a contact named Z' → contacts, NOT chat.",
    "classroom": "Google Classroom: courses, assignments, announcements",
    "summarizer": "Transform text the user provides INLINE: 'summarize this text: …', 'rewrite this more formally: …', 'make this shorter: …', TL;DR. The text is already in the message — single tool call, NOT planner, NOT chat.",
    "input_control": "Low-level mouse/keyboard: press keys/shortcuts ('press ctrl+shift+t'), move the mouse to exact coordinates, type text into the active window. NOT browser_control.",
    "accessibility": "Inspect running apps via the OS accessibility tree (read-only): 'what windows are open', 'list running applications', 'inspect the UI tree of app X'.",
    "gui_action": "Interact with a specific labeled UI element in a desktop app",
    "computer_use": "Autonomous multi-step desktop automation using screen vision",
    "browser": "Advanced web page parsing: HTML, CSS selectors, image extraction",
    "feed": "Fetch and parse RSS/Atom feeds. action=fetch url=<feed_url> for a direct URL, action=category category=<name> for a group (tech, world, tr_news, science, gaming), action=list to show available categories.",
    "delegate_task": "Delegate a complex sub-task to a specialist agent. Roles: researcher (web search & synthesis), developer (code, shell, files), reviewer (validation, quality check). Use when a task needs focused multi-step expertise.",
    "run_workflow": "Execute a predefined YAML workflow by name, list available workflows, or create a new one. action=run name=<name> inputs={...} to execute, action=list to show available workflows, action=create name=<name> yaml_content=<yaml>. Use for repeated/deterministic multi-step pipelines.",
    "image": "Download, cache, and render images. action=render url=<image_url> for ANSI terminal art via chafa, action=download to cache only, action=raw for Telegram send_photo bytes.",
    "gui": "Desktop GUI automation via pyautogui + xdotool. action=click x=<int> y=<int>, action=click_image template=<path>, action=type text=<str>, action=focus_window title=<pattern> wm_class=<class>, action=screenshot region=<x,y,w,h>, action=scroll x=<int> y=<int> clicks=<int>, action=action_log to inspect recorded actions.",
    "screen_query": "Answer questions about the CURRENT screen or click an element the user describes. Triggers (Turkish forms may pass through translation): 'what do you see' / 'ekranda ne var' (what's on the screen) = describe; 'ne yazıyor' (what does it say) = read; 'click the blue button' / 'şu ikona tıkla' (click that icon) = click. query=<user request>.",
    "vision_execute": "Long multi-step desktop task guided by screen vision (screenshot → decide → act each step): 'play X on the music app', 'find Y in app Z and do W'. goal=<task>. For music: intent=play_music artist=<name> pulls favorites from profile.",
}


def _build_compact_schemas(tool_schemas: list[dict]) -> str:
    """One-line-per-tool schema string for the routing prompt."""
    lines = []
    for t in tool_schemas:
        name = t["name"]
        hint = _ROUTING_HINTS.get(name, t["description"][:80])
        lines.append(f"  {name}: {hint}")
    return "\n".join(lines)


COT_SYSTEM = """\
You are Bantz's intent router. Pick the best tool for the user's request.

TOOLS:
{tool_schemas}

<thinking>
1. What does the user want? (one sentence)
2. Which tool handles this? Pick the BEST match.
3. Extract parameters from the user's words.
4. Needs 2+ different tools in sequence? → route="planner".
</thinking>

RULES:
- CASUAL GREETINGS (hey, hi, hello, hey bud, yo, sup, what's up, howdy, good morning, good evening, how are you, how's it going) → ALWAYS route="chat". NEVER run_workflow, NEVER briefing, NEVER any tool. These are conversational openers, not requests.
- ALWAYS pick a tool when the user wants something DONE. Only route to "chat" for greetings, casual chitchat, or pure opinions.
- Factual questions about SPECIFIC current things (a price, a named entity's status, a schedule) → web_search.
- "search for X", "find X online", "look up X", "ara", "bul" → web_search (quick lookup).
- "research X", "i need research on X", "deep dive into X", "look into X (in depth)", "investigate X", "araştır", "write a report on X" → web_research (slow, in-depth report). NOT web_search, NOT planner.
- "news about X", "latest on X", "what happened with X", "haberler", "gündem" → web_news.
- "what's in the news", "today's headlines", "tech news", "what happened today" → news tool. ALWAYS news for headline/digest browsing — NEVER web_search, NEVER planner.
- "summarize / rewrite / shorten / TL;DR this text: …" with the text pasted in the message → summarizer. The text is already provided: SINGLE tool call, NOT planner, NOT chat.
- "X's phone number", "find Y's email address", "do I have a contact named Z" → contacts tool, NOT chat.
- "reply to the email from X", "respond to X's email" → gmail action=reply.
- "delegate this to the X agent", "have the X agent do Y", "ask the X agent to Z" → delegate_task role=X. This is ONE tool call even though it sounds multi-step — NEVER planner.
- "what do you know about X", "tell me what you know about X", "what can you tell me about X" → route="chat". Answer from training knowledge — do NOT run a web search.
- "who is X" → route="chat". Answer from your training knowledge. Only use web_search if the user explicitly says "search for" or "look up".
- "do you remember X", "do you know me", "have I told you", "who am I", "do you know who I am" → route="chat". Use conversation context, never a tool.
- "check my mail/email/inbox" → gmail action=summary. NEVER hallucinate email content.
- "add/create/schedule event/dinner/meeting at TIME" → calendar action=create with title and time. NOT action=today.
- "delete/cancel/remove reminder #N" or "delete the no N reminder" → reminder action=cancel id=N. The tool name is "reminder" NOT "cancel_reminder".
- "remind me in X minutes/a minute", "set a reminder", "set an alarm", "remind me at X", "remind me to X", "in X minutes remind me", "X minutes from now remind" → reminder action=add. Parse time carefully: "a minute" = 1 minute, NOT 1 hour. NEVER route reminder requests to planner.
- "weather in X tomorrow/forecast/this week/will it rain" → weather tool directly. The weather tool already returns a 3-day forecast — NEVER route weather queries to planner.
- "just open YouTube/Spotify/Netflix" (no specific content) → browser_control action=open app=firefox url=https://youtube.com
- "open Gemini/ChatGPT/Claude" or any WEB APP → browser_control action=navigate url=<correct URL>. Known web apps: gemini=https://gemini.google.com, chatgpt=https://chatgpt.com, claude=https://claude.ai, perplexity=https://perplexity.ai, github=https://github.com, reddit=https://reddit.com
- "play/search/find/watch X on YouTube" or "listen to X on YT Music" → route="planner" (needs search + wait + click steps).
- "play X" / "listen to X" (music intent, no site specified) → route="planner" (open YT Music + search + click).
- ANY music-listening wish without "play" — "i want to listen (to) X", "put on some X", "can we hear X" → route="planner" too. NEVER web_search and NEVER news: music requests are ACTIONS to perform, not information lookups.
- "research X and write a report", "research X and give a detailed summary", "deep dive into X and write it up" → web_research. The web_research tool ALREADY produces the full written report itself — do NOT send report/summary research to the planner.
- route="agent" ONLY for tasks that need a specialist working autonomously with verification: "thoroughly research X and cite/verify sources", "fact-check X across multiple sources", "have an agent handle X", "triage my inbox". Set agent_role and agent_task=<the full task>. Role guide: finding/verifying information ONLINE (web research, news, fact-checks) → agent_role="web"; reviewing code/text the user PROVIDES → "reviewer"; writing/running code or files → "developer". A bare "research X" / "araştır" stays web_research; a simple question stays chat. When unsure, do NOT pick agent.
- Creating events/meetings/dinners → calendar. Reminders/timers → reminder.
- Literal bash commands (ls, df, top) → shell.
- Multi-step with "then" / "and" / "after that" → route="planner".
- Route to planner ONLY when research must be COMBINED with a different action the report tool cannot do itself — e.g. "research X THEN email it to me", "research X and add it to my calendar", "research X then save it to ~/notes.md". A bare "research X" / "araştır" / "write me a report on X" is ALWAYS web_research, never planner.
- "delegate this to researcher/developer/reviewer" or "have the researcher look into X" → delegate_task with the specified role.
- "run workflow X", "execute my morning briefing", "list workflows" → run_workflow. If the user mentions a known workflow name, route to run_workflow action=run.
- Do NOT hallucinate data — always route to the real tool.
- browser_control action names are EXACT: find_and_click (not 'click'), navigate (not 'navigate_to'), type_in_element (not 'type_in').
- Tool name must be EXACT registry name in snake_case. Never invent tool names like "cancel_reminder" or "delete_event". Use the base tool with the right action param.
- Clicking a UI element → visual_click with "target" param (e.g. visual_click: click a button or link). NOT shell, NOT browser_control.
- "click the X button / the dialog / an icon" on the desktop or in a native app window → visual_click (or desktop), NOT browser_control. browser_control clicks ONLY inside web pages.
- "what windows are open", "what apps are running", "list running applications" → accessibility, NOT browser_control.
- "press <keys>", "keyboard shortcut", "move the mouse to X,Y", "type ... into the active window" → input_control, NOT browser_control.
- Deictic screen requests, including raw Turkish that passed through translation — "şu ikona tıkla" (click that icon), "sağdaki butona bas" (press the button on the right), "bu ne yazıyor" (what does this say), "ekrana bak ve söyle" (look at the screen and tell me) → screen_query with query=<the request>. Plain English "click the X button" stays visual_click; "what do you see on my screen" stays screenshot action=analyze.
- A goal that requires WATCHING the screen across many steps inside desktop apps ("find Y inside app Z and change W") → vision_execute goal=<task>. Play/music requests still go to route="planner" (the planner uses vision_execute).
- Create folder + file in one step → filesystem action=create_folder_and_file with folder_path and file_name params.
- Entity lookups ("what is the capital of X", "who founded X company") → web_search only when the answer is time-sensitive or obscure.

EMOTIONAL AND PERSONAL STATEMENT RULES — HIGHEST PRIORITY:
- Any message starting with "I feel", "I'm feeling", "I am feeling" → ALWAYS route="chat". No exceptions.
- Messages expressing stress, overwhelm, exhaustion, frustration, or sadness → ALWAYS route="chat". NEVER route to planner, calendar, gmail, or any task tool.
- Venting and personal statements ("I have so much to do", "everything is piling up", "I'm exhausted", "I can't handle this") → route="chat". The user is sharing, not issuing a command.
- A sentence that lists multiple topics without an explicit action verb (do, check, open, set, find, search, send) is a personal statement → route="chat".

ANTI-FALSE-POSITIVE RULES (when in doubt, route to chat):
- Never guess a tool. Only trigger a tool when the intent is unambiguous and explicit.
- Read the full sentence meaning, not individual keywords. "I don't stand for this" does NOT mean "stand" = music/player.
- Slang and idioms → always route="chat". e.g. "you got me wrong", "that's fire", "I'm dead", "let's bounce".
- Emotional or corrective statements (frustration, clarification, pushback) → route="chat".
- When in doubt between tool and chat, always choose route="chat".

OUTPUT — single JSON, no markdown:
{{"route": "tool|planner|chat|agent", "tool_name": "exact_name", "tool_args": {{}}, "agent_role": "web|developer|reviewer (route=agent only)", "agent_task": "full task (route=agent only)", "risk_level": "safe|moderate|destructive", "confidence": 0.0-1.0, "reasoning": "one sentence"}}\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)

# Matches a single <thinking> open tag (with optional whitespace / nesting)
_THINKING_OPEN = re.compile(r"<thinking\s*>", re.IGNORECASE)
# Matches a single </thinking> close tag
_THINKING_CLOSE = re.compile(r"</thinking\s*>", re.IGNORECASE)
# Detects start of JSON output — our CoT always returns {"route": ...}
# When this pattern appears in the buffer, the thinking phase is done.
_JSON_MARKER = re.compile(r'\{\s*"route"\s*:', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Aggressively remove ``<thinking>…</thinking>`` internal monologues.

    Applied at the **earliest** point where raw LLM output is received so
    that downstream JSON parsers never choke on leaked reasoning tags.
    Handles nested/multiline blocks and trailing whitespace (#214).
    Also strips orphaned/unclosed ``<thinking>`` blocks (#273).
    """
    result = _THINKING_RE.sub("", text)
    # Handle unclosed tags — remove from <thinking> to end of string
    result = re.sub(r"<thinking\s*>.*", "", result, flags=re.DOTALL | re.IGNORECASE)
    return result


_REFUSAL_PATTERNS = (
    "can't assist", "cannot assist", "i'm unable",
    "i cannot help", "i cannot provide", "not able to",
    "inappropriate", "i'm not able", "i refuse",
    "sorry, i can't", "sorry, i cannot",
)


def _is_refusal(text: str) -> bool:
    """Detect model safety-refusal.

    Strips ``<thinking>`` blocks first so that CoT reasoning containing
    stray words like 'sorry' doesn't falsely abort tool routing (#282).
    """
    t = strip_thinking(text).lower().strip()
    return any(p in t for p in _REFUSAL_PATTERNS)


_VALID_ROUTES = frozenset({"tool", "planner", "chat", "agent"})

# Genuine multi-step structure: strong markers only. A naive "and" check
# would break rescued single-tool cases like "move the mouse to 500,300 and
# click" (inp02) — require "then"/"after that" or comma-joined clauses that
# open a new "(and|ve) <verb> my/the …" segment.
_MULTISTEP = re.compile(
    r"\b(then|after that|ve sonra|followed by)\b"
    r"|,\s*(and|ve)\s+\w+\s+(my|the)\b",
    re.IGNORECASE,
)

# "play X on yt music / play some music" — the COT rules deliberately route
# these to planner (open app → search → click); exempt them from the rescue.
_PLAY_MUSIC = re.compile(
    r"\b(play|çal|listen to|oyna)\b.{0,40}\b(music|müzik|song|şarkı|yt|youtube|spotify)\b",
    re.IGNORECASE,
)

# Back-references that justify injecting conversation history into the
# routing prompt. Turkish pronouns included because quick paths may see the
# original text; "o/ona/ondan" = he/she/it forms.
_ANAPHORA = re.compile(
    r"\b(he|she|him|her|it|that one|those|again|earlier|last time|"
    r"what (did|have) (we|you)|do you remember|remember (me|that|what)|"
    r"o|ona|ondan|onu|tekrar|az önce|demin)\b",
    re.IGNORECASE,
)


# Router-side invented-name aliases — mirrors agent/planner.py's
# _PLANNER_ALIASES (the battle-tested set) plus browser_control ACTION
# names the model promotes to tool names. Consulted only for tool names
# that fail registry lookup.
_ROUTER_TOOL_ALIASES: dict[str, str] = {
    "firefox": "browser_control",
    "chrome": "browser_control",
    "chromium": "browser_control",
    "browse": "browser_control",
    "navigate": "browser_control",
    "open": "browser_control",
    "find_and_click": "browser_control",
    "type_in_element": "browser_control",
    "email": "gmail",
    "mail": "gmail",
    "bash": "shell",
    "terminal": "shell",
    "command": "shell",
    "file": "filesystem",
    "files": "filesystem",
    "search": "web_search",
    "google": "web_search",
    "news_tool": "news",
    "remind": "reminder",
    "events": "calendar",
    # Action names promoted to tool names (gemma4 shape)
    "capture": "screenshot",
    "render": "image",
}


def _extract_json(text: str, utterance: str = "") -> dict:
    """Extract the first JSON object from *text*, ignoring markdown fences and thinking blocks.

    Also normalises common llama3.1 mistakes:
    - route field contains a tool name instead of "tool"/"planner"/"chat"

    *utterance* (the English routing input) is used only to guard the
    planner→tool_name rescue: verdicts for genuinely multi-step requests
    (``_MULTISTEP``) or play/music requests (``_PLAY_MUSIC``) keep their
    planner route even when tool_name names a single tool.
    """
    text = strip_thinking(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group() if m else text)

    # Fix: if 'route' is missing or not one of the valid values, the model likely
    # put the tool name there. Move it to tool_name and set route="tool".
    # Gemma variant: route holds the REAL tool and tool_name holds an ACTION
    # name ({"route": "screenshot", "tool_name": "capture"}) — when route is a
    # registered tool and tool_name is not, the route value wins.
    if isinstance(data, dict) and data.get("route") not in _VALID_ROUTES:
        wrong_route = data.get("route")
        log.debug("_extract_json: normalising route=%r → route='tool'", wrong_route)
        if isinstance(wrong_route, str) and wrong_route:
            try:
                from bantz.tools import registry as _registry
                route_is_tool = _registry.get(wrong_route) is not None
                tn = data.get("tool_name")
                name_is_tool = (
                    isinstance(tn, str) and tn and _registry.get(tn) is not None
                )
            except Exception:  # registry unavailable (bare unit tests)
                route_is_tool = name_is_tool = False
            if route_is_tool and not name_is_tool:
                data["tool_name"] = wrong_route
            else:
                data["tool_name"] = data.get("tool_name") or wrong_route
        data["route"] = "tool"

    # Rescue: llama3.1 frequently answers a single-tool request with
    # route="planner" while naming the correct tool (with populated args) in
    # tool_name — measured at 35/42 planner over-routes in
    # eval/routing_eval_results.json.  "planner" is a valid route, so the
    # normalisation above never touches it.  When tool_name is a single
    # registered tool and tool_args is a populated dict, trust tool_name.
    # Known cost: a genuine multi-step request whose JSON also names its first
    # tool is demoted to one tool call — ~2/100 on the eval vs 35/100 rescued.
    if isinstance(data, dict) and data.get("route") == "planner":
        if utterance and (
            _MULTISTEP.search(utterance) or _PLAY_MUSIC.search(utterance)
        ):
            log.debug(
                "rescue skipped: multi-step/play-music structure in %.60r",
                utterance,
            )
            return data
        tool_name = data.get("tool_name")
        tool_args = data.get("tool_args")
        if (
            isinstance(tool_name, str)
            and tool_name
            and tool_name != "planner"
            # delegate_task inside a planner verdict signals genuine
            # multi-step intent (measured: pla01/pla04 on gemma4) — the
            # planner will emit the delegate step itself if appropriate.
            and tool_name != "delegate_task"
            and isinstance(tool_args, dict)
            and tool_args
        ):
            try:
                from bantz.tools import registry as _registry
                known = _registry.get(tool_name) is not None
            except Exception:
                known = False
            if known:
                log.debug(
                    "route/tool_name mismatch rescued: planner → %s", tool_name
                )
                data["route"] = "tool"

    # Invented-name repair: the model sometimes copies a hint's verb phrase
    # ("read_and_summarize_document" for document) or an ACTION name
    # ("navigate" for browser_control) into the tool name.  The planner has
    # _PLANNER_ALIASES for this class; mirror it here for the single-tool
    # path. Two deterministic steps, applied only when tool_name is unknown:
    #   1. alias table (action/topic names → registry names)
    #   2. unique token-substring match against registry names
    if isinstance(data, dict) and data.get("route") == "tool":
        tool_name = data.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            try:
                from bantz.tools import registry as _registry
                if _registry.get(tool_name) is None:
                    norm = tool_name.strip().lower().replace(" ", "_").replace("-", "_")
                    repaired = _ROUTER_TOOL_ALIASES.get(norm)
                    if repaired is None:
                        hits = [
                            n for n in _registry.names()
                            if n in norm.split("_") or (len(n) > 3 and n in norm)
                        ]
                        if len(set(hits)) == 1:
                            repaired = hits[0]
                    if repaired is not None and _registry.get(repaired) is not None:
                        log.debug(
                            "invented tool name repaired: %r → %r",
                            tool_name, repaired,
                        )
                        data["tool_name"] = repaired
            except Exception:  # registry unavailable (e.g. bare unit tests)
                pass

    # Autonomy dial (#492) → whether the user must confirm before this tool
    # runs. Monotonic caution ladder:
    #   low      → always confirm any tool/planner action (even safe)
    #   medium   → confirm moderate + destructive
    #   high     → confirm destructive only (default)
    #   absolute → never confirm, even destructive
    if isinstance(data, dict):
        try:
            from bantz.config import config
            autonomy = (config.autonomy or "high").lower()
        except Exception:
            autonomy = "high"
        if data.get("route") in ("tool", "planner"):
            risk = data.get("risk_level")
            if autonomy == "low":
                data["requires_confirm"] = True
            elif autonomy == "absolute":
                data["requires_confirm"] = False
            elif autonomy == "medium":
                data["requires_confirm"] = risk in ("moderate", "destructive")
            else:  # high (default)
                data["requires_confirm"] = risk == "destructive"
        else:
            data["requires_confirm"] = False

    return data


def _log_thinking(raw: str, tag: str = "") -> None:
    """Pretty-log the thinking section at DEBUG level."""
    m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    if m:
        thought = m.group(1).strip()
        log.debug("Thinking%s: %s", f" ({tag})" if tag else "", thought)


# ── Public API ─────────────────────────────────────────────────────────────────

def _format_recent_history(recent_history: list[dict]) -> str:
    """Format recent conversation turns for context injection."""
    if not recent_history:
        return ""
    lines = []
    for msg in recent_history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


# Maximum tokens to wait for a </thinking> close tag before force-closing.
# Prevents unbounded buffering from models that never emit the close tag.
_THINKING_MAX_TOKENS = 512

# Ollama generation options for routing — cap output length for speed.
# 768 tokens gives enough room for ~4-line <thinking> + JSON output.
# Prevents llama3.1:8b from generating thousands of reasoning tokens.
# Model families with NATIVE thinking support in Ollama. Passing "think" to
# a model without it returns a 400, so the flag is sent only to these.
_NATIVE_THINK_FAMILIES = ("gemma4", "deepseek-r1", "qwen3", "gpt-oss", "magistral")


def _supports_native_think(model: str) -> bool:
    m = (model or "").lower()
    return any(fam in m for fam in _NATIVE_THINK_FAMILIES)


# Routing is a classification task: pin temperature to 0 so identical inputs
# route identically. Measured: FULL (temp 0) 51/100 vs FULL_PROD (prod
# sampling) 45/100 on the same cases; live tests 06/13 flip-flopped between
# runs 30 minutes apart (eval/test_report_current.md).
_ROUTING_OPTIONS: dict = {"num_predict": 768, "temperature": 0}


async def _stream_and_collect(
    messages: list[dict],
    *,
    emit_thinking: bool = True,
    source: str = "cot_route",
    options: dict | None = None,
    model_override: str = "",
    think: bool | None = False,
) -> str:
    """Stream an Ollama chat call, emitting ``thinking_*`` events.

    Accumulates the full response and returns it.  While inside a
    ``<thinking>`` block, each token's inner text (tags stripped) is
    emitted as ``thinking_token`` on the EventBus.  When the block
    closes (or is force-closed after ``_THINKING_MAX_TOKENS``), a
    ``thinking_done`` event is emitted.

    Args:
        messages: Chat messages to send to Ollama.
        emit_thinking: Whether to emit thinking events to the bus.
        source: Label for the ``thinking_done`` event data.

    Returns:
        The complete raw response string (with ``<thinking>`` tags intact
        for downstream JSON extraction).
    """
    buf = ""
    in_thinking = False
    thinking_tokens = 0
    thinking_started = False
    # Once the thinking block closes, never re-detect it.
    # Without this flag, old <thinking> tags accumulate in `buf` and every
    # subsequent token falsely re-triggers detection → ThinkingPanel spam. (#273)
    thinking_complete = False

    if emit_thinking:
        await bus.emit("thinking_start", source=source)

    try:
        # think=False by default: routing is classification — NATIVE thinking
        # (gemma4/deepseek-r1/qwen3) adds unbounded latency before the JSON
        # verdict. The PROMPTED <thinking> in COT_SYSTEM still applies and is
        # bounded by _THINKING_MAX_TOKENS. Models without native thinking
        # ignore the flag... except Ollama 400s on unsupported models, so we
        # only send it when the model is known to support native thinking.
        _think = think if _supports_native_think(model_override or ollama.model) else None
        async for token in ollama.chat_stream(messages, options=options, model_override=model_override, think=_think):
            buf += token

            if not emit_thinking or thinking_complete:
                continue

            # ── Detect JSON start (CoT always ends with {"route": ...) ──
            # When the JSON marker appears, the thinking phase is done.
            if not in_thinking and _JSON_MARKER.search(buf):
                thinking_complete = True
                await bus.emit("thinking_done", source=source)
                continue

            # ── Detect <thinking> open (only before we've seen one block) ─
            if not in_thinking:
                if _THINKING_OPEN.search(buf):
                    in_thinking = True
                    thinking_tokens = 0
                    thinking_started = True
                    # Emit any text *after* the opening tag so far
                    m = _THINKING_OPEN.search(buf)
                    if m:
                        after = buf[m.end():]
                        inner = _clean_thinking_text(after)
                        if inner:
                            await bus.emit("thinking_token", token=inner, source=source)
                    continue
                else:
                    # No <thinking> tag yet — emit token directly as thinking content.
                    # This shows the model's reasoning even for models that don't use tags.
                    inner = _clean_thinking_text(token)
                    if inner:
                        if not thinking_started:
                            thinking_started = True
                        await bus.emit("thinking_token", token=inner, source=source)
                    continue

            # ── Inside <thinking> block ───────────────────────────────
            if in_thinking:
                thinking_tokens += 1
                # Check for close tag only in the current token (not full buf)
                # to avoid re-matching the tag after the block has closed.
                if _THINKING_CLOSE.search(token) or _THINKING_CLOSE.search(buf[-80:]):
                    in_thinking = False
                    thinking_complete = True
                    await bus.emit("thinking_done", source=source)
                    continue

                # Force-close if model never emits </thinking> (#273 Critique 2)
                if thinking_tokens > _THINKING_MAX_TOKENS:
                    in_thinking = False
                    thinking_complete = True
                    log.warning("Force-closing <thinking> after %d tokens (no close tag)", thinking_tokens)
                    await bus.emit("thinking_done", source=source)
                    continue

                # Emit the clean inner token
                inner = _clean_thinking_text(token)
                if inner:
                    await bus.emit("thinking_token", token=inner, source=source)

    except Exception as exc:
        log.warning("Stream error during %s: %s", source, exc)
        # If we were mid-thinking, close it
        if (in_thinking or thinking_started) and not thinking_complete and emit_thinking:
            await bus.emit("thinking_done", source=source)
        raise

    # If thinking phase never completed (e.g. model returned no JSON), emit done
    if not thinking_complete and emit_thinking:
        await bus.emit("thinking_done", source=source)

    return buf


def _clean_thinking_text(text: str) -> str:
    """Strip literal ``<thinking>``/``</thinking>`` tags from token text.

    Only the *inner content* should reach the TUI — raw XML tags must
    never appear on screen (#273 Critique 2).
    """
    text = _THINKING_OPEN.sub("", text)
    text = _THINKING_CLOSE.sub("", text)
    return text.strip()


# ── Fast pre-route regexes — compiled once at module load ─────────────────────
# Both checked against the English input BEFORE the LLM routing call.
# Turkish never reaches this logic; translation happens upstream in ws_server.

_REMINDER_FAST = re.compile(
    r"(?i)"
    r"\bremind me\b|"
    r"\bset (a |an )?(reminder|alarm)\b|"
    r"\breminder in \d+\b|"
    r"\bin \d+ (minute|hour|second)s?\b.*(remind|alert|notify)\b|"
    r"\b(remind|alert|notify) me in \d+\b",
)

_CHAT_FAST = re.compile(
    r"(?i)"
    r"what do you know about\b|"
    r"tell me what you know\b|"
    r"what can you tell me about\b|"
    r"do you know (who|what|anything about)\b|"
    r"do you remember\b|"
    r"have (i|you) told you\b|"
    r"\bwho (?:is|was|were|are)\b(?!.*(?:ceo|founder|president|prime minister|capital))|"
    r"^i (feel|felt|am feeling|was feeling|have been feeling)\b|"
    r"^i (am|was|have been) (so )?(tired|exhausted|stressed|overwhelmed|sad|anxious|frustrated|angry|happy|excited|worried|scared|afraid|depressed)\b|"
    r"\bi (can't|cannot) (handle|deal with|cope)\b|"
    r"\beverything is (piling up|too much|overwhelming)\b|"
    r"\bi have (so much|too much|a lot of) (to do|work)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Music-listening intent — unambiguous by construction, so it earns a
# pre-LLM fast-path. Measured motivation: "i want to listen acdc" drew
# four DIFFERENT LLM verdicts across four temp-0 runs (vision_execute,
# web_search, vision_execute, browser_control) — Ollama's concurrent
# batching makes boundary verdicts nondeterministic. Bare "play X" is NOT
# matched (could be video/game/chess); "listen" wishes and explicit
# music-service phrasings are.
_MUSIC_FAST = re.compile(
    r"(?i)"
    r"\b(?:i (?:want|wanna|would like) to listen|wanna listen|let'?s listen|listen to)\b|"
    r"\bput on some\b|"
    r"\bplay\b.{1,40}\bon (?:yt|youtube) ?music\b|"
    r"\bplay some music\b|"
    r"\b(?:müzik|şarkı) (?:aç|çal)\b",
)

# Video-watching intent — same reasoning as _MUSIC_FAST.
# Also covers the natural "open/show/find <topic> video(s)" phrasing and the
# bare "<topic> videos" shorthand ("star videos", "quantum videos") — both
# mean "find that on YouTube". The bare form requires a SINGLE leading topic
# word so it can't swallow other-verb sentences like "delete my videos".
_VIDEO_FAST = re.compile(
    r"(?i)"
    r"\b(?:let'?s|let us|i (?:want|wanna|would like) to|wanna|can we)\s+watch\b|"
    r"\bwatch\b.{1,50}\bon youtube\b|"
    r"\bizleyelim\b|\bizlemek istiyorum\b|"
    r"^\s*(?:open|show(?: me)?|find|watch|pull up|put on|play)\s+[\w'/&. -]{1,40}\bvideos?\b[.!?]?\s*$|"
    r"^\s*[\w'&-]{2,20}\s+videos?[.!?]?\s*$|"
    r"\bvideo (?:aç|izle)\b"
)

# Workspace / virtual-desktop switching — unambiguous, so it gets a
# deterministic fast-path to the desktop tool (Hyprland/Sway/X11). Without
# this, "workspace 3" fell to the LLM emitting a raw hyprctl command, which
# the user saw fail ("hyprctl failed").
_WORKSPACE_NUM = re.compile(
    r"\bworkspace\s+(\d{1,2})\b|"
    r"\b(?:çalışma alanı|masaüstü)\s*(\d{1,2})\b",
    re.IGNORECASE,
)
_WORKSPACE_REL = re.compile(
    r"\b(next|previous|prev)\s+workspace\b|"
    r"\bworkspace\s+(next|previous|prev)\b",
    re.IGNORECASE,
)

# Click-by-description — "click on X", "click the X button", "double click X",
# Turkish "X'e tıkla". Deterministic so a one-word or unfamiliar target (e.g.
# "click on deneme") isn't second-guessed into chat by the router — which then
# leaks a fake "visual_click(deneme)" string back to the user instead of
# actually clicking.
_CLICK_FAST = re.compile(
    r"^\s*(?P<act>double[ -]?click|right[ -]?click|left[ -]?click|click)"
    r"(?:\s+(?:on|the|upon))?\s+(?P<target>.{1,60}?)\s*[.!?]*$",
    re.IGNORECASE,
)
_CLICK_FAST_TR = re.compile(
    r"^\s*(?P<target>.{1,50}?)['’]?\s*[a-zçğıöşü]{0,3}\s+(?:tıkla|tikla)\s*[.!?]*$",
    re.IGNORECASE,
)


def _fastpath_route(en_input: str) -> tuple[dict, None] | None:
    """Deterministic pre-route regexes, checked BEFORE the LLM call.

    Returns a ``(plan, None)`` tuple exactly like a successful cot_route
    verdict, or ``None`` to fall through to the LLM. Extracted from
    cot_route so the C1 recovery loop can bypass the whole family with
    ``skip_fastpath=True`` (#502) — these match on input text, which does
    not change after a tool failure, so re-entering them from a re-decide
    call would re-select the same failed tool forever.
    """
    # "investigate: <anomaly> — <detail>" comes from the Anomaly Watch
    # Investigate button — Bantz should analyze it conversationally, never
    # delegate it to a sub-agent.
    if en_input.strip().lower().startswith("investigate:"):
        log.debug("cot_route fast-path: pre-route to chat for investigate directive")
        return {"route": "chat", "tool_name": None, "tool_args": {}, "confidence": 0.97, "reasoning": "pre-route: investigate → conversational analysis"}, None

    if _REMINDER_FAST.search(en_input):
        log.debug("cot_route fast-path: pre-route to reminder for: %.80s", en_input)
        _time_m = re.search(r"in (\d+)\s*(minute|hour|second)", en_input, re.IGNORECASE)
        _minutes = int(_time_m.group(1)) if _time_m else 10
        if _time_m and "hour" in _time_m.group(2).lower():
            _minutes *= 60
        return {"route": "tool", "tool_name": "reminder", "tool_args": {"action": "add", "title": en_input, "minutes": _minutes}, "confidence": 0.9, "reasoning": "pre-route: reminder pattern"}, None

    if _CHAT_FAST.search(en_input):
        log.debug("cot_route fast-path: pre-route to chat for: %.80s", en_input)
        return {"route": "chat", "tool_name": None, "tool_args": {}, "confidence": 0.95, "reasoning": "pre-route: knowledge/emotional pattern"}, None

    if _MUSIC_FAST.search(en_input):
        log.debug("cot_route fast-path: pre-route to vision_execute (music) for: %.80s", en_input)
        return {
            "route": "tool", "tool_name": "vision_execute",
            "tool_args": {"goal": en_input, "intent": "play_music"},
            "risk_level": "moderate", "confidence": 0.95,
            "reasoning": "pre-route: music-listening pattern",
        }, None

    if _VIDEO_FAST.search(en_input):
        log.debug("cot_route fast-path: pre-route to vision_execute (video) for: %.80s", en_input)
        return {
            "route": "tool", "tool_name": "vision_execute",
            "tool_args": {"goal": en_input, "intent": "watch_video"},
            "risk_level": "moderate", "confidence": 0.95,
            "reasoning": "pre-route: video-watching pattern",
        }, None

    _ws = _WORKSPACE_NUM.search(en_input)
    if _ws:
        target = _ws.group(1) or _ws.group(2)
        log.debug("cot_route fast-path: pre-route to desktop workspace %s", target)
        return {
            "route": "tool", "tool_name": "desktop",
            "tool_args": {"action": "workspace", "target": target},
            "risk_level": "moderate", "confidence": 0.95,
            "reasoning": "pre-route: workspace switch",
        }, None
    if _WORKSPACE_REL.search(en_input):
        target = "next" if "next" in en_input.lower() else "prev"
        log.debug("cot_route fast-path: pre-route to desktop workspace %s", target)
        return {
            "route": "tool", "tool_name": "desktop",
            "tool_args": {"action": "workspace", "target": target},
            "risk_level": "moderate", "confidence": 0.95,
            "reasoning": "pre-route: workspace switch (relative)",
        }, None

    _clk = _CLICK_FAST.match(en_input)
    _clk_target = _clk.group("target").strip() if _clk else ""
    _clk_act = (_clk.group("act").lower().replace(" ", "").replace("-", "")
                if _clk else "")
    if not _clk:
        _clk_tr = _CLICK_FAST_TR.match(en_input)
        if _clk_tr:
            _clk_target, _clk_act = _clk_tr.group("target").strip(), "click"
    # Skip pure-coordinate clicks ("click 960 540") — those use /click; and
    # skip empty targets.
    if _clk_target and not re.fullmatch(r"[\d ,]+", _clk_target):
        action = {"doubleclick": "double_click", "rightclick": "right_click",
                  "leftclick": "click", "click": "click"}.get(_clk_act, "click")
        log.debug("cot_route fast-path: pre-route to visual_click(%r, %s)",
                  _clk_target, action)
        return {
            "route": "tool", "tool_name": "visual_click",
            "tool_args": {"target": _clk_target, "action": action},
            "risk_level": "moderate", "confidence": 0.95,
            "reasoning": "pre-route: click-by-description",
        }, None

    return None


async def cot_route(
    en_input: str,
    tool_schemas: list[dict],
    *,
    recent_history: list[dict] | None = None,
    tool_context: str = "",
    confidence_threshold: float = 0.4,
    skip_fastpath: bool = False,
) -> tuple[dict | None, str | None]:
    """
    Chain-of-Thought routing via Ollama **streaming** (#273).

    Returns ``(plan, routing_error)``:

    - ``(plan_dict, None)``     — successful routing (tool or chat).
    - ``(None, None)``          — pure chat / refusal / low confidence
                                  (no tool was attempted).
    - ``(None, error_string)``  — a tool route was *attempted* but JSON
                                  parsing or validation failed.  The
                                  caller **must not** silently fall back
                                  to chat — it should inform the user
                                  (#253 People-Pleaser fix).

    Now streams via ``ollama.chat_stream()`` and emits ``thinking_token``
    events on the EventBus so the TUI can display the chain-of-thought
    in real-time.

    Parameters
    ----------
    tool_context : str
        Optional dynamic context block (e.g. recent email IDs, calendar
        events) injected only when relevant (#275 — avoids bloating the
        prompt when unrelated queries are asked).
    skip_fastpath : bool
        Bypass ALL pre-route regexes (including the ``investigate:``
        pre-route) and go straight to the LLM. REQUIRED for the C1
        recovery loop's re-decide calls (audit C1, #502): the fast-paths
        match on the *input text*, which does not change after a tool
        failure — re-entering them would re-select the same failed tool
        every iteration (an infinite same-tool loop, not a re-decision).
        Do not remove this parameter or default it to True.
    """
    # Pre-route regexes live in _fastpath_route; the C1 recovery loop's
    # re-decide calls MUST bypass them (see skip_fastpath in the docstring).
    if not skip_fastpath:
        fast = _fastpath_route(en_input)
        if fast is not None:
            return fast

    schema_str = _build_compact_schemas(tool_schemas)

    # Build optional history block for coreference resolution.
    # Anaphora-gated: inject history ONLY when the input actually contains a
    # back-reference. Unconditional injection caused chat answers to inherit
    # the previous turn's topic (live re-run tests 19/20 both answered about
    # Alan Turing when asked unrelated questions).
    history_block = ""
    if recent_history and _ANAPHORA.search(en_input):
        formatted = _format_recent_history(recent_history)
        history_block = (
            f"\n\nRECENT CONVERSATION — use ONLY to resolve references like "
            f"'him', 'it', 'that file', 'again'. Do NOT answer about these "
            f"topics unless the current message explicitly asks:\n{formatted}"
        )

    # Build optional dynamic tool context (#275)
    tool_ctx_block = ""
    if tool_context:
        tool_ctx_block = f"\n\n{tool_context}"

    messages: list[dict] = [
        {"role": "system", "content": COT_SYSTEM.format(tool_schemas=schema_str) + history_block + tool_ctx_block},
        {"role": "user", "content": en_input},
    ]

    raw: str = ""

    # ── Attempt 1 (Ollama streaming) ──────────────────────────────────
    try:
        raw = await _stream_and_collect(
            messages, emit_thinking=True, source="cot_route",
            options=_ROUTING_OPTIONS, model_override=ollama.routing_model,
        )

        if _is_refusal(raw):
            log.warning("CoT refused (attempt 1): %.100s", raw)
            return None, None

        _log_thinking(raw)
        plan = _extract_json(raw, utterance=en_input)

        log.info("CoT parsed: route=%s tool=%s conf=%.2f",
                 plan.get("route"), plan.get("tool_name"), plan.get("confidence", 0))

        if plan.get("confidence", 0.5) < confidence_threshold:
            log.info("CoT low confidence (%.2f) — falling back", plan["confidence"])
            return None, None

        return plan, None

    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("CoT JSON parse failed (attempt 1): %s — raw: %.200s", exc, raw)
        # Fall through to attempt 2
    except Exception as exc:
        # If the routing model is missing/broken, retry once with main model
        err_lower = str(exc).lower()
        if any(hint in err_lower for hint in ("404", "not found", "model")):
            log.warning(
                "Routing model '%s' failed (%s) — retrying with main model '%s'",
                ollama.routing_model, exc, ollama.model,
            )
            try:
                raw = await _stream_and_collect(
                    messages, emit_thinking=True, source="cot_route_fallback",
                    options=_ROUTING_OPTIONS, model_override=ollama.model,
                )
                if not _is_refusal(raw):
                    _log_thinking(raw)
                    plan = _extract_json(raw, utterance=en_input)
                    log.info("CoT fallback parsed: route=%s tool=%s conf=%.2f",
                             plan.get("route"), plan.get("tool_name"),
                             plan.get("confidence", 0))
                    if plan.get("confidence", 0.5) >= confidence_threshold:
                        # Fix routing_model for subsequent calls this session
                        ollama.routing_model = ollama.model
                        return plan, None
            except Exception as fb_exc:
                log.warning("CoT fallback also failed: %s", fb_exc)
        log.warning("CoT error (attempt 1): %s", exc)
        return None, f"CoT routing error: {exc}"

    # ── Attempt 2: retry with correction (streaming, no thinking events) ──
    try:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "That was not valid JSON. Ensure you include the <thinking> tags FIRST, "
                "then output ONLY a single JSON object "
                "with keys: route, tool_name, tool_args, risk_level, confidence, reasoning. "
                "route must be one of: tool, planner, chat. "
                "No markdown. No explanation."
            ),
        })

        raw2 = await _stream_and_collect(
            messages, emit_thinking=False, source="cot_route_retry",
            options=_ROUTING_OPTIONS, model_override=ollama.routing_model,
        )
        _log_thinking(raw2, tag="retry")
        plan = _extract_json(raw2, utterance=en_input)
        return plan, None

    except Exception as exc:
        log.warning("CoT parse failed (attempt 2): %s", exc)
        return None, f"Intent routing failed after 2 attempts: {exc}"
