with open("src/bantz/core/routing_engine.py", "r") as f:
    content = f.read()

old_code = '''    # ── App launches (unambiguous desktop apps) ───────────────────────
    # These bypass LLM entirely for instant response (#340 speed fix).
    m = re.search(
        r"(?:open|launch|start|run)\s+"
        r"(firefox|chrome|chromium|terminal|files|vscode|gedit)",
        both,
    )
    if m:
        return {"tool": "browser_control", "args": {"action": "open", "app": m.group(1)}}

    # ── URL navigation (explicit URL in input) ────────────────────────
    m = re.search(
        r"(?:go\s+to|open|navigate\s+(?:to)?)\s+"
        r"(https?://\S+|(?:www\.)\S+)",
        both,
    )
    if m:
        url = m.group(1)
        if not url.startswith("http"):
            url = "https://" + url
        return {"tool": "browser_control", "args": {"action": "navigate", "url": url}}

    # ── Well-known web apps (open Gemini, open ChatGPT, etc.) ─────────
    _WEB_APP_URLS: dict[str, str] = {'''

new_code = '''    # ── App launches (unambiguous desktop apps) ───────────────────────
    # These bypass LLM entirely for instant response (#340 speed fix).
    # Removed in #340 fix
    # m = re.search(
    #     r"(?:open|launch|start|run)\s+"
    #     r"(firefox|chrome|chromium|terminal|files|vscode|gedit)",
    #     both,
    # )
    # if m:
    #     return {"tool": "browser_control", "args": {"action": "open", "app": m.group(1)}}

    # ── URL navigation (explicit URL in input) ────────────────────────
    # Removed in #340 fix
    # m = re.search(
    #     r"(?:go\s+to|open|navigate\s+(?:to)?)\s+"
    #     r"(https?://\S+|(?:www\.)\S+)",
    #     both,
    # )
    # if m:
    #     url = m.group(1)
    #     if not url.startswith("http"):
    #         url = "https://" + url
    #     return {"tool": "browser_control", "args": {"action": "navigate", "url": url}}

    # ── Well-known web apps (open Gemini, open ChatGPT, etc.) ─────────
    _WEB_APP_URLS: dict[str, str] = {'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("src/bantz/core/routing_engine.py", "w") as f:
        f.write(content)
    print("Fixed src/bantz/core/routing_engine.py")
else:
    print("Could not find old code")
