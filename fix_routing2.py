with open("src/bantz/core/routing_engine.py", "r") as f:
    content = f.read()

old_code = r'''    # ── URL navigation (explicit URL in input) ────────────────────────
    m = re.search(
        r"(?:go\s+to|open|navigate\s+(?:to)?)\s+"
        r"(https?://\S+|(?:www\.)\S+)",
        both,
    )
    if m:
        url = m.group(1)
        if not url.startswith("http"):
            url = "https://" + url
        return {"tool": "browser_control", "args": {"action": "navigate", "url": url}}'''

new_code = r'''    # ── URL navigation (explicit URL in input) ────────────────────────
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
    #     return {"tool": "browser_control", "args": {"action": "navigate", "url": url}}'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("src/bantz/core/routing_engine.py", "w") as f:
        f.write(content)
    print("Fixed src/bantz/core/routing_engine.py")
else:
    print("Could not find old code")
