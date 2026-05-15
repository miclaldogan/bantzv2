with open("src/bantz/core/routing_engine.py", "r") as f:
    content = f.read()

old_code = '''        "stackoverflow": "https://stackoverflow.com",
    }
    m = re.search(
        r"(?:open|launch|go\s+to|navigate\s+to)\s+(\w+)"
        r"(?:\s+(?:in\s+(?:the\s+)?)?(?:web\s*)?browser|\s+web(?:site)?)?",
        both,
    )
    if m:
        app_name = m.group(1).lower()
        web_url = _WEB_APP_URLS.get(app_name)
        if web_url:
            return {"tool": "browser_control", "args": {"action": "navigate", "url": web_url}}'''

new_code = '''        "stackoverflow": "https://stackoverflow.com",
    }
    # Removed in #340 fix
    # m = re.search(
    #     r"(?:open|launch|go\s+to|navigate\s+to)\s+(\w+)"
    #     r"(?:\s+(?:in\s+(?:the\s+)?)?(?:web\s*)?browser|\s+web(?:site)?)?",
    #     both,
    # )
    # if m:
    #     app_name = m.group(1).lower()
    #     web_url = _WEB_APP_URLS.get(app_name)
    #     if web_url:
    #         return {"tool": "browser_control", "args": {"action": "navigate", "url": web_url}}'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("src/bantz/core/routing_engine.py", "w") as f:
        f.write(content)
    print("Fixed src/bantz/core/routing_engine.py")
else:
    print("Could not find old code")
