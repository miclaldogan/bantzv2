def fix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Let's add `import asyncio` right after `from __future__ import annotations`
    if "import asyncio\n" not in content:
        content = content.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport asyncio\n")

    with open(filepath, 'w') as f:
        f.write(content)

fix('src/bantz/agent/workflows/maintenance.py')
fix('src/bantz/agent/workflows/reflection.py')
