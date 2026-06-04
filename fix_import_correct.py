def fix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Let's add `import asyncio` right after `from typing import ...`
    if "import asyncio\n" not in content:
        content = content.replace("from typing import ", "import asyncio\nfrom typing import ")

    with open(filepath, 'w') as f:
        f.write(content)

fix('src/bantz/agent/workflows/maintenance.py')
fix('src/bantz/agent/workflows/reflection.py')
