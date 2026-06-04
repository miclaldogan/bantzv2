def fix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # We need to add import asyncio at the top level, AFTER __future__ and AFTER the docstring.
    # The file already had `import asyncio` but maybe in a specific block or not at the top.
    # Let's just add it near the other imports.
    import_block = "import logging\nimport os\n"
    if import_block in content and "import asyncio\n" not in content[:content.find(import_block)+100]:
        content = content.replace(import_block, "import asyncio\n" + import_block)

    with open(filepath, 'w') as f:
        f.write(content)

fix('src/bantz/agent/workflows/maintenance.py')
fix('src/bantz/agent/workflows/reflection.py')
