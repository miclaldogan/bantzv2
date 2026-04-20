import re

with open("src/bantz/data/sqlite_store.py", "r") as f:
    content = f.read()

def repl(m):
    table = m.group(1)
    return f"""    def exists(self) -> bool:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM {table} LIMIT 1"
            ).fetchone()
        return row is not None"""

new_content = re.sub(
    r'    def exists\(self\) -> bool:\n        with get_pool\(\)\.connection\(\) as conn:\n            row = conn\.execute\(\n                "SELECT COUNT\(\*\) FROM ([^"]+)"\n            \)\.fetchone\(\)\n        return row\[0\] > 0',
    repl,
    content
)

with open("src/bantz/data/sqlite_store.py", "w") as f:
    f.write(new_content)
