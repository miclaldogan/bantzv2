import sys

filepath = 'tests/core/test_routing_engine.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = """             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.finalize_plan") as mock_finalizer:"""

new_code = """             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:"""

content = content.replace(old_code, new_code)

with open(filepath, 'w') as f:
    f.write(content)
print("Patched routing engine tests successfully")
