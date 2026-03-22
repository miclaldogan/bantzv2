import re

with open("src/bantz/core/brain.py", "r") as f:
    content = f.read()

# Looks like it's already there in __init__ but maybe it's not?
# Wait, let's check line 128 of brain.py in the CI
