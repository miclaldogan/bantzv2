import re

with open("tests/tools/test_gui_tool.py", "r") as f:
    content = f.read()

# Replace the sys.modules['pyautogui'] hack with a cleaner patch approach, since it alters the test environment
new_imports = """
import os
import subprocess
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from bantz.tools.gui_tool import (
    CACHE_DIR,
    GUITool,
    GUIToolError,
    gui_tool,
)
"""

content = re.sub(r'import subprocess\nfrom types import SimpleNamespace\nfrom unittest\.mock import MagicMock, patch\nimport sys\n\n# Mock pyautogui before importing bantz\.tools\.gui_tool to avoid DISPLAY error\nsys\.modules\[\'pyautogui\'\] = MagicMock\(\)\n\nimport pytest\n\n\nfrom bantz\.tools\.gui_tool import \(\n    GUITool,\n    GUIToolError,\n    gui_tool,\n\)', new_imports.strip(), content)

with open("tests/tools/test_gui_tool.py", "w") as f:
    f.write(content)
