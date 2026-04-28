import re

with open('tests/tools/test_gui_tool.py', 'r') as f:
    content = f.read()

content = content.replace('from bantz.tools.gui_tool import (\n    CACHE_DIR,\n    GUITool,\n    GUIToolError,\n    gui_tool,\n)', 'from bantz.tools.gui_tool import (\n    CACHE_DIR,\n    GUITool,\n    GUIToolError,\n    gui_tool,\n    _get_pyautogui,\n)')
content = content.replace('@patch("bantz.tools.gui_tool.pyautogui")', '@patch("bantz.tools.gui_tool._get_pyautogui")')

lines = content.split('\n')
new_lines = []
for line in lines:
    if line.strip().startswith('def test_') and 'mock_pag' in line:
        new_lines.append(line.replace('mock_pag', 'mock_get_pag'))
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * (indent + 4) + 'mock_pag = MagicMock()')
        new_lines.append(' ' * (indent + 4) + 'mock_get_pag.return_value = mock_pag')
    elif line.strip().startswith('async def test_') and 'mock_pag' in line:
        new_lines.append(line.replace('mock_pag', 'mock_get_pag'))
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * (indent + 4) + 'mock_pag = MagicMock()')
        new_lines.append(' ' * (indent + 4) + 'mock_get_pag.return_value = mock_pag')
    else:
        new_lines.append(line)

content = '\n'.join(new_lines)
# Also fix the TestPyautoguiConfig
content = content.replace('import pyautogui', 'pg = _get_pyautogui()\n        if not pg: return\n')
content = content.replace('assert pyautogui.FAILSAFE is False', 'assert pg.FAILSAFE is False')
content = content.replace('assert pyautogui.PAUSE == 0', 'assert pg.PAUSE == 0')

with open('tests/tools/test_gui_tool.py', 'w') as f:
    f.write(content)
