import re

with open('tests/tools/test_gui_tool.py', 'r') as f:
    content = f.read()

def replacer(match):
    prefix = match.group(1)
    args = match.group(2)
    # mock_pag -> mock_get_pag
    new_args = args.replace('mock_pag', 'mock_get_pag')
    body = match.group(3)
    new_body = body.replace('mock_pag', 'mock_get_pag.return_value')
    # but some places use mock_pag to reference the mock object itself
    # e.g., mock_pag.ImageNotFoundException
    # so we need mock_pag = mock_get_pag.return_value inside the function
    return f"{prefix}{new_args}:\n        mock_pag = MagicMock()\n        mock_get_pag.return_value = mock_pag{body}"


# Let's use a simpler approach.
# Just replace mock_pag with mock_get_pag in args, then add mock_pag = MagicMock(); mock_get_pag.return_value = mock_pag
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'def test_' in line and 'mock_pag' in line:
        new_lines.append(line.replace('mock_pag', 'mock_get_pag'))
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * (indent + 4) + 'mock_pag = MagicMock()')
        new_lines.append(' ' * (indent + 4) + 'mock_get_pag.return_value = mock_pag')
    else:
        new_lines.append(line)

with open('tests/tools/test_gui_tool.py', 'w') as f:
    f.write('\n'.join(new_lines))
