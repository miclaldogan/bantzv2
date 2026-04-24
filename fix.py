# Error: FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/bantzv2/bantzv2/src/bantz/interface/tui/styles.tcss'
# The tests in `tests/tui/test_header.py` and `tests/tui/test_toast.py` rely on `src/bantz/interface/tui/styles.tcss`.
# The file probably doesn't exist anymore or has been renamed, but the tests are still trying to access it.
# Let's check if the file exists or is named differently.
import os
print("Checking for styles.tcss:", os.path.exists("src/bantz/interface/tui/styles.tcss"))
print("Files in tui:", os.listdir("src/bantz/interface/tui"))
