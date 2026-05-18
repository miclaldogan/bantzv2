with open("src/bantz/tools/desktop.py") as f:
    content = f.read()
if "score = element.get" in content:
    print("score still found in desktop.py")
