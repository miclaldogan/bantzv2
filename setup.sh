#!/usr/bin/env bash
# Bantz v2 — Quick setup
set -e

echo "🔧 Bantz v2 setup starting..."

# Python interpreter check (prefer python3.11+, fallback to python3 if suitable)
PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"; then
  PYTHON_BIN="python3"
else
  echo "❌ Python 3.11+ required"
  exit 1
fi

# Venv
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

# Packages
pip install -e "." --quiet

# .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "✓ .env created from .env.example"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python -m bantz --doctor    # system check"
echo "  python -m bantz             # start TUI"