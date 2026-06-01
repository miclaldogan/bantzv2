#!/usr/bin/env bash
# Bantz installer — curl -fsSL https://raw.githubusercontent.com/miclaldogan/bantzv2/main/install.sh | bash
set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
CYN='\033[0;36m'; BLD='\033[1m';    RST='\033[0m'

ok()   { echo -e "${GRN}[✓]${RST} $*"; }
warn() { echo -e "${YLW}[!]${RST} $*"; }
err()  { echo -e "${RED}[✗]${RST} $*" >&2; }
die()  { err "$@"; exit 1; }
step() { echo -e "\n${CYN}${BLD}[$1]${RST} $2"; }

# ── ASCII Header ─────────────────────────────────────────────────────────────
cat <<'BANNER'

  ██████╗  █████╗ ███╗   ██╗████████╗███████╗
  ██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝╚══███╔╝
  ██████╔╝███████║██╔██╗ ██║   ██║     ███╔╝
  ██╔══██╗██╔══██║██║╚██╗██║   ██║    ███╔╝
  ██████╔╝██║  ██║██║ ╚████║   ██║   ███████╗
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝

  Your local-first AI assistant — terminal-native, privacy-first.
  ─────────────────────────────────────────────────────────────────

BANNER

echo -e "${BLD}Welcome to the Bantz installer.${RST}"
echo "This script clones Bantz, installs it, walks you through configuration,"
echo "and runs a health check. Press Ctrl-C at any time to abort."

# ── Step 1: Python 3.11+ ─────────────────────────────────────────────────────
step "1/9" "Checking Python version…"
if ! command -v python3 &>/dev/null; then
  die "Python 3.11+ is required. Install it from https://python.org and re-run."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  die "Python 3.11+ required (found ${PY_VER}). Please upgrade and re-run."
fi
ok "Python ${PY_VER}"

# ── Step 2: git ───────────────────────────────────────────────────────────────
step "2/9" "Checking git…"
if ! command -v git &>/dev/null; then
  die "git is required. Install it (e.g. sudo apt install git) and re-run."
fi
ok "git $(git --version | awk '{print $3}')"

# ── Step 3: Ollama ────────────────────────────────────────────────────────────
step "3/9" "Checking Ollama…"
if curl -sf --connect-timeout 3 http://localhost:11434 &>/dev/null; then
  ok "Ollama reachable at http://localhost:11434"
else
  warn "Ollama is NOT reachable at http://localhost:11434."
  warn "Install Ollama from https://ollama.com and start it before using Bantz."
  warn "Continuing install — run 'bantz --doctor' later to verify connectivity."
fi

# ── Step 4: Clone or update ───────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.local/share/bantz/src"
step "4/9" "Cloning / updating source at ${INSTALL_DIR}…"
mkdir -p "$(dirname "$INSTALL_DIR")"

if [ -d "${INSTALL_DIR}/.git" ]; then
  echo "  Existing install found — pulling latest changes…"
  git -C "$INSTALL_DIR" pull --ff-only
  ok "Source updated"
else
  git clone https://github.com/miclaldogan/bantzv2.git "$INSTALL_DIR"
  ok "Repository cloned to ${INSTALL_DIR}"
fi

# ── Step 5: pip install ───────────────────────────────────────────────────────
step "5/9" "Installing Bantz Python package (this may take a minute)…"
PIP=pip3
if ! command -v pip3 &>/dev/null; then
  if command -v pip &>/dev/null; then
    PIP=pip
  else
    die "pip not found. Fix with: python3 -m ensurepip --upgrade"
  fi
fi
"$PIP" install -e "${INSTALL_DIR}[dev]" --quiet
ok "Package installed"

# ── Step 6: PATH ──────────────────────────────────────────────────────────────
LOCAL_BIN="${HOME}/.local/bin"
step "6/9" "Verifying PATH…"
if command -v bantz &>/dev/null; then
  ok "bantz found at $(command -v bantz)"
else
  warn "bantz not found in PATH. Adding ${LOCAL_BIN} to PATH…"
  EXPORT_LINE='export PATH="${HOME}/.local/bin:${PATH}"'
  for RC in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [ -f "$RC" ] && ! grep -q ".local/bin" "$RC"; then
      { echo ""; echo "# Added by Bantz installer"; echo "$EXPORT_LINE"; } >> "$RC"
      echo "  Updated: $RC"
    fi
  done
  export PATH="${LOCAL_BIN}:${PATH}"
  ok "PATH updated — restart your shell or run: export PATH=\"${LOCAL_BIN}:\${PATH}\""
fi

# ── Step 7: Setup wizard ──────────────────────────────────────────────────────
step "7/9" "${BLD}Setup Wizard${RST} — press Enter to accept defaults shown in [brackets]."
echo

_ask() {
  local prompt="$1" default="$2"
  printf "  %b%s%b [%b%s%b]: " "$BLD" "$prompt" "$RST" "$CYN" "$default" "$RST" >&2
  read -r REPLY < /dev/tty
  printf '%s' "${REPLY:-$default}"
}

_ask_yn() {
  local prompt="$1" default="$2"
  printf "  %b%s%b (y/n) [%b%s%b]: " "$BLD" "$prompt" "$RST" "$CYN" "$default" "$RST" >&2
  read -r REPLY < /dev/tty
  printf '%s' "${REPLY:-$default}"
}

OLLAMA_MODEL=$(_ask "Ollama model for Bantz" "llama3.1:8b")
LANGUAGE=$(_ask "Preferred language (tr/en)" "tr")

GEMINI_ENABLED="false"; GEMINI_KEY=""
GEMINI_RESP=$(_ask_yn "Enable Gemini as a fallback LLM?" "n")
if [[ "$GEMINI_RESP" =~ ^[Yy] ]]; then
  GEMINI_ENABLED="true"
  GEMINI_KEY=$(_ask "Gemini API key" "")
fi

PICO_KEY=""; WAKE_WORD="false"
PICO_RESP=$(_ask_yn "Do you have a Porcupine key for wake word detection?" "n")
if [[ "$PICO_RESP" =~ ^[Yy] ]]; then
  WAKE_WORD="true"
  PICO_KEY=$(_ask "Porcupine access key" "")
fi

TG_TOKEN=""; TG_USERS=""
TG_RESP=$(_ask_yn "Set up Telegram remote access?" "n")
if [[ "$TG_RESP" =~ ^[Yy] ]]; then
  TG_TOKEN=$(_ask "Telegram bot token" "")
  TG_USERS=$(_ask "Whitelisted Telegram user ID(s) (comma-separated)" "")
fi

# ── Step 8: Write .env ────────────────────────────────────────────────────────
ENV_DIR="${HOME}/.local/share/bantz"
ENV_FILE="${ENV_DIR}/.env"
TEMPLATE="${INSTALL_DIR}/.env.example"
step "8/9" "Writing ${ENV_FILE}…"
mkdir -p "$ENV_DIR"

_set_key() {
  local key="$1" val="$2" file="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$file"
  else
    echo "${key}=${val}" >> "$file"
  fi
}

cp "$TEMPLATE" "$ENV_FILE"
_set_key "BANTZ_OLLAMA_MODEL"      "$OLLAMA_MODEL"  "$ENV_FILE"
_set_key "BANTZ_LANGUAGE"          "$LANGUAGE"      "$ENV_FILE"
_set_key "BANTZ_GEMINI_ENABLED"    "$GEMINI_ENABLED" "$ENV_FILE"
[ -n "$GEMINI_KEY"  ] && _set_key "BANTZ_GEMINI_API_KEY"        "$GEMINI_KEY"  "$ENV_FILE"
_set_key "BANTZ_WAKE_WORD_ENABLED" "$WAKE_WORD"     "$ENV_FILE"
[ -n "$PICO_KEY"    ] && _set_key "BANTZ_PICOVOICE_ACCESS_KEY"  "$PICO_KEY"   "$ENV_FILE"
[ -n "$TG_TOKEN"    ] && _set_key "TELEGRAM_BOT_TOKEN"          "$TG_TOKEN"   "$ENV_FILE"
[ -n "$TG_USERS"    ] && _set_key "TELEGRAM_ALLOWED_USERS"      "$TG_USERS"   "$ENV_FILE"

# Symlink into source dir so bantz can load it from its working directory
if [ ! -e "${INSTALL_DIR}/.env" ]; then
  ln -s "$ENV_FILE" "${INSTALL_DIR}/.env"
fi
ok ".env written"

# ── Step 9: Doctor ────────────────────────────────────────────────────────────
step "9/9" "Running bantz --doctor…"
echo
BANTZ_BIN="$(command -v bantz 2>/dev/null || echo "${LOCAL_BIN}/bantz")"
"$BANTZ_BIN" --doctor 2>&1 || true

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo -e "${GRN}${BLD}═══════════════════════════════════════════════════${RST}"
echo -e "${GRN}${BLD}  Bantz is installed and ready.${RST}"
echo -e "${GRN}${BLD}═══════════════════════════════════════════════════${RST}"
echo
echo -e "  Run ${BLD}bantz${RST}           to start the terminal interface."
echo -e "  Run ${BLD}bantz --daemon${RST}  for headless background mode."
echo -e "  Run ${BLD}bantz --doctor${RST}  to check your configuration at any time."
echo
