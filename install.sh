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

# ── OS Detection ─────────────────────────────────────────────────────────────
_os="$(uname -s 2>/dev/null || echo unknown)"
if [[ "$_os" == MINGW* || "$_os" == CYGWIN* || "$_os" == MSYS* ]]; then
  echo
  echo -e "${YLW}${BLD}Windows detected (Git Bash / MSYS).${RST}"
  echo
  echo "The Bantz backend requires Linux. You have two options:"
  echo
  echo -e "  ${BLD}Option 1 — WSL2 (recommended, full Bantz)${RST}"
  echo "    Install WSL2 from the Microsoft Store, then inside WSL run:"
  echo "    curl -fsSL https://raw.githubusercontent.com/miclaldogan/bantzv2/main/install.sh | bash"
  echo
  echo -e "  ${BLD}Option 2 — UI only (Windows-native desktop app)${RST}"
  echo "    Install Node.js (https://nodejs.org) and Rust (https://rustup.rs), then:"
  echo "    git clone https://github.com/miclaldogan/bantzv2.git"
  echo "    cd bantzv2/bantz-ui && npm install && npm run tauri:build"
  echo "    The installer appears in src-tauri/target/release/bundle/"
  echo "    Point it at a Bantz backend on WSL2 or a Linux machine."
  echo
  exit 0
fi

# ── Step 1: Python 3.11+ ─────────────────────────────────────────────────────
step "1/9" "Checking Python 3.11+…"

_PYTHON=""

# Helper: returns minor version of a given python binary, or empty if < 3.11
_check_py() {
  local bin="$1"
  command -v "$bin" &>/dev/null || return 1
  local minor
  minor=$("$bin" -c "import sys; print(sys.version_info.minor)" 2>/dev/null) || return 1
  local major
  major=$("$bin" -c "import sys; print(sys.version_info.major)" 2>/dev/null) || return 1
  [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && echo "$bin"
}

# 1. Look for an existing 3.11+ binary (newest first)
for _candidate in python3.13 python3.12 python3.11 python3 python; do
  if _PYTHON=$(_check_py "$_candidate"); then
    break
  fi
  _PYTHON=""
done

# 2. If not found, try to get one automatically
if [ -z "$_PYTHON" ]; then
  _found_ver=$(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "none")
  warn "Python 3.11+ not found (system has: ${_found_ver})."
  echo "  Trying to install Python 3.11 automatically…"
  echo

  # ── 2a. uv (already installed) ───────────────────────────────────────────
  if command -v uv &>/dev/null; then
    echo "  [→] uv found — installing Python 3.11…"
    uv python install 3.11
    _PYTHON=$(uv python find 3.11 2>/dev/null || echo "")

  # ── 2b. pyenv (already installed) ────────────────────────────────────────
  elif command -v pyenv &>/dev/null; then
    echo "  [→] pyenv found — installing Python 3.11…"
    pyenv install -s 3.11
    _PYTHON=$(pyenv prefix 3.11)/bin/python3

  # ── 2c. conda / miniforge / micromamba ───────────────────────────────────
  elif command -v conda &>/dev/null; then
    echo "  [→] conda found — creating bantz env with Python 3.11…"
    conda create -y -n bantz python=3.11 --quiet
    _PYTHON=$(conda run -n bantz which python3)

  # ── 2d. apt (Debian/Ubuntu) ───────────────────────────────────────────────
  elif command -v apt-get &>/dev/null; then
    echo "  [→] apt found — installing python3.11…"
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    _PYTHON=python3.11

  # ── 2e. dnf (Fedora/RHEL) ────────────────────────────────────────────────
  elif command -v dnf &>/dev/null; then
    echo "  [→] dnf found — installing python3.11…"
    sudo dnf install -y python3.11

    _PYTHON=python3.11

  # ── 2f. pacman (Arch) ────────────────────────────────────────────────────
  elif command -v pacman &>/dev/null; then
    echo "  [→] pacman found — installing python311…"
    sudo pacman -S --noconfirm python311
    _PYTHON=python3.11

  # ── 2g. brew (macOS / Linuxbrew) ─────────────────────────────────────────
  elif command -v brew &>/dev/null; then
    echo "  [→] brew found — installing python@3.11…"
    brew install python@3.11
    _PYTHON=$(brew --prefix python@3.11)/bin/python3.11

  # ── 2h. Last resort: install uv and use it ───────────────────────────────
  else
    echo "  [→] No package manager found — installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    uv python install 3.11
    _PYTHON=$(uv python find 3.11 2>/dev/null || echo "")
  fi

  # Verify we actually got something usable
  if [ -z "$_PYTHON" ] || ! _check_py "$_PYTHON" &>/dev/null; then
    die "Could not install Python 3.11 automatically.\nPlease install it manually: https://python.org/downloads"
  fi
fi

_PY_VER=$("$_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
ok "Python ${_PY_VER} (${_PYTHON})"

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

# ── Step 5: Virtual env + install ────────────────────────────────────────────
VENV_DIR="${HOME}/.local/share/bantz/venv"
step "5/9" "Setting up virtual environment and installing Bantz…"

# Create or reuse venv with the Python we found
if [ ! -f "${VENV_DIR}/bin/python" ]; then
  echo "  Creating venv at ${VENV_DIR}…"
  "$_PYTHON" -m venv "$VENV_DIR"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet -e "${INSTALL_DIR}[dev]"
ok "Bantz installed in isolated venv"

# ── Step 6: PATH ──────────────────────────────────────────────────────────────
LOCAL_BIN="${HOME}/.local/bin"
BANTZ_BIN="${VENV_DIR}/bin/bantz"
step "6/9" "Wiring bantz command to PATH…"
mkdir -p "$LOCAL_BIN"

# Symlink venv binary → ~/.local/bin/bantz so it's on PATH anywhere
ln -sf "$BANTZ_BIN" "${LOCAL_BIN}/bantz"

EXPORT_LINE='export PATH="${HOME}/.local/bin:${PATH}"'
for RC in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
  if [ -f "$RC" ] && ! grep -q ".local/bin" "$RC"; then
    { echo ""; echo "# Added by Bantz installer"; echo "$EXPORT_LINE"; } >> "$RC"
    echo "  Updated: $RC"
  fi
done
export PATH="${LOCAL_BIN}:${PATH}"
ok "bantz → ${LOCAL_BIN}/bantz"

# ── Step 7: Setup wizard ──────────────────────────────────────────────────────
step "7/9" "${BLD}Setup Wizard${RST} — press Enter to accept defaults shown in [brackets]."
echo

# _ask VAR PROMPT DEFAULT
# No subshell — printf goes straight to terminal, read from /dev/tty
_ask() {
  local _var="$1" _prompt="$2" _default="$3" _reply
  printf "  %b%s%b [%b%s%b]: " "$BLD" "$_prompt" "$RST" "$CYN" "$_default" "$RST"
  IFS= read -r _reply < /dev/tty
  printf -v "$_var" '%s' "${_reply:-$_default}"
}

_ask_yn() {
  local _var="$1" _prompt="$2" _default="$3" _reply
  printf "  %b%s%b (y/n) [%b%s%b]: " "$BLD" "$_prompt" "$RST" "$CYN" "$_default" "$RST"
  IFS= read -r _reply < /dev/tty
  printf -v "$_var" '%s' "${_reply:-$_default}"
}

_ask OLLAMA_MODEL "Ollama model for Bantz"        "llama3.1:8b"
_ask LANGUAGE     "Preferred language (tr/en)"    "tr"

GEMINI_ENABLED="false"; GEMINI_KEY=""
_ask_yn _GEMINI "Enable Gemini as a fallback LLM?" "n"
if [[ "$_GEMINI" =~ ^[Yy] ]]; then
  GEMINI_ENABLED="true"
  _ask GEMINI_KEY "Gemini API key" ""
fi

PICO_KEY=""; WAKE_WORD="false"
_ask_yn _PICO "Do you have a Porcupine key for wake word detection?" "n"
if [[ "$_PICO" =~ ^[Yy] ]]; then
  WAKE_WORD="true"
  _ask PICO_KEY "Porcupine access key" ""
fi

TG_TOKEN=""; TG_USERS=""
_ask_yn _TG "Set up Telegram remote access?" "n"
if [[ "$_TG" =~ ^[Yy] ]]; then
  _ask TG_TOKEN "Telegram bot token"                                 ""
  _ask TG_USERS "Whitelisted Telegram user ID(s) (comma-separated)" ""
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
_set_key "BANTZ_OLLAMA_MODEL"      "$OLLAMA_MODEL"   "$ENV_FILE"
_set_key "BANTZ_LANGUAGE"          "$LANGUAGE"       "$ENV_FILE"
_set_key "BANTZ_GEMINI_ENABLED"    "$GEMINI_ENABLED" "$ENV_FILE"
[ -n "$GEMINI_KEY" ] && _set_key "BANTZ_GEMINI_API_KEY"       "$GEMINI_KEY" "$ENV_FILE"
_set_key "BANTZ_WAKE_WORD_ENABLED" "$WAKE_WORD"      "$ENV_FILE"
[ -n "$PICO_KEY"   ] && _set_key "BANTZ_PICOVOICE_ACCESS_KEY" "$PICO_KEY"   "$ENV_FILE"
[ -n "$TG_TOKEN"   ] && _set_key "TELEGRAM_BOT_TOKEN"         "$TG_TOKEN"   "$ENV_FILE"
[ -n "$TG_USERS"   ] && _set_key "TELEGRAM_ALLOWED_USERS"     "$TG_USERS"   "$ENV_FILE"

# Symlink into source dir so bantz finds it from its own working directory too
[ ! -e "${INSTALL_DIR}/.env" ] && ln -s "$ENV_FILE" "${INSTALL_DIR}/.env"
ok ".env written"

# ── Step 9: Doctor ────────────────────────────────────────────────────────────
step "9/9" "Running bantz --doctor…"
echo
"${LOCAL_BIN}/bantz" --doctor 2>&1 || true

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo -e "${GRN}${BLD}═══════════════════════════════════════════════════${RST}"
echo -e "${GRN}${BLD}  Bantz is installed and ready.${RST}"
echo -e "${GRN}${BLD}═══════════════════════════════════════════════════${RST}"
echo
echo -e "  Run ${BLD}bantz --ui${RST}       to open the desktop Operations Center."
echo -e "  Run ${BLD}bantz${RST}           to start the terminal interface."
echo -e "  Run ${BLD}bantz --daemon${RST}  for headless background mode."
echo -e "  Run ${BLD}bantz --doctor${RST}  to check your configuration at any time."
echo
