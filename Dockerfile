# ── Bantz v3 — Oracle Free Tier ARM64 (#93) ──────────────────────────────────
# Targets: linux/arm64 (Oracle Free Tier Ampere A1)
# Build:   docker build -t bantz .
# Run:     docker compose up -d

FROM python:3.11-slim

# ── System packages ────────────────────────────────────────────────────────────
# curl: BrowserTool fetch; pup: HTML CSS selector; Node: readability-cli;
# nodejs/npm: readability-cli; portaudio: PyAudio (voice pipeline, optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        nodejs \
        npm \
        gosu \
    && npm install -g @mozilla/readability-cli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── pup (CSS selector tool for BrowserTool.query) ─────────────────────────────
# Pre-built binary — avoids Go toolchain in image
ARG PUP_VERSION=0.4.0
RUN curl -fsSL \
    "https://github.com/ericchiang/pup/releases/download/v${PUP_VERSION}/pup_v${PUP_VERSION}_linux_arm64.zip" \
    -o /tmp/pup.zip \
    && unzip /tmp/pup.zip pup -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/pup \
    && rm /tmp/pup.zip

# ── App user (non-root) ────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash bantz
WORKDIR /home/bantz/app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# ── Application source ─────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Data volume ────────────────────────────────────────────────────────────────
# ~/.local/share/bantz — SQLite DB, logs, reflections
RUN mkdir -p /home/bantz/.local/share/bantz \
    && chown -R bantz:bantz /home/bantz

VOLUME ["/home/bantz/.local/share/bantz"]

# ── Config ─────────────────────────────────────────────────────────────────────
# Mount .env at runtime:  -v $(pwd)/.env:/home/bantz/app/.env:ro
ENV PYTHONUNBUFFERED=1

USER bantz

# ── Default: headless daemon (Telegram bot + scheduler, no TUI) ───────────────
CMD ["python", "-m", "bantz", "--daemon"]
