## 2024-05-24 - [Insecure File Permissions]
**Vulnerability:** Found a TOCTOU (Time-of-Check to Time-of-Use) vulnerability when creating sensitive files (like `.env` files with API keys). The code used `Path.write_text()` followed immediately by `Path.chmod(0o600)`.
**Learning:** This creates a small window where the file exists with default permissions (like 0o644) before `chmod` applies the correct secure permissions. During this time, another process could potentially read sensitive tokens like `TELEGRAM_BOT_TOKEN`.
**Prevention:** Use `os.open()` with `os.O_CREAT | os.O_WRONLY | os.O_TRUNC` and an explicit `0o600` mode parameter. Then, use `os.fdopen()` to write the content. This securely creates the file with correct permissions from the start.
## 2024-05-24 - [Token File Permission TOCTOU]
**Vulnerability:** In `token_store.py`, `os.open(..., os.O_CREAT | ..., 0o600)` does not change permissions if the token file already exists. If an existing token file had loose permissions, subsequent saves wouldn't secure it.
**Learning:** Python's `os.open` with a mode argument only applies when creating a new file. Relying on it to secure an existing file is a TOCTOU flaw.
**Prevention:** Always append a `Path.chmod()` call after `os.open` when enforcing strict file permissions to ensure existing files get their permissions corrected.
