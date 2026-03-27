## 2024-05-24 - [Insecure File Permissions]
**Vulnerability:** Found a TOCTOU (Time-of-Check to Time-of-Use) vulnerability when creating sensitive files (like `.env` files with API keys). The code used `Path.write_text()` followed immediately by `Path.chmod(0o600)`.
**Learning:** This creates a small window where the file exists with default permissions (like 0o644) before `chmod` applies the correct secure permissions. During this time, another process could potentially read sensitive tokens like `TELEGRAM_BOT_TOKEN`.
**Prevention:** Use `os.open()` with `os.O_CREAT | os.O_WRONLY | os.O_TRUNC` and an explicit `0o600` mode parameter. Then, use `os.fdopen()` to write the content. This securely creates the file with correct permissions from the start.
