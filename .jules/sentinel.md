## 2024-05-24 - [Insecure File Permissions]
**Vulnerability:** Found a TOCTOU (Time-of-Check to Time-of-Use) vulnerability when creating sensitive files (like `.env` files with API keys). The code used `Path.write_text()` followed immediately by `Path.chmod(0o600)`.
**Learning:** This creates a small window where the file exists with default permissions (like 0o644) before `chmod` applies the correct secure permissions. During this time, another process could potentially read sensitive tokens like `TELEGRAM_BOT_TOKEN`.
**Prevention:** Use `os.open()` with `os.O_CREAT | os.O_WRONLY | os.O_TRUNC` and an explicit `0o600` mode parameter. Then, use `os.fdopen()` to write the content. This securely creates the file with correct permissions from the start.
## 2024-05-24 - [BrowserTool Argument Injection]
**Vulnerability:** Argument injection via f-strings passing user-provided URLs to `shlex.split` before `subprocess.run` in `BrowserTool.fetch` and `BrowserTool.extract_text`. An attacker could inject flags (e.g. `-o /tmp/evil`) into the `curl` or `readability-cli` commands.
**Learning:** `shlex.split(f"cmd {var}")` splits variables containing spaces, causing injected options to be parsed as actual arguments to the binary.
**Prevention:** Avoid formatting unvalidated variables into shell-like strings destined for `shlex.split`. Always use `list[str]` format explicitly for `subprocess.run` or use `shlex.quote()` when forced to use strings.
## 2026-04-11 - [GPS Server CSRF/SSRF Mitigation]
**Vulnerability:** The local GPS server's `/update` POST endpoint accepted arbitrary Content-Types (allowing simple POST requests to bypass some CSRF protections) and indiscriminately responded with wildcard CORS headers (`Access-Control-Allow-Origin: *`).
**Learning:** Returning wildcard CORS headers on local servers allows any website to read responses (SSRF risk). Accepting non-JSON content-types for JSON endpoints allows simple cross-site POSTs to execute state-changing actions.
**Prevention:** Strictly validate `Content-Type: application/json` on endpoints expecting JSON payloads to enforce pre-flight requests from browsers. Avoid returning `Access-Control-Allow-Origin: *` on local servers.
