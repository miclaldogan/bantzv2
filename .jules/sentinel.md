## 2024-05-24 - [Insecure File Permissions]
**Vulnerability:** Found a TOCTOU (Time-of-Check to Time-of-Use) vulnerability when creating sensitive files (like `.env` files with API keys). The code used `Path.write_text()` followed immediately by `Path.chmod(0o600)`.
**Learning:** This creates a small window where the file exists with default permissions (like 0o644) before `chmod` applies the correct secure permissions. During this time, another process could potentially read sensitive tokens like `TELEGRAM_BOT_TOKEN`.
**Prevention:** Use `os.open()` with `os.O_CREAT | os.O_WRONLY | os.O_TRUNC` and an explicit `0o600` mode parameter. Then, use `os.fdopen()` to write the content. This securely creates the file with correct permissions from the start.
## 2024-05-24 - [BrowserTool Argument Injection]
**Vulnerability:** Argument injection via f-strings passing user-provided URLs to `shlex.split` before `subprocess.run` in `BrowserTool.fetch` and `BrowserTool.extract_text`. An attacker could inject flags (e.g. `-o /tmp/evil`) into the `curl` or `readability-cli` commands.
**Learning:** `shlex.split(f"cmd {var}")` splits variables containing spaces, causing injected options to be parsed as actual arguments to the binary.
**Prevention:** Avoid formatting unvalidated variables into shell-like strings destined for `shlex.split`. Always use `list[str]` format explicitly for `subprocess.run` or use `shlex.quote()` when forced to use strings.
## 2024-05-24 - [Overly Permissive CORS and Missing CSRF Validation]
**Vulnerability:** The local GPS server allowed wildcard CORS (`Access-Control-Allow-Origin: *`) across all endpoints and lacked `Content-Type` validation on its state-changing `/update` POST endpoint, exposing users to CSRF and SSRF attacks from malicious websites.
**Learning:** Local servers, even those intended only for LAN use, are accessible from any origin in a standard browser unless restricted by CORS. An HTML `<form>` submission bypasses CORS preflight checks, meaning lack of `Content-Type` validation on POST endpoints enables trivial CSRF attacks.
**Prevention:** Never use `Access-Control-Allow-Origin: *` for state-changing local APIs unless strictly necessary. Always enforce `Content-Type: application/json` for JSON APIs to block simple HTML form submissions.
## 2025-04-10 - [TOCTOU Vulnerability in File Creations]
**Vulnerability:** A TOCTOU (Time-of-Check to Time-of-Use) vulnerability was found where `path.chmod(0o600)` was used immediately after writing content to sensitive files.
**Learning:** This leaves a race condition window between file creation and permission changes. A malicious symlink attack could change the target of `path.chmod`, or simply read the sensitive content before permissions are restricted.
**Prevention:** Using `os.fchmod(fd, 0o600)` right after `os.open` tightly binds the permission changes to the file descriptor, rather than relying on a subsequent path-based `chmod` that is vulnerable to symlink race conditions.
## 2025-05-24 - [XXE Vulnerability in Feed Parsers]
**Vulnerability:** Found XXE (XML External Entity) injection vulnerabilities in `src/bantz/tools/news.py` and `src/bantz/tools/feed_tool.py` when parsing untrusted RSS/Atom XML payloads using Python's standard `xml.etree.ElementTree`.
**Learning:** Python's standard `xml` module is vulnerable to XXE attacks (like the billion laughs attack or arbitrary file reads via external entities) by default. Untrusted XML sources, like remote news RSS feeds, should never be parsed with the standard library's `ElementTree.fromstring`.
**Prevention:** Replaced `import xml.etree.ElementTree as ET` with `import defusedxml.ElementTree as ET` to safely parse external XML. Added `defusedxml` to dependencies and ensured `DefusedXmlException` alongside `ParseError` are correctly caught to fail safely when malicious payloads are encountered.
