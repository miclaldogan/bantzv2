# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Bantz, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **GitHub Private Vulnerability Reporting** (preferred):
   Go to the [Security Advisories](https://github.com/miclaldogan/bantzv2/security/advisories) page and click "Report a vulnerability".

2. **Email**: Send details to **security@bantz.dev** (or open a private advisory on GitHub).

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix release**: Within 2 weeks for critical issues

### Scope

The following are in scope:
- Authentication/authorization bypasses (Google OAuth, Telegram bot whitelist)
- Shell command injection via tool execution
- SQLite injection in memory/habits queries
- Credential exposure (API keys, OAuth tokens)
- Path traversal in filesystem operations
- Dependency vulnerabilities

### Out of Scope

- Issues in third-party services (Ollama, Google APIs)
- Denial of service on local-only components
- Issues requiring physical access to the machine

## Security Best Practices for Users

- Keep `.env` file permissions restricted (`chmod 600 .env`)
- Use `TELEGRAM_ALLOWED_USERS` to whitelist Telegram access
- Review shell commands before confirming destructive operations
- Keep dependencies updated (`pip install --upgrade bantz`)
- Store OAuth tokens securely (default: `~/.local/share/bantz/`)
