"""
Bantz v2 — Token Store
Multi-account Google credential manager.

Each service has its own token file, potentially different Google accounts:
  gmail_token.json      ← personal Gmail
  classroom_token.json  ← school/work account
  calendar_token.json   ← shares Gmail by default

Usage:
    from bantz.auth.token_store import token_store
    creds = token_store.get("gmail")       # raises TokenNotFoundError if missing
    creds = token_store.get_or_none("gmail")  # returns None if missing
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TokenNotFoundError(Exception):
    """Raised when a service token is missing. Message includes setup instructions."""
    pass


# Which services share tokens — currently each service gets its own token
_TOKEN_ALIASES: dict[str, str] = {}

# Required OAuth scopes per service
SERVICE_SCOPES: dict[str, list[str]] = {
    "gmail": [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ],
    "classroom": [
        "https://www.googleapis.com/auth/classroom.courses.readonly",
        "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
        "https://www.googleapis.com/auth/classroom.announcements.readonly",
    ],
    "calendar": [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ],
}


class TokenStore:
    def __init__(self) -> None:
        self._dir = Path.home() / ".local" / "share" / "bantz" / "tokens"
        self._dir.mkdir(parents=True, exist_ok=True)

    def token_path(self, service: str) -> Path:
        # Resolve alias (calendar → gmail)
        resolved = _TOKEN_ALIASES.get(service, service)
        return self._dir / f"{resolved}_token.json"

    def credentials_path(self) -> Path:
        """Path to Google Cloud credentials.json (user provides once)."""
        return self._dir / "credentials.json"

    def get(self, service: str):
        """
        Return valid Credentials for service.
        Auto-refreshes if expired.
        Raises TokenNotFoundError if token missing.
        """
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        path = self.token_path(service)

        if not path.exists():
            raise TokenNotFoundError(
                f"No token found for '{service}'.\n"
                f"Run: bantz --setup google {service}"
            )

        creds = Credentials.from_authorized_user_file(str(path))

        # Auto-refresh if expired
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save(service, creds)
                logger.info(f"Token refreshed for {service}")
            except Exception as exc:
                raise TokenNotFoundError(
                    f"Token for '{service}' expired and refresh failed: {exc}\n"
                    f"Run: bantz --setup google {service}"
                )

        return creds

    def get_or_none(self, service: str):
        """Return credentials or None (no exception)."""
        try:
            return self.get(service)
        except (TokenNotFoundError, Exception):
            return None

    def save(self, service: str, creds) -> None:
        """Save credentials to token file with secure permissions."""
        self._save(service, creds)

    def _save(self, service: str, creds) -> None:
        resolved = _TOKEN_ALIASES.get(service, service)
        path = self._dir / f"{resolved}_token.json"
        path.write_text(creds.to_json())
        # Secure: owner read/write only
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        logger.info(f"Token saved: {path}")

    def is_configured(self, service: str) -> bool:
        return self.token_path(service).exists()

    def has_credentials_json(self) -> bool:
        return self.credentials_path().exists()

    def status(self) -> dict[str, str]:
        """Return setup status for all services."""
        result = {}
        for svc in ("gmail", "classroom", "calendar"):
            path = self.token_path(svc)
            if not path.exists():
                result[svc] = "not configured"
            else:
                try:
                    self.get(svc)
                    result[svc] = "ok"
                except TokenNotFoundError as e:
                    result[svc] = f"expired ({e})"
                except Exception:
                    result[svc] = "invalid token"
        return result


token_store = TokenStore()