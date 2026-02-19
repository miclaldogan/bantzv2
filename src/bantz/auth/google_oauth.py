"""
Bantz v2 — Google OAuth2 Flow
Single-command browser-based setup.

Usage:
    bantz --setup google gmail
    bantz --setup google classroom   ← can be different Google account

Flow:
    1. Check credentials.json exists
    2. Open browser to Google consent page
    3. Local server catches callback on localhost:8765
    4. Save token to token_store
"""
from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

from bantz.auth.token_store import token_store, SERVICE_SCOPES

logger = logging.getLogger(__name__)

REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"


def setup_google(service: str) -> bool:
    """
    Run OAuth2 flow for a Google service.
    Opens browser, waits for callback, saves token.
    Returns True on success.
    """
    service = service.lower().strip()

    if service not in SERVICE_SCOPES:
        print(f"❌ Unknown service: '{service}'")
        print(f"   Available: {', '.join(SERVICE_SCOPES.keys())}")
        return False

    # Check credentials.json
    creds_path = token_store.credentials_path()
    if not creds_path.exists():
        _print_credentials_help(creds_path)
        return False

    scopes = SERVICE_SCOPES[service]

    print(f"\n🔑 Setting up Google {service.title()}...")
    if service == "classroom":
        print("   ℹ️  You can use a different Google account than Gmail.")
    print(f"   Scopes: {', '.join(s.split('/')[-1] for s in scopes)}")
    print()

    try:
        import os
        import wsgiref.simple_server
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

        from google_auth_oauthlib.flow import InstalledAppFlow

        # google_auth_oauthlib explicitly sets allow_reuse_address=False
        # before creating the server. Patch make_server to force SO_REUSEADDR
        # so back-to-back setups don't fail on TIME_WAIT sockets.
        _orig_make = wsgiref.simple_server.make_server

        def _make_reuse(*a, **kw):
            wsgiref.simple_server.WSGIServer.allow_reuse_address = True
            return _orig_make(*a, **kw)

        wsgiref.simple_server.make_server = _make_reuse

        flow = InstalledAppFlow.from_client_secrets_file(
            str(creds_path),
            scopes=scopes,
        )

        # Run local server flow — opens browser automatically
        print(f"   Opening browser... (listening on localhost:{REDIRECT_PORT})")
        creds = flow.run_local_server(
            port=REDIRECT_PORT,
            prompt="consent",          # always show consent to allow account switching
            access_type="offline",     # get refresh_token
            open_browser=True,
        )

        # Restore original
        wsgiref.simple_server.make_server = _orig_make

        token_store.save(service, creds)
        print(f"\n✅ {service.title()} connected successfully!")
        print(f"   Token saved to: {token_store.token_path(service)}")
        return True

    except Exception as exc:
        print(f"\n❌ OAuth failed: {exc}")
        logger.exception(f"OAuth flow failed for {service}")
        return False


def _print_credentials_help(path: Path) -> None:
    print(f"\n❌ credentials.json not found at: {path}")
    print()
    print("To get credentials.json:")
    print("  1. Go to https://console.cloud.google.com/")
    print("  2. Create a project (or select existing)")
    print("  3. Enable APIs:")
    print("       Gmail API, Google Calendar API, Google Classroom API")
    print("  4. Go to 'APIs & Services' → 'Credentials'")
    print("  5. Create 'OAuth 2.0 Client ID' → Desktop app")
    print("  6. Download JSON → rename to 'credentials.json'")
    print(f"  7. Move to: {path}")
    print()
    print("Then run: bantz --setup google gmail")