"""
google_auth.py - Google login for Calendar and Tasks.

Run this ONCE by hand:
    python google_auth.py
It opens your browser, you approve access, and it saves token.json next to
this file. After that, bot.py uses token.json automatically (no browser) and
refreshes it by itself. If Google ever logs you out (for example after 7 days
while the app is in "Testing" mode), just run this script again.

Files (keep both PRIVATE, never upload them anywhere):
    credentials.json  - downloaded from Google Cloud Console (OAuth client)
    token.json        - created by this script (your saved login)

In the cloud there is no browser to log in with, so the finished token.json is
supplied as a secret instead (see ensure_token_file below).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"
# Where Render mounts a "Secret File" named token.json (read-only).
SECRET_FILE_TOKEN = Path("/etc/secrets/token.json")

# Least privilege: events only (not settings or other calendars' sharing),
# plus Tasks. If you change these, delete token.json and log in again.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
]


class GoogleAuthError(Exception):
    """Google isn't connected, or the saved login no longer works."""


def _write_private(text: str) -> None:
    """Write token.json readable only by you (chmod 600)."""
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _save(creds: Credentials) -> None:
    _write_private(creds.to_json())


def ensure_token_file() -> bool:
    """
    True if a saved Google login (token.json) is available.

    On your own computer it's simply the file next to this script. In the cloud
    the finished token.json is supplied as a secret, in one of two ways:
      1. a secret FILE named token.json (Render mounts it at /etc/secrets/token.json), or
      2. an environment variable GOOGLE_TOKEN_B64 holding the base64 text of token.json.
    It's copied to a normal writable file because Google's library refreshes
    the login and saves the update. (credentials.json isn't needed in the cloud:
    token.json already contains everything required to refresh itself.)
    """
    if TOKEN_FILE.exists():
        return True

    try:
        if SECRET_FILE_TOKEN.exists():
            text = SECRET_FILE_TOKEN.read_text()
        elif os.getenv("GOOGLE_TOKEN_B64"):
            text = base64.b64decode(os.environ["GOOGLE_TOKEN_B64"]).decode()
        else:
            return False
        json.loads(text)  # make sure it's really JSON before saving it
    except (OSError, ValueError) as exc:  # includes bad base64 and bad JSON
        print(f"The Google token secret is unreadable ({type(exc).__name__}). "
              "Re-create it from a fresh token.json.", file=sys.stderr)
        return False

    _write_private(text)
    return True


def get_credentials() -> Credentials:
    """Load the saved login, refreshing it if needed. Never opens a browser."""
    if not ensure_token_file():
        raise GoogleAuthError("Google isn't connected yet. Run: python google_auth.py")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.valid:
        return creds

    if creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise GoogleAuthError(
                "The Google login expired. Run: python google_auth.py"
            ) from exc
        _save(creds)
        return creds

    raise GoogleAuthError("The Google login is incomplete. Run: python google_auth.py")


def _login() -> Credentials:
    """Open the browser consent screen and save the result."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CREDENTIALS_FILE.exists():
        sys.exit(
            f"Can't find {CREDENTIALS_FILE.name} in {BASE_DIR}.\n"
            "Download it from Google Cloud Console (APIs & Services > Credentials > "
            "your OAuth client > Download JSON), rename it to credentials.json, "
            "and put it in this folder."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    # prompt="consent" makes Google hand out a long-lived refresh token.
    creds = flow.run_local_server(port=0, prompt="consent")
    _save(creds)
    return creds


def _self_test(creds: Credentials) -> None:
    """Quick proof that both APIs answer."""
    from googleapiclient.discovery import build

    calendar = build("calendar", "v3", credentials=creds, cache_discovery=False)
    tz = calendar.events().list(calendarId="primary", maxResults=1).execute().get("timeZone")
    print(f"Calendar OK  (your calendar's time zone: {tz})")

    tasks = build("tasks", "v1", credentials=creds, cache_discovery=False)
    lists = tasks.tasklists().list(maxResults=5).execute().get("items", [])
    names = ", ".join(item["title"] for item in lists) or "none yet"
    print(f"Tasks OK     (task lists: {names})")


if __name__ == "__main__":
    login_creds = _login()
    print(f"\nSaved login to {TOKEN_FILE.name}.")
    _self_test(login_creds)
    print("\nAll set. You can start the bot with: python bot.py")
