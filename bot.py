"""
bot.py - Step 4: the Telegram front door for the Digital Twin.

Receives Telegram messages, sends them to the AI brain (brain.py), and sends
the twin's reply back. Uses asynchronous polling, so no server, domain, or
open port is needed.

Who may chat (set in .env):
    ALLOWED_USER_ID=123456789                  <- you, the owner
    FRIEND_USERS=987654321:Santos,555444333:Reyes   <- optional friends (id:LastName)
Everyone else is silently ignored. Only the owner gets Google Calendar and Tasks.

Run (from the project folder, venv active):
    python bot.py
Stop with Ctrl+C.
"""

import logging
import os
import sys

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from brain import USER_LAST_NAME, Brain, BrainError
from heartbeat import start_heartbeat

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
# httpx logs full request URLs at INFO level, and those URLs contain your bot
# token. Keep them out of the terminal and out of any log files.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("twin.bot")

# ---------------------------------------------------------------------------
# 2. Configuration: who is allowed to talk to the twin
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_MESSAGE_LIMIT = 4000  # Telegram's hard limit is 4096 characters


def _read_owner_id():
    """Your Telegram user ID from .env (ALLOWED_USER_ID). None until you set it."""
    raw = os.getenv("ALLOWED_USER_ID", "").strip()
    return int(raw) if raw.isdigit() else None


def _read_friends() -> dict:
    """Friends from .env: FRIEND_USERS=123:Santos,456:Reyes  ->  {123: "Santos", ...}"""
    friends = {}
    for entry in os.getenv("FRIEND_USERS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        user_id, _, last_name = entry.partition(":")
        if user_id.strip().isdigit() and last_name.strip():
            friends[int(user_id)] = last_name.strip()
        else:
            log.warning("Skipping bad FRIEND_USERS entry %r (expected id:LastName)", entry)
    return friends


OWNER_ID = _read_owner_id()
FRIENDS = _read_friends()
# user id -> the last name the twin uses for that person
ALLOWED_USERS = ({OWNER_ID: USER_LAST_NAME, **FRIENDS}) if OWNER_ID else {}


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
async def check_access(update: Update):
    """
    Return the caller's last name if they may chat with the twin, else None.
    Strangers are silently ignored (only logged).
    """
    if OWNER_ID is None:
        await update.message.reply_text(
            "Um, I'm not set up yet. Send /id, put that number in your .env file "
            "as ALLOWED_USER_ID, then restart me."
        )
        return None

    user = update.effective_user
    last_name = ALLOWED_USERS.get(user.id) if user else None
    if last_name is None:
        log.warning("Ignored a message from user id %s", user.id if user else None)
    return last_name


def make_request(pool_size: int) -> HTTPXRequest:
    """
    HTTP client for talking to Telegram. The library's default is a 5-second
    timeout, which is too short on slow or flaky connections, so we allow 30.

    Optional: put FORCE_IPV4=1 in .env if connecting hangs while other sites work.
    Some networks have broken IPv6, and this makes the bot use IPv4 only.
    """
    kwargs = {}
    if os.getenv("FORCE_IPV4") == "1":
        kwargs["httpx_kwargs"] = {"transport": httpx.AsyncHTTPTransport(local_address="0.0.0.0")}
    return HTTPXRequest(
        connection_pool_size=pool_size,
        connect_timeout=30,
        read_timeout=30,
        write_timeout=30,
        pool_timeout=30,
        **kwargs,
    )


def load_google_tools():
    """Calendar/Tasks for the owner. Returns None (and says why) if not set up yet."""
    try:
        from google_auth import ensure_token_file
        from google_tools import GoogleTools
    except ImportError as exc:
        log.warning(
            "Calendar/Tasks are off (missing library: %s). Run: "
            "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2",
            exc,
        )
        return None
    if not ensure_token_file():
        log.warning(
            "Calendar/Tasks are off: no Google login found. Run `python google_auth.py` "
            "once on your computer. In the cloud, add token.json as a secret file "
            "(or set GOOGLE_TOKEN_B64)."
        )
        return None
    return GoogleTools()


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT):
    """Split a long reply into chunks that fit in one Telegram message."""
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        yield text[:cut].rstrip()
        text = text[cut:].lstrip()
    if text:
        yield text


# ---------------------------------------------------------------------------
# 4. Handlers
# ---------------------------------------------------------------------------
async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/id - show your Telegram user ID. Works for anyone: it's how a friend
    finds the number to send you so you can add them to .env."""
    await update.message.reply_text(f"Your Telegram user ID is {update.effective_user.id}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start - a short, static greeting (no Gemini call needed)."""
    name = await check_access(update)
    if not name:
        return
    await update.message.reply_text(
        f"Oh, hi, {name}. Um, I'm your digital twin. You can text me about "
        "anything: exams, circuits, whatever. I'm a little shy, but I promise I'm friendly."
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reset - forget the current conversation and start fresh."""
    name = await check_access(update)
    if not name:
        return
    context.chat_data["history"] = []
    await update.message.reply_text(f"Okay, fresh start, {name}. Hi again!")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Any normal text message: ask the brain, send back the reply."""
    name = await check_access(update)
    if not name:
        return

    brain: Brain = context.application.bot_data["brain"]
    # Short-term memory lives per chat (so each person has their own), in memory
    # only: it's cleared when the bot restarts.
    history = context.chat_data.setdefault("history", [])

    await update.effective_chat.send_action(ChatAction.TYPING)

    # Calendar and Tasks belong to the owner only, never to friends.
    is_owner = update.effective_user.id == OWNER_ID
    tools = context.application.bot_data.get("google") if is_owner else None

    try:
        reply, model_used = await brain.reply(
            history, update.message.text, last_name=name, tools=tools
        )
    except BrainError as exc:
        log.error("Brain failed: %s", exc)
        if exc.actions_done:
            await update.message.reply_text(
                f"Um, sorry {name}, I started on that but my brain lagged before I could "
                "confirm it. Could you check your calendar or tasks before asking me again?"
            )
        else:
            await update.message.reply_text(
                f"Um, sorry {name}, my brain's lagging right now. "
                "Could you try again in a minute?"
            )
        return

    log.info("Replied to user %s using %s", update.effective_user.id, model_used)
    for chunk in split_message(reply):
        await update.message.reply_text(chunk)  # plain text: no formatting surprises


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch anything unexpected so one bad message can't crash the bot."""
    log.error("Unhandled error", exc_info=context.error)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        sys.exit("Missing TELEGRAM_BOT_TOKEN in your .env file.")
    try:
        brain = Brain()
    except RuntimeError as exc:
        sys.exit(str(exc))

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(make_request(pool_size=8))
        .get_updates_request(make_request(pool_size=1))
        .build()
    )
    app.bot_data["brain"] = brain
    app.bot_data["google"] = load_google_tools()

    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)

    if OWNER_ID is None:
        log.warning("ALLOWED_USER_ID isn't set yet. Send /id to the bot to get your ID.")
    else:
        log.info("Allowed users: you + %d friend(s)", len(FRIENDS))
    log.info("Calendar/Tasks: %s", "on (owner only)" if app.bot_data["google"] else "off")
    log.info("Twin is online. Press Ctrl+C to stop.")
    # In the cloud, the host sets PORT and expects something listening on it.
    # Locally PORT isn't set, so this does nothing.
    port = os.getenv("PORT", "")
    if port.isdigit():
        start_heartbeat(int(port))

    # Optional: DROP_PENDING_UPDATES=1 ignores messages that arrived while the bot was
    # asleep or offline (for example overnight). By default the bot answers them when it
    # wakes up, because Telegram holds undelivered messages for a while.
    drop_pending = os.getenv("DROP_PENDING_UPDATES") == "1"
    if drop_pending:
        log.info("DROP_PENDING_UPDATES is on: messages sent while offline will be ignored.")

    # bootstrap_retries=-1: if Telegram can't be reached at startup (or after a
    # network drop), keep retrying instead of quitting.
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=-1,
        drop_pending_updates=drop_pending,
    )


if __name__ == "__main__":
    main()
