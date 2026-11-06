"""
test_ai.py - quick terminal check of the AI brain (no Telegram involved).

Usage (from the project folder, venv active):
    python test_ai.py                      # uses the built-in test message
    python test_ai.py "your own message"   # try any message you like
    python test_ai.py --models             # list the models your key can use
"""

import asyncio
import logging
import sys

from google.genai import errors

from brain import MODEL_NAME, Brain, BrainError

DEFAULT_TEST_MESSAGE = "Hey, I have a circuits exam tomorrow and I'm stressed"

# Show retry notices (e.g. "503, retrying...") in the terminal.
logging.basicConfig(level=logging.WARNING, format="  %(message)s")


def show_models(brain: Brain) -> None:
    print("Models your key can use for chat (set one as GEMINI_MODEL in .env):")
    try:
        for name in brain.list_models():
            print("  -", name)
    except errors.APIError as exc:
        print(f"  (couldn't list models: {exc})")


def main() -> None:
    try:
        brain = Brain()
    except RuntimeError as exc:
        sys.exit(str(exc))

    args = sys.argv[1:]
    if "--models" in args:
        show_models(brain)
        return

    user_message = " ".join(args).strip() or DEFAULT_TEST_MESSAGE
    print(f"Model : {MODEL_NAME}")
    print(f"You   : {user_message}\n")

    try:
        reply, used_model = asyncio.run(brain.reply([], user_message))
    except BrainError as exc:
        if exc.code == 404:
            print("That model name wasn't found for your key.\n")
            show_models(brain)
        sys.exit(f"\nGemini failed: {exc}")

    if used_model != MODEL_NAME:
        print(f"(answered by fallback model: {used_model})")
    print(f"Twin  : {reply}")


if __name__ == "__main__":
    main()
