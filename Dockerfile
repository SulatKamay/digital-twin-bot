# Works on any Docker host (Render, Railway, Fly.io, a VPS, ...).
FROM python:3.12-slim

# No .pyc clutter, logs appear immediately. PORT is where the heartbeat page
# listens (Render replaces it with its own value at runtime).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Run as a normal user instead of root. The /app folder must belong to that user
# too, otherwise the bot can't save its refreshed Google login (token.json) there.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir /app \
    && chown appuser:appuser /app
WORKDIR /app

# Install dependencies first so rebuilds are fast when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the bot's code, by name. That way .env, token.json and
# credentials.json can never end up inside the image, even by accident.
COPY --chown=appuser:appuser bot.py brain.py google_auth.py google_tools.py heartbeat.py ./

USER appuser
EXPOSE 10000
CMD ["python", "bot.py"]
