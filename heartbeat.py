"""
heartbeat.py - a tiny web server that answers "ok" so a cloud host (and an
uptime pinger such as UptimeRobot) can see that the bot is alive.

Why it's here: some free hosts only count a service as "in use" when it
receives web traffic, and they put idle services to sleep. The bot itself uses
Telegram polling (it only makes outgoing connections), so it never receives
any. Pinging this page every few minutes counts as traffic.

It uses only Python's standard library and runs in a background thread, so it
can't interfere with the bot's own event loop. It answers every GET and HEAD
request with a short "ok" and exposes nothing else (no secrets, no status
details).
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("twin.heartbeat")


class _Handler(BaseHTTPRequestHandler):
    timeout = 10  # drop connections that go quiet instead of hanging forever

    def _reply(self, send_body: bool) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:  # HEAD requests get the headers only
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._reply(send_body=True)

    def do_HEAD(self) -> None:
        self._reply(send_body=False)

    def log_message(self, *args) -> None:
        pass  # don't print a log line for every ping


def start_heartbeat(port: int) -> ThreadingHTTPServer:
    """Start the heartbeat server in a background thread and return it."""
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, name="heartbeat", daemon=True).start()
    log.info("Heartbeat server listening on port %s", port)
    return server
