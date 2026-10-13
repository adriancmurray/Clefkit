"""Threaded HTTP front end for one SystemOne engine. Standard library only."""

from __future__ import annotations

import ipaddress
import json
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from clefkit_server.engines import Engine

DEFAULT_PORT = 8790
MAX_BODY_BYTES = 4 * 1024 * 1024
SYSTEMONE_PATH = "/v1/systemone"


def is_loopback(host: str) -> bool:
    """Return True when ``host`` is ``localhost`` or a loopback IP address."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_loopback(host: str, allow_remote: bool) -> None:
    """Raise ValueError when ``host`` is not loopback and ``allow_remote`` is off."""
    if not allow_remote and not is_loopback(host):
        raise ValueError(
            f"refusing to bind {host!r}: not a loopback address (pass --allow-remote to expose it)"
        )


class _ClefServer(ThreadingHTTPServer):
    """Threaded server that holds the engine and the lock serialising decisions."""

    def __init__(self, address: tuple[str, int], engine: Engine) -> None:
        # Only IPv6 literals contain ':', and they need an AF_INET6 socket.
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)
        self.engine = engine
        # The model is not thread-safe, so one decision runs at a time; /health does not take this lock.
        self.decide_lock = threading.Lock()


def make_server(engine: Engine, host: str, port: int) -> ThreadingHTTPServer:
    """Return a bound server for ``engine`` that is not yet serving; port 0 picks a free port."""
    return _ClefServer((host, port), engine)


class _Handler(BaseHTTPRequestHandler):
    """Serves GET /health and POST /v1/systemone; every other path is 404."""

    server: _ClefServer
    timeout = 30  # seconds to wait for a slow client's body

    def do_GET(self) -> None:
        """Report the engine's backend, model and load state."""
        if urlsplit(self.path).path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        engine = self.server.engine
        self._send_json(
            200,
            {"ok": True, "backend": engine.name, "model": engine.model_id, "loaded": engine.loaded},
        )

    def do_POST(self) -> None:
        """Answer one SystemOne decision."""
        if urlsplit(self.path).path != SYSTEMONE_PATH:
            self._send_json(404, {"error": "not found"})
            return
        request = self._read_body()
        if request is None:
            return
        try:
            with self.server.decide_lock:
                result = self.server.engine.decide(request)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"error": "internal error"})
        else:
            self._send_json(200, result)

    def _read_body(self) -> dict[str, Any] | None:
        """Parse the JSON object body; send 400 or 413 and return None when it is unusable."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        # Reject on the declared length, before reading, so an oversize body is never buffered.
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes"})
            return None
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        # Deeply nested input exhausts the parser's recursion limit, which is a client error.
        except (ValueError, RecursionError) as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return None
        if not isinstance(body, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return None
        return body

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        """Write one JSON response with its status and length."""
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
