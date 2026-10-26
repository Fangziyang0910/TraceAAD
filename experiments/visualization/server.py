"""Serve the training dashboard and its JSON views."""

import argparse
import gzip
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from .data import DEFAULT_RESULTS_ROOT, ResultsMonitor

HTML_FILE = Path(__file__).with_name("static") / "index.html"


def _json_bytes(payload):
    return None if payload is None else json.dumps(
        payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


class ResponseCache:
    """Serialized-JSON cache keyed by request, validated by a cheap signature.

    Concurrent requests for the same key share one build instead of each
    re-parsing the same journals.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[Any, tuple[Any, bytes]] = {}
        self._key_locks: dict[Any, Lock] = {}

    def get_or_build(self, key, signature, builder):
        with self._lock:
            key_lock = self._key_locks.setdefault(key, Lock())
        with key_lock:
            with self._lock:
                hit = self._store.get(key)
                if hit is not None and hit[0] == signature:
                    return hit[1], True
            body = builder()
            with self._lock:
                self._store[key] = (signature, body)
            return body, False


def make_request_handler(monitor: ResultsMonitor) -> type[BaseHTTPRequestHandler]:
    cache = ResponseCache()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path in {"/", "/index.html"}:
                return self._serve_asset(HTML_FILE, "text/html")
            if parsed.path in {"/style.css", "/app.js"}:
                content_type = "text/css" if parsed.path.endswith(".css") else "text/javascript"
                return self._serve_asset(HTML_FILE.with_name(parsed.path[1:]), content_type)
            if parsed.path == "/api/batches":
                return self._send_json({"batches": monitor.batches(), "default_batch": monitor.default_batch()})
            if parsed.path == "/api/state":
                batch = params.get("batch", [None])[0]
                body, _ = cache.get_or_build(
                    ("state", batch), monitor.state_signature(batch),
                    lambda: _json_bytes(monitor.overview(batch)))
                return self._send_body(body)
            if parsed.path == "/api/cohorts":
                return self._send_json({"cohorts": monitor.cohorts()})
            if parsed.path == "/api/compare":
                cohorts = list(dict.fromkeys(c for c in params.get("cohorts", [""])[0].split(",") if c))
                signature = monitor.compare_signature(cohorts)
                body, _ = cache.get_or_build(
                    ("compare", tuple(cohorts)), signature,
                    lambda: _json_bytes(monitor.compare(cohorts)))
                return self._send_body(body)
            if parsed.path == "/api/run":
                batch = params.get("batch", [""])[0]
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                signature = monitor.run_signature(batch, task, name)
                body, _ = cache.get_or_build(
                    ("run", batch, task, name), signature,
                    lambda: _json_bytes(monitor.run_detail(batch, task, name)))
                if body is None:
                    return self.send_error(404, "Run not found")
                return self._send_body(body)
            self.send_error(404, "Endpoint not found")

        def _serve_asset(self, path, content_type) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                return self.send_error(404, "Page not found")
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            self._send_body(_json_bytes(payload))

        def _send_body(self, body: bytes) -> None:
            headers = {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"}
            payload = body
            if len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                payload = gzip.compress(body, 5)
                headers["Content-Encoding"] = "gzip"
            self.send_response(200)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment results monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--batch", default=None)
    args = parser.parse_args()

    monitor = ResultsMonitor(args.results_dir, args.experiment, args.batch)
    server = ThreadingHTTPServer((args.host, args.port), make_request_handler(monitor))
    selected = monitor.overview().get("batch") or "无实验"
    print(f"Experiment monitor: http://127.0.0.1:{args.port} ({selected})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
