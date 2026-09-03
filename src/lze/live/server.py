"""Live prediction HTTP server for the ground station.

Serves the dashboard and streams predictions over Server-Sent Events (SSE). A
background worker pulls telemetry from any :class:`TelemetrySource`, runs the
:class:`LandingPredictor`, and fans each prediction out to connected browsers.

Deliberately stdlib-only (``http.server``) so it runs on a bare Raspberry Pi
with no web framework and no internet.
"""
from __future__ import annotations

import json
import queue
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import REPO_ROOT
from ..telemetry.schema import TelemetryPacket
from .predictor import LandingPredictor, LivePrediction

DASHBOARD_HTML = REPO_ROOT / "dashboard" / "index.html"


class _Hub:
    """Fan-out of prediction JSON to subscriber queues.

    Keeps the full flight history and replays it to every new subscriber, so a
    browser that connects (or refreshes) mid-flight immediately gets the whole
    track and profile instead of a stream that starts in mid-air. A flight at
    1 Hz is a few hundred frames, so the buffer is small.
    """

    def __init__(self, history_len: int = 4096) -> None:
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=history_len)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=8192)
        with self._lock:
            for payload in self._history:
                q.put_nowait(payload)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, payload: str) -> None:
        with self._lock:
            self._history.append(payload)
            for q in list(self._subs):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass


def _make_handler(hub: _Hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default logging
            pass

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML.read_text())
            elif self.path == "/events":
                self._send_events()
            else:
                self.send_error(404)

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    payload = q.get()
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


class LiveServer:
    """Runs the predictor over a telemetry source and serves the dashboard."""

    def __init__(
        self,
        predictor: LandingPredictor,
        source: Iterable[TelemetryPacket],
        host: str = "127.0.0.1",
        port: int = 8000,
        truth: Optional[dict] = None,
    ):
        self.predictor = predictor
        self.source = source
        self.host = host
        self.port = port
        self.truth = truth
        self.hub = _Hub()
        self._httpd: Optional[ThreadingHTTPServer] = None

    def _worker(self) -> None:
        for pkt in self.source:
            pred: LivePrediction = self.predictor.process(pkt)
            record = pred.to_dict()
            if self.truth is not None:
                record["truth"] = self.truth
            self.hub.publish(json.dumps(record))

    def serve_forever(self) -> None:
        handler = _make_handler(self.hub)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        url = f"http://{self.host}:{self.port}"
        print(f"Kronos live dashboard: {url}  (Ctrl-C to stop)")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.shutdown()
