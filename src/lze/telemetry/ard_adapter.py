"""Bridge the ARD Ground-Station Dashboard telemetry into the LZE predictor.

The ARD dashboard (a separate repo) streams telemetry to its browser as a
Socket.IO ``telemetry_data`` event carrying a nested *envelope*::

    {
      "type": "telemetry",
      "timestamp": <ms wall clock>,
      "packet":  {"time": <ms since launch>, "altitude": <m AGL>, ...},
      "derived": {"latitude": .., "longitude": .., "velocity": ..,
                   "east_m": .., "north_m": .., "azimuth_deg": ..}
    }

The LZE surrogate, by contrast, consumes :class:`TelemetryPacket` -- GPS
lat/lon, barometric AGL, and a *velocity vector* in the launch-pad ENU frame.
This module converts one into the other so the landing-zone estimator can run
**alongside** the ARD dashboard without any change to the ARD repo: point the
LZE live server at the ARD backend and it serves its own offline recovery map.

Design choices
--------------
* The ARD envelope has no ENU *velocity vector* -- only a scalar speed and a
  horizontal azimuth. We therefore reconstruct ``(ve, vn, vu)`` by finite-
  differencing successive fixes in **our own** ENU frame (so the vector is
  self-consistent with the position the estimator derives from lat/lon), with a
  light EMA to tame 1-Hz differencing noise. The reported scalar speed is used
  only as a first-sample fallback.
* Everything here is pure-Python and offline. The Socket.IO client is imported
  lazily so the package still works on a bare Pi without ``python-socketio``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from ..geo import Origin
from .schema import TelemetryPacket
from .source import TelemetrySource


class ArdFrameError(ValueError):
    """An ARD envelope could not be converted, with the reason why.

    Raised rather than returning ``None`` so a bad feed is *loud*. Silently
    dropping frames is the worst failure mode here: the operator sees an empty
    recovery map and has no idea whether the rocket is quiet, the link is down,
    or the telemetry simply lacks the fields the estimator needs.
    """


def _get(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """First present, non-null key from *d*, coerced to float.

    Only for genuinely optional fields (link health, hints). Anything the
    prediction depends on must go through :func:`_require` instead -- defaulting
    a missing altitude to 0 m would produce a confident, wrong landing zone.
    """
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _require(d: Dict[str, Any], key: str, where: str) -> float:
    """Read a field the prediction cannot do without, or raise."""
    v = d.get(key)
    if v is None or v == "":
        raise ArdFrameError(f"missing {where}.{key}")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ArdFrameError(f"{where}.{key} is not a number: {v!r}") from None


class ArdTelemetryAdapter:
    """Stateful converter: ARD envelope dict -> :class:`TelemetryPacket`.

    Stateful because the ENU velocity vector is reconstructed from successive
    positions. Feed envelopes in flight order via :meth:`to_packet`.
    """

    def __init__(self, origin: Origin, smoothing: float = 0.5):
        self.origin = origin
        self._alpha = float(smoothing)
        self._prev_t: Optional[float] = None
        self._prev_e: Optional[float] = None
        self._prev_n: Optional[float] = None
        self._prev_u: Optional[float] = None
        self._ve = 0.0
        self._vn = 0.0
        self._vu = 0.0
        self._count = 0

    def reset(self) -> None:
        """Forget history (e.g. before replaying a new flight)."""
        self.__init__(self.origin, self._alpha)

    def to_packet(self, envelope: Dict[str, Any]) -> TelemetryPacket:
        """Convert one ARD envelope, or raise :class:`ArdFrameError` saying why.

        Strict on purpose: every field read here feeds the landing prediction,
        so a missing one must surface as an error, not a zero. Callers that must
        survive a noisy link (the live sources) catch this and report counts
        rather than dying on a single corrupt frame.
        """
        if not isinstance(envelope, dict):
            raise ArdFrameError(f"envelope is {type(envelope).__name__}, not an object")
        packet = envelope.get("packet") or {}
        derived = envelope.get("derived") or {}
        # Link health lives in its own block in the ARD envelope, not in packet.
        quality = envelope.get("quality") or {}
        if not packet:
            raise ArdFrameError("envelope has no 'packet' block")
        # The landing estimator is useless without a horizontal position fix.
        # NOTE: ARD's downlink does not yet carry GPS -- until it does, this is
        # the error you will see, and it is the correct thing to see.
        if not derived:
            raise ArdFrameError("envelope has no 'derived' block (no position fix)")

        t = _require(packet, "time", "packet") / 1000.0      # ms since launch -> s
        lat = _require(derived, "latitude", "derived")
        lon = _require(derived, "longitude", "derived")
        alt_agl = _require(packet, "altitude", "packet")     # ARD altitude is AGL (m)
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ArdFrameError(f"position out of range: lat={lat}, lon={lon}")
        # LZE's estimator expects GPS altitude ASL; add the site elevation so it
        # recovers the same AGL after subtracting the origin elevation.
        alt_gps = alt_agl + self.origin.elevation

        # Reconstruct the ENU velocity vector in our own frame for consistency
        # with the position the estimator will derive from this lat/lon.
        e, n, _ = self.origin.geo_to_enu(lat, lon, alt_gps)
        ve, vn, vu = self._reconstruct_velocity(t, e, n, alt_agl, derived)

        pkt_id = int(_get(quality, "packet_id", default=_get(
            packet, "packet_id", default=float(self._count))))
        rssi = _get(quality, "rssi", default=_get(packet, "rssi", default=0.0))
        self._count += 1
        return TelemetryPacket(
            t=t,
            lat=lat,
            lon=lon,
            alt_gps=alt_gps,
            alt_baro_agl=max(0.0, alt_agl),
            ve=ve,
            vn=vn,
            vu=vu,
            packet_id=pkt_id,
            rssi=rssi,
        )

    def _reconstruct_velocity(
        self, t: float, e: float, n: float, u: float, derived: Dict[str, Any]
    ):
        prev_t = self._prev_t
        if prev_t is None or t <= prev_t:
            # First fix (or a stalled/duplicate timestamp): fall back to the
            # scalar speed + heading ARD provides, treating it as horizontal.
            if prev_t is None:
                speed = _get(derived, "velocity", "speed", default=0.0)
                az = _get(derived, "azimuth_deg", default=0.0)
                import math

                self._ve = speed * math.sin(math.radians(az))
                self._vn = speed * math.cos(math.radians(az))
                self._vu = 0.0
        else:
            dt = t - prev_t
            raw_ve = (e - self._prev_e) / dt
            raw_vn = (n - self._prev_n) / dt
            raw_vu = (u - self._prev_u) / dt
            a = self._alpha
            self._ve = (1 - a) * self._ve + a * raw_ve
            self._vn = (1 - a) * self._vn + a * raw_vn
            self._vu = (1 - a) * self._vu + a * raw_vu
        self._prev_t, self._prev_e, self._prev_n, self._prev_u = t, e, n, u
        return self._ve, self._vn, self._vu


class _RejectReporter:
    """Counts rejected frames and makes sustained failure impossible to miss.

    A live radio link drops the odd frame, so one bad envelope must not kill the
    server. But a feed where *everything* is rejected looks exactly like a quiet
    rocket -- an empty map -- which is the failure mode we refuse to ship. So:
    tolerate individual frames, shout about a pattern.
    """

    def __init__(self, label: str, announce_after: int = 5):
        self.label = label
        self.announce_after = announce_after
        self.accepted = 0
        self.rejected = 0
        self._streak = 0
        self._announced = False

    def ok(self) -> None:
        if self._announced and self._streak >= self.announce_after:
            print(f"[{self.label}] recovered: telemetry is being accepted again "
                  f"({self.rejected} frame(s) rejected in total).")
        self.accepted += 1
        self._streak = 0
        self._announced = False

    def bad(self, reason: str) -> None:
        self.rejected += 1
        self._streak += 1
        if self._streak == 1 and self.accepted == 0 and self.rejected == 1:
            print(f"[{self.label}] rejected a telemetry frame: {reason}")
        if self._streak == self.announce_after and not self._announced:
            self._announced = True
            print(
                f"\n[{self.label}] ERROR: {self._streak} consecutive frames "
                f"rejected -- no landing prediction is being produced.\n"
                f"  reason: {reason}\n"
                f"  (accepted {self.accepted} frame(s) so far). If this says a "
                f"position fix is missing, the telemetry has no GPS and the "
                f"landing estimator cannot run on it.\n"
            )


def ard_envelopes_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load ARD envelopes from a newline-delimited-JSON capture file.

    Handy for offline replay: record the dashboard's ``telemetry_data`` events
    to a ``.jsonl`` file, then replay them through the predictor with no radio
    or network. Lines that are not telemetry envelopes are skipped.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "packet" in obj and "derived" in obj:
                out.append(obj)
    return out


class ArdReplaySource(TelemetrySource):
    """Replay ARD envelopes (e.g. from a capture file) as LZE packets."""

    def __init__(self, envelopes: List[Dict[str, Any]], origin: Origin):
        self._envelopes = envelopes
        self._adapter = ArdTelemetryAdapter(origin)

    def __iter__(self) -> Iterator[TelemetryPacket]:
        self._adapter.reset()
        report = _RejectReporter("ard-file")
        for env in self._envelopes:
            try:
                pkt = self._adapter.to_packet(env)
            except ArdFrameError as exc:
                report.bad(str(exc))
                continue
            report.ok()
            yield pkt
        if report.rejected:
            print(f"[ard-file] {report.accepted} frame(s) used, "
                  f"{report.rejected} rejected.")


class ArdRestSource(TelemetrySource):
    """Live source over the ARD backend's **REST API** (no extra dependencies).

    An alternative to :class:`ArdSocketIOSource` for when the websocket is
    awkward -- CORS trouble, a proxy in the way, or simply not wanting
    ``python-socketio`` on the Pi. It uses only the documented endpoints:

    * ``GET /telemetry/history`` once at start, to backfill the flight so far
      (the same reason the LZE server replays history to late browsers -- a
      predictor that joins mid-flight should not start blind);
    * ``GET /telemetry/latest`` polled thereafter, de-duplicated on packet time.

    Polling is inherently lossier than the push stream: at 4 Hz against 10 Hz
    telemetry you see roughly every third frame. That is fine for landing
    prediction -- the estimator is driven by GPS-rate motion, not packet count --
    but prefer the Socket.IO source when you can.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:5000",
        origin: Optional[Origin] = None,
        poll_hz: float = 4.0,
        backfill: bool = True,
    ):
        if origin is None:
            raise ValueError("ArdRestSource requires an ENU origin")
        self.url = url.rstrip("/")
        self.poll_interval = 1.0 / max(poll_hz, 0.1)
        self.backfill = backfill
        self._adapter = ArdTelemetryAdapter(origin)

    def _get_json(self, path: str, timeout: float = 3.0) -> Optional[Any]:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(self.url + path, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def health(self) -> bool:
        """True if the ARD backend answers on ``GET /health``."""
        body = self._get_json("/health", timeout=2.0)
        return bool(body) and body.get("status") == "ok"

    @staticmethod
    def _unwrap(body: Any) -> Any:
        """ARD wraps REST payloads as ``{"success": true, "data": ...}``."""
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

    def __iter__(self) -> Iterator[TelemetryPacket]:
        import time

        self._adapter.reset()
        last_t: Optional[float] = None
        report = _RejectReporter("ard-rest")
        offline_streak = 0

        if self.backfill:
            history = self._unwrap(self._get_json("/telemetry/history")) or []
            for env in history:
                try:
                    pkt = self._adapter.to_packet(env)
                except ArdFrameError as exc:
                    report.bad(str(exc))
                    continue
                report.ok()
                if last_t is None or pkt.t > last_t:
                    last_t = pkt.t
                    yield pkt

        while True:
            body = self._get_json("/telemetry/latest")
            if body is None:
                # The backend is unreachable. Say so instead of spinning quietly.
                offline_streak += 1
                if offline_streak in (1, 10) or offline_streak % 60 == 0:
                    print(f"[ard-rest] backend unreachable at {self.url} "
                          f"({offline_streak} failed poll(s)) -- retrying.")
                time.sleep(self.poll_interval)
                continue
            if offline_streak:
                print(f"[ard-rest] backend reachable again after "
                      f"{offline_streak} failed poll(s).")
                offline_streak = 0

            env = self._unwrap(body)
            if env:
                try:
                    pkt = self._adapter.to_packet(env)
                except ArdFrameError as exc:
                    report.bad(str(exc))
                    time.sleep(self.poll_interval)
                    continue
                report.ok()
                # Poll rate and telemetry rate are unrelated, so the same frame
                # comes back repeatedly; only advance on a genuinely new one.
                if last_t is None or pkt.t > last_t:
                    last_t = pkt.t
                    yield pkt
            time.sleep(self.poll_interval)


class ArdSocketIOSource(TelemetrySource):
    """Live source: subscribe to the ARD backend's Socket.IO telemetry stream.

    Connects to the ARD Flask/Socket.IO backend (default ``http://127.0.0.1:5000``),
    listens for ``telemetry_data`` events, and yields converted packets in real
    time. Requires ``python-socketio[client]`` (an optional dependency); the
    import is deferred so the rest of the package runs without it.
    """

    def __init__(self, url: str = "http://127.0.0.1:5000", origin: Optional[Origin] = None):
        if origin is None:
            raise ValueError("ArdSocketIOSource requires an ENU origin")
        self.url = url
        self._adapter = ArdTelemetryAdapter(origin)

    def __iter__(self) -> Iterator[TelemetryPacket]:  # pragma: no cover - network
        try:
            import queue

            import socketio  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ArdSocketIOSource needs python-socketio: "
                "pip install 'python-socketio[client]'"
            ) from exc

        q: "queue.Queue[Optional[TelemetryPacket]]" = queue.Queue(maxsize=1024)
        sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)

        report = _RejectReporter("ard")

        @sio.on("telemetry_data")
        def _on_telemetry(data):  # noqa: ANN001
            try:
                pkt = self._adapter.to_packet(data)
            except ArdFrameError as exc:
                report.bad(str(exc))
                return
            report.ok()
            try:
                q.put_nowait(pkt)
            except queue.Full:
                pass

        @sio.event
        def connect():  # noqa: D401
            # Ask the backend to flush buffered history so we start immediately.
            try:
                sio.emit("request_telemetry")
            except Exception:  # noqa: BLE001
                pass

        sio.connect(self.url, transports=["websocket", "polling"])
        try:
            while True:
                pkt = q.get()
                if pkt is None:
                    break
                yield pkt
        finally:
            try:
                sio.disconnect()
            except Exception:  # noqa: BLE001
                pass
