"""Telemetry source abstraction: where live packets come from.

The live predictor consumes an iterable of :class:`TelemetryPacket`, decoupled
from transport. On the ground station (Raspberry Pi) that is the LoRa receiver
over serial; in the demo it is a replayed trajectory. Implementations here:

* :class:`ReplaySource`  -- replay a simulated flight (used by the demo/tests).
* :class:`SerialLoRaSource` -- read newline-delimited JSON from a serial LoRa
  modem (the real Pi path). Requires ``pyserial``; kept import-light so the rest
  of the package works without it.
* :class:`UDPSource` -- read JSON datagrams (e.g. from a separate radio daemon).

:func:`anchor_origin` takes the local map origin from the first real fixes
rather than from a configured coordinate -- see its docstring for why that is
the better default.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Iterable, Iterator, List, Optional, Tuple

from ..geo import Origin
from .replay import stream_packets
from .schema import TelemetryPacket


def anchor_origin(
    packets: Iterable[TelemetryPacket],
    n: int = 10,
    min_still: int = 3,
    still_ms: float = 2.0,
    give_up_after: int = 200,
) -> Tuple[Iterator[TelemetryPacket], Optional[Origin]]:
    """Derive the ENU origin from fixes taken while the tracker sits still.

    Returns ``(stream, origin)``, or ``(stream, None)`` if no stationary fixes
    were seen -- the caller then falls back to the configured pad.

    The origin is only the reference point of the local east/north frame, and
    the tracker already knows where that is: it is sitting on the pad when it
    gets its first fixes. Using those beats a coordinate typed into a config.
    They are measured by the *same receiver* that will measure the flight, so
    bias common to both cancels out of the offsets that matter; they supply the
    pad elevation for free, so altitude AGL reads zero on the ground instead of
    being clamped against a guessed site height; and there is nothing to look
    up or mistype.

    **Stationary is the whole trick.** An earlier version took the first *n*
    fixes outright, which is wrong whenever the feed starts at or after
    liftoff: ten seconds of boost medianed to 920 m of "pad elevation" and put
    the landing prediction out by 52 m. So a fix only counts toward the origin
    while both vertical and horizontal speed are below *still_ms*, and
    collection stops at the first sign of movement. Sitting on the pad for
    twenty minutes costs nothing; powering up mid-flight yields ``None``
    instead of a confident wrong answer.

    At least *min_still* such fixes are required and up to *n* are medianed:
    one fix carries a few metres of noise, and a median of three is not moved
    by a single bad sample. On a real pad there are hundreds to choose from, so
    the floor costs nothing and rules out anchoring to one borderline reading. Buffered
    packets are re-emitted with AGL recomputed against the chosen origin, so
    nothing downstream sees a provisional value.

    The one assumption left: the tracker is stationary *where you want the
    origin*. Sitting in the car in the car park anchors the map to the car park.
    """
    it = iter(packets)
    still: List[TelemetryPacket] = []
    seen: List[TelemetryPacket] = []

    for pkt in it:
        seen.append(pkt)
        moving = (abs(pkt.vu) > still_ms or pkt.horizontal_speed > still_ms)
        if moving:
            break                       # it has started flying; stop collecting
        still.append(pkt)
        if len(still) >= n or len(seen) >= give_up_after:
            break

    if len(still) < min_still:
        # Too little stationary data: the feed began in motion, or only caught
        # a moment of near-stillness. Say nothing here -- the
        # caller has the context to explain and to choose the fallback.
        def _passthrough() -> Iterator[TelemetryPacket]:
            yield from seen
            yield from it
        return _passthrough(), None

    origin = Origin(
        lat=statistics.median(p.lat for p in still),
        lon=statistics.median(p.lon for p in still),
        elevation=statistics.median(p.alt_gps for p in still),
    )

    def _stream() -> Iterator[TelemetryPacket]:
        for p in seen:                  # includes the first moving packet
            yield dataclasses.replace(
                p, alt_baro_agl=max(0.0, p.alt_gps - origin.elevation))
        for p in it:
            yield dataclasses.replace(
                p, alt_baro_agl=max(0.0, p.alt_gps - origin.elevation))

    return _stream(), origin


class TelemetrySource(Iterable[TelemetryPacket]):
    """Base class: iterate to receive telemetry packets."""

    def __iter__(self) -> Iterator[TelemetryPacket]:  # pragma: no cover - abstract
        raise NotImplementedError


class ReplaySource(TelemetrySource):
    """Replay a pre-built list of packets (from a simulated flight)."""

    def __init__(self, packets: List[TelemetryPacket], realtime: bool = False, speed: float = 1.0):
        self._packets = packets
        self._realtime = realtime
        self._speed = speed

    def __iter__(self) -> Iterator[TelemetryPacket]:
        return stream_packets(self._packets, realtime=self._realtime, speed=self._speed)


class SerialLoRaSource(TelemetrySource):
    """Read newline-delimited JSON telemetry from a serial LoRa modem.

    This is the ground-station path: the SX1276 receiver forwards each decoded
    frame as a JSON line matching :class:`TelemetryPacket`. Kept as a thin
    reader so the flight-computer firmware format can evolve independently.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.port = port
        self.baud = baud

    def __iter__(self) -> Iterator[TelemetryPacket]:  # pragma: no cover - hardware
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyserial is required for SerialLoRaSource (pip install pyserial)"
            ) from exc
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    yield TelemetryPacket.from_json(line)
                except Exception:  # noqa: BLE001 - skip malformed frames
                    continue


class UDPSource(TelemetrySource):
    """Receive JSON telemetry datagrams over UDP."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port

    def __iter__(self) -> Iterator[TelemetryPacket]:  # pragma: no cover - network
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        while True:
            data, _ = sock.recvfrom(4096)
            line = data.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                yield TelemetryPacket.from_json(line)
            except Exception:  # noqa: BLE001
                continue