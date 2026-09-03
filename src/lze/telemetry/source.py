"""Telemetry source abstraction: where live packets come from.

The live predictor consumes an iterable of :class:`TelemetryPacket`, decoupled
from transport. On the ground station (Raspberry Pi) that is the LoRa receiver
over serial; in the demo it is a replayed trajectory. Implementations here:

* :class:`ReplaySource`  -- replay a simulated flight (used by the demo/tests).
* :class:`SerialLoRaSource` -- read newline-delimited JSON from a serial LoRa
  modem (the real Pi path). Requires ``pyserial``; kept import-light so the rest
  of the package works without it.
* :class:`UDPSource` -- read JSON datagrams (e.g. from a separate radio daemon).
"""
from __future__ import annotations

from typing import Iterable, Iterator, List

from .replay import stream_packets
from .schema import TelemetryPacket


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
