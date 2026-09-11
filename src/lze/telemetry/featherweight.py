"""Featherweight GPS Tracker / Ground Station v2 telemetry source.

The Featherweight Ground Station v2 exposes the received tracker telemetry over
its micro-USB port as a serial stream (115200 8N1, no flow control). Among the
ASCII packets it emits, ``GPS_STAT`` carries everything the landing estimator
needs -- and unlike the ARD downlink, it carries a **real GPS fix**::

    @ GPS_STAT 203 2020 11 15 01:20:21.986 CRC_OK TRK secondTrk Alt 5655
      lt 39.55612 ln -105.1032 Vel 0 -155 0 Fix 3 # 9 4 2 0 ... CRC: 6A1D

Field meanings (from the tracker manual, Appendix A):

===============  ===========================================================
``CRC_OK``       the received LoRa packet passed its checksum
``TRK``          GPS unit type -- see below, this filter matters
``Alt``          altitude **ASL in feet**
``lt`` / ``ln``  latitude / longitude, decimal degrees
``Vel``          horizontal speed (ft/s), heading (deg), vertical speed (ft/s)
``Fix``          0 = no fix, 2 = 2-D, 3 = 3-D
``#``            satellites used in the solution
===============  ===========================================================

Three filters are applied to ``GPS_STAT``, and each one prevents a specific
wrong answer:

* **Unit type.** The ground station reports its *own* GPS position with type
  ``GS``. Accepting those would track the launch table instead of the rocket --
  and the resulting map would look entirely plausible. Only ``TRK`` (tracker
  aboard the vehicle) and ``FND`` (a landed unit reporting its final position)
  are used.
* **Fix quality.** A 2-D fix has no trustworthy altitude, and the whole flight
  phase machine keys off altitude, so anything below a 3-D fix is refused.
* **CRC.** Packets the ground station marks as failing their checksum are
  dropped rather than fed into the predictor.

Velocity arrives directly here, so -- unlike the ARD bridge -- nothing has to
be reconstructed by differencing successive positions.

There is no barometer on this link, and measurement says that is fine. The
state estimator fuses altitude as ``0.7 x baro + 0.3 x GPS``; fed from here
both slots hold the same GPS number, so nothing averages out and the fused
altitude carries the full GPS vertical noise -- 5.0 m against the 2.05 m the
model trained on (``gps_vertical_noise_m`` and ``baro_noise_m`` in the config),
2.4x noisier. Over 60 paired flights -- the same flight run twice, differing
only in this channel -- that costs **+5.1 m of final landing error, SE 2.9 m,
95% CI [-0.5, +10.7] m: not distinguishable from zero** against a ~50 m final
error. It lands softly for two reasons: at 3 km altitude 5 m is 0.17% of the
value and the trees split on altitude far more coarsely than that; and ``vu``
is downlinked directly rather than differenced from altitude, so descent rate
-- what actually drives the drift prediction -- never touches the noisy
channel. No retrain is needed before flying this source.

``GPS_STAT`` carries no link health, so ``RX_NOMTK`` / ``RX_FOUND`` are parsed
alongside it for RSSI, SNR and tracker battery, and the most recent values are
stamped onto the next position packet. Without this the dashboard shows
``RSSI 0`` for the whole flight, which reads as a dead link.

Reading the port
----------------
The port carries binary frames as well as ASCII, so a bare ``readline()`` can
swallow a real packet when it lands mid-binary. Every reader here instead
syncs on the ``@`` start byte and reads the rest of the line -- which is why
the ``@`` is **optional** in the patterns below: once you have synced on it,
it is no longer part of the text you parse.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional

from ..geo import Origin
from .schema import TelemetryPacket
from .source import TelemetrySource

FT_TO_M = 0.3048

#: GPS unit types that describe the flying vehicle, not the ground station.
VEHICLE_UNIT_TYPES = ("TRK", "FND")

# Anchored on the field labels rather than token positions, for two reasons:
# the tracker ID is user-settable ("set TrackerID My_Rocket01"), so the number
# of tokens before "Alt" is not fixed; and counting positions is how you end up
# reading the tracker's *name* where you meant to read its unit type.
#
# The leading "@" is optional: a reader that syncs on it has already consumed
# it, while a captured log file still has it.
_GPS_STAT = re.compile(
    r"@?\s*GPS_STAT\s+\d+\s+"                     # sync, type, packet length
    r"(?P<y>\d+)\s+(?P<mo>\d+)\s+(?P<d>\d+)\s+"   # year month date
    r"(?P<time>[\d:.]+)\s+"                       # HH:MM:SS.mmm
    r"(?P<crc>\S+)\s+"                            # CRC_OK / otherwise
    r"(?P<unit>TRK|GS|FND)\b.*?"                  # GPS unit type
    r"\bAlt\s+(?P<alt>-?\d+(?:\.\d+)?)\s+"        # altitude ASL, feet
    r"lt\s+(?P<lat>-?\d+(?:\.\d+)?)\s+"
    r"ln\s+(?P<lon>-?\d+(?:\.\d+)?)\s+"
    r"Vel\s+(?P<vh>-?\d+(?:\.\d+)?)\s+"           # horizontal speed, ft/s
    r"(?P<hdg>-?\d+(?:\.\d+)?)\s+"                # heading, degrees
    r"(?P<vu>-?\d+(?:\.\d+)?)\s+"                 # vertical speed, ft/s
    r"Fix\s+(?P<fix>\d+)",
    re.IGNORECASE,
)

#: Satellite count, optional -- it follows the "#" marker after the fix.
_SATS = re.compile(r"Fix\s+\d+\s+#\s+(?P<sats>\d+)", re.IGNORECASE)

#: Radio health, reported per received packet rather than inside GPS_STAT.
_RX_STAT = re.compile(
    r"@?\s*RX_(?P<kind>NOMTK|FOUND)\b.*?"
    r"\bRSSI\s+(?P<rssi>-?\d+(?:\.\d+)?)\s+"
    r"SNR\s+(?P<snr>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

#: Tracker battery, millivolts. Present on some RX packets, absent on others.
_TRK_BATT = re.compile(r"\btrk_B_V\s+(?P<mv>\d+(?:\.\d+)?)", re.IGNORECASE)

#: Ground-station flight-state transitions -- surfaced, not acted on.
_FS_CHNGE = re.compile(r"@?\s*FS_CHNGE\b.*?\bstate:\s*(?P<state>-?\d+)",
                       re.IGNORECASE)


class FeatherweightFrameError(ValueError):
    """A Featherweight line could not be used, with the reason why."""


def parse_gps_stat(line: str) -> dict:
    """Parse one ``GPS_STAT`` line into SI units, or raise.

    Returns a dict with ``lat``, ``lon``, ``alt_asl_m``, ``ve``, ``vn``,
    ``vu`` (m/s, ENU), ``fix``, ``sats``, ``unit``, ``crc_ok`` and ``sod``
    (seconds of day). Raising rather than returning ``None`` keeps a bad feed
    loud.
    """
    m = _GPS_STAT.search(line)
    if not m:
        raise FeatherweightFrameError("not a GPS_STAT packet")

    try:
        h, mi, s = m.group("time").split(":")
        sod = int(h) * 3600 + int(mi) * 60 + float(s)
    except (ValueError, AttributeError):
        raise FeatherweightFrameError(
            f"unparseable timestamp {m.group('time')!r}") from None

    sats = _SATS.search(line)
    vh = float(m.group("vh")) * FT_TO_M          # horizontal speed, m/s
    hdg = math.radians(float(m.group("hdg")))    # heading, deg clockwise from N
    return {
        "unit": m.group("unit").upper(),
        "crc_ok": m.group("crc").upper() == "CRC_OK",
        "fix": int(m.group("fix")),
        "sats": int(sats.group("sats")) if sats else 0,
        "sod": sod,
        "lat": float(m.group("lat")),
        "lon": float(m.group("lon")),
        "alt_asl_m": float(m.group("alt")) * FT_TO_M,
        # Heading is clockwise from north, so east is sin and north is cos.
        "ve": vh * math.sin(hdg),
        "vn": vh * math.cos(hdg),
        "vu": float(m.group("vu")) * FT_TO_M,
    }


def parse_rx_stat(line: str) -> Optional[dict]:
    """Parse link health out of an ``RX_NOMTK`` / ``RX_FOUND`` line.

    Returns ``None`` for anything else. Unlike :func:`parse_gps_stat` this does
    not raise: link health is a nice-to-have on the dashboard, and a flight
    that loses it still predicts a landing perfectly well.
    """
    m = _RX_STAT.search(line)
    if not m:
        return None
    batt = _TRK_BATT.search(line)
    return {
        "kind": m.group("kind").upper(),
        "rssi": float(m.group("rssi")),
        "snr": float(m.group("snr")),
        # Millivolts on the wire; volts is what anyone reads off a screen.
        "batt_v": float(batt.group("mv")) / 1000.0 if batt else None,
    }


def parse_fs_chnge(line: str) -> Optional[int]:
    """Return the new flight state from an ``FS_CHNGE`` line, else ``None``."""
    m = _FS_CHNGE.search(line)
    return int(m.group("state")) if m else None


class FeatherweightDecoder:
    """Stateful line -> :class:`TelemetryPacket` decoder.

    Stateful for two reasons. First, ``TelemetryPacket.t`` is seconds since
    launch while the tracker reports wall-clock time: sitting on the pad with
    the ground station powered up for twenty minutes would otherwise hand the
    estimator a ``t`` of 1200 s, and its phase machine -- which separates boost
    from coast using the motor burn time -- would call the whole flight
    "coast". So the clock starts on a detected launch: the first sustained
    upward velocity. Second, link health arrives on *different packets* than
    position, so the most recent values are carried forward.
    """

    def __init__(
        self,
        origin: Origin,
        unit_types: Iterable[str] = VEHICLE_UNIT_TYPES,
        require_3d_fix: bool = True,
        launch_detect_vu_ms: float = 15.0,
    ):
        self.origin = origin
        self.unit_types = tuple(u.upper() for u in unit_types)
        self.require_3d_fix = require_3d_fix
        self.launch_vu = float(launch_detect_vu_ms)
        self._t0: Optional[float] = None       # seconds-of-day at launch
        self._count = 0
        # Most recent link health, stamped onto the next position packet.
        self.rssi: float = 0.0
        self.snr: float = 0.0
        self.tracker_batt_v: Optional[float] = None
        self.flight_state: Optional[int] = None
        self.sats: int = 0

    @property
    def launched(self) -> bool:
        return self._t0 is not None

    def feed(self, line: str) -> Optional[TelemetryPacket]:
        """Route one serial line by packet type.

        Returns a packet for a usable ``GPS_STAT``, ``None`` for a line that
        was absorbed (link health, flight state) or is not ours, and raises
        :class:`FeatherweightFrameError` for a ``GPS_STAT`` that exists but
        cannot be trusted -- the case worth being loud about.
        """
        rx = parse_rx_stat(line)
        if rx is not None:
            self.rssi = rx["rssi"]
            self.snr = rx["snr"]
            if rx["batt_v"] is not None:
                self.tracker_batt_v = rx["batt_v"]
            return None

        state = parse_fs_chnge(line)
        if state is not None:
            self.flight_state = state
            return None

        if not _GPS_STAT.search(line):
            return None            # BATT_BLE, TX_STAT, binary frames, noise
        return self.to_packet(line)

    def to_packet(self, line: str) -> TelemetryPacket:
        """Decode one ``GPS_STAT`` line, or raise."""
        f = parse_gps_stat(line)

        if f["unit"] not in self.unit_types:
            raise FeatherweightFrameError(
                f"unit type {f['unit']} is not the vehicle "
                f"(expected one of {', '.join(self.unit_types)}) -- "
                f"'GS' is the ground station's own position")
        if not f["crc_ok"]:
            raise FeatherweightFrameError("packet failed its CRC check")
        if self.require_3d_fix and f["fix"] < 3:
            raise FeatherweightFrameError(
                f"fix quality {f['fix']} (need 3-D); altitude is not trustworthy")

        # Start the flight clock at launch, not at power-on.
        if self._t0 is None and f["vu"] >= self.launch_vu:
            self._t0 = f["sod"]
        t = 0.0 if self._t0 is None else max(0.0, f["sod"] - self._t0)

        alt_agl = f["alt_asl_m"] - self.origin.elevation
        self._count += 1
        self.sats = f["sats"]
        return TelemetryPacket(
            t=t,
            lat=f["lat"],
            lon=f["lon"],
            alt_gps=f["alt_asl_m"],
            # No barometer on this link, so the estimator's baro channel is
            # fed GPS altitude too. Measured cost: +5.1 m of final error,
            # SE 2.9 m -- not distinguishable from zero. See the docstring.
            alt_baro_agl=max(0.0, alt_agl),
            ve=f["ve"],
            vn=f["vn"],
            vu=f["vu"],
            packet_id=self._count,
            rssi=self.rssi,     # carried over from the last RX_NOMTK/RX_FOUND
        )


class _FwReporter:
    """Counts what the port is doing and says so periodically.

    Silence is the one thing a live reader must never do. Without a heartbeat,
    a wrong port, a wrong baud, a tracker that is not transmitting and a
    perfectly healthy link waiting for liftoff all look exactly the same on
    screen: nothing. The counters below turn that into a diagnosis.
    """

    def __init__(self, announce_after: int = 8, heartbeat_s: float = 5.0):
        self.announce_after = announce_after
        self.heartbeat_s = heartbeat_s
        self.accepted = 0
        self.rejected = 0
        self.lines = 0
        self.last_reason = ""
        self._streak = 0
        self._announced = False
        self._t0 = time.monotonic()
        self._next_beat = self._t0 + heartbeat_s

    def saw_line(self) -> None:
        self.lines += 1

    def heartbeat(self, decoder: "FeatherweightDecoder") -> None:
        """Print a one-line status if it is time. Cheap enough to call per line."""
        now = time.monotonic()
        if now < self._next_beat:
            return
        self._next_beat = now + self.heartbeat_s
        age = now - self._t0
        if self.accepted:
            extra = (f"fix ok, {decoder.sats} sats"
                     f"{f', RSSI {decoder.rssi:.0f} dBm' if decoder.rssi else ''}"
                     f"{'' if decoder.launched else ', waiting for liftoff'}")
        elif self.rejected:
            extra = f"NONE USABLE -- last reason: {self.last_reason}"
        elif self.lines:
            extra = ("no GPS_STAT packets in that traffic -- wrong baud, or the "
                     "tracker is not transmitting")
        else:
            extra = ("NOTHING ON THE PORT -- wrong port, cable, or the ground "
                     "station is off")
        print(f"[featherweight] {age:4.0f}s  lines {self.lines:<6} "
              f"packets {self.accepted:<6} rejected {self.rejected:<5} {extra}")

    def ok(self) -> None:
        self.accepted += 1
        self._streak = 0
        self._announced = False

    def bad(self, reason: str) -> None:
        self.rejected += 1
        self.last_reason = reason
        self._streak += 1
        if self._streak == self.announce_after and not self._announced:
            self._announced = True
            print(
                f"\n[featherweight] WARNING: {self._streak} consecutive GPS_STAT "
                f"packets rejected -- no landing prediction is being produced.\n"
                f"  reason: {reason}\n"
                f"  (accepted {self.accepted} so far)\n"
            )


def iter_serial_lines(ser, sync: bytes = b"@") -> Iterator[str]:
    """Yield ASCII packet lines from an open serial port, synced on ``@``.

    The port interleaves binary frames with ASCII, so a bare ``readline()``
    can consume a real packet as part of a binary run. Syncing on the start
    byte first is what makes the reader survive that -- and it means the
    ``@`` is already consumed, so the yielded line begins at the packet type.
    """
    while True:
        # read_until does the scan in C; it stops after consuming the sync byte.
        if not ser.read_until(sync).endswith(sync):
            continue                       # timed out with no packet start
        yield ser.readline().decode("ascii", errors="ignore").strip()


def stamp_line(line: str, t_mono0: float) -> str:
    """Prefix a raw line with when the ground station received it.

    The packet already carries the *tracker's* GPS time. What it cannot tell
    you is when the line arrived here, so link latency is unmeasurable and a
    dropout is indistinguishable from a slow packet. Two clocks are recorded:
    the wall clock, so the log lines up with everything else that day, and a
    monotonic one, because a wall clock can step mid-flight (NTP, someone
    setting the time) while a monotonic clock cannot run backwards.

    The prefix is deliberately harmless to the parsers: they anchor on
    ``GPS_STAT`` / ``RX_`` with :func:`re.search`, so a stamped capture replays
    exactly like an unstamped one -- pinned by
    ``test_stamped_capture_parses_identically``.
    """
    utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return f"{utc} {time.monotonic() - t_mono0:.3f} {line}"


class FeatherweightSource(TelemetrySource):
    """Read live tracker telemetry from a Ground Station v2 over USB serial.

    ``pyserial`` is imported lazily so the package still works without it::

        python scripts/run_live.py --source featherweight \\
            --serial-port /dev/ttyUSB0

    On macOS the port is typically ``/dev/cu.usbserial-XXXXXXXX``; on Linux and
    the Raspberry Pi, ``/dev/ttyUSB0``.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,          # per the tracker manual, Appendix A
        origin: Optional[Origin] = None,
        capture_path: Optional[str] = None,
        **decoder_kwargs,
    ):
        if origin is None:
            raise ValueError("FeatherweightSource requires an ENU origin")
        self.port = port
        self.baud = baud
        self.capture_path = capture_path
        self.decoder = FeatherweightDecoder(origin, **decoder_kwargs)

    def __iter__(self) -> Iterator[TelemetryPacket]:  # pragma: no cover - hardware
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for FeatherweightSource "
                "(pip install pyserial)"
            ) from exc

        report = _FwReporter()
        announced_launch = False
        # A launch happens once. Without a capture there is nothing to re-run
        # the estimator against afterwards, so the log is opened before the
        # port and every line is flushed as it arrives -- a power loss then
        # still leaves a usable file rather than an empty buffer.
        cap = open(self.capture_path, "w") if self.capture_path else None
        t_mono0 = time.monotonic()
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                print(f"[featherweight] listening on {self.port} at {self.baud} baud; "
                      f"waiting for launch (t stays 0 until liftoff is detected)")
                if cap is not None:
                    print(f"[featherweight] capturing raw lines to {self.capture_path}")
                first = True
                for line in iter_serial_lines(ser):
                    report.heartbeat(self.decoder)
                    if not line:
                        continue
                    report.saw_line()
                    if cap is not None:
                        cap.write(stamp_line(line, t_mono0) + "\n")
                        cap.flush()
                    try:
                        pkt = self.decoder.feed(line)
                    except FeatherweightFrameError as exc:
                        report.bad(str(exc))
                        continue
                    if pkt is None:
                        continue       # link health, flight state, other traffic
                    report.ok()
                    if first:
                        first = False
                        print(f"[featherweight] RECEIVING: first fix at "
                              f"{pkt.lat:.5f}, {pkt.lon:.5f}, "
                              f"{pkt.alt_gps:.0f} m ASL, {self.decoder.sats} sats")
                    if self.decoder.launched and not announced_launch:
                        announced_launch = True
                        print("[featherweight] liftoff detected -- flight clock started")
                    yield pkt
        finally:
            if cap is not None:
                cap.close()


class FeatherweightReplaySource(TelemetrySource):
    """Replay a captured Ground Station text log. No hardware, no network.

    Capture one alongside a live run with ``--record NAME`` (it writes
    ``NAME.txt``), or with any terminal program that can log to a file.
    """

    def __init__(self, lines: List[str], origin: Origin, **decoder_kwargs):
        self._lines = list(lines)
        self._origin = origin
        self._kwargs = decoder_kwargs

    @classmethod
    def from_file(cls, path: str, origin: Origin, **decoder_kwargs
                  ) -> "FeatherweightReplaySource":
        with open(path, "r", errors="ignore") as fh:
            return cls(fh.readlines(), origin, **decoder_kwargs)

    def __iter__(self) -> Iterator[TelemetryPacket]:
        decoder = FeatherweightDecoder(self._origin, **self._kwargs)
        report = _FwReporter()
        for line in self._lines:
            try:
                pkt = decoder.feed(line)
            except FeatherweightFrameError as exc:
                report.bad(str(exc))
                continue
            if pkt is None:
                continue
            report.ok()
            yield pkt
        print(f"[featherweight] replayed {report.accepted} GPS_STAT packet(s), "
              f"{report.rejected} rejected")