"""ARD dashboard bridge, forecast-wind seeding, and live-uncertainty inflation.

Covers the new integration surface: converting ARD telemetry envelopes into the
LZE packet/prediction pipeline, the optional launch-day forecast seed, and the
data-driven recovery-radius inflation.
"""
import math

import numpy as np
import pytest

from lze.estimator.state import OnlineStateEstimator
from lze.geo import Origin, wind_vector
from lze.live.predictor import LandingPredictor
from lze.model.train import train_surrogate
from lze.sim import PHASE_BOOST, simulate
from lze.sim.dataset import generate_dataset
from lze.telemetry.ard_adapter import (
    ArdFrameError,
    ArdReplaySource,
    ArdRestSource,
    ArdTelemetryAdapter,
    ard_envelopes_from_jsonl,
)
from lze.telemetry.replay import trajectory_to_packets
from lze.telemetry.schema import TelemetryPacket


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _envelope(origin: Origin, t_s: float, e: float, n: float, alt_agl: float,
              velocity=None, azimuth=None):
    """Build an ARD-shaped telemetry envelope for a point in the ENU frame."""
    lat, lon, _ = origin.enu_to_geo(e, n, alt_agl + origin.elevation)
    derived = {"latitude": lat, "longitude": lon, "east_m": e, "north_m": n}
    if velocity is not None:
        derived["velocity"] = velocity
    if azimuth is not None:
        derived["azimuth_deg"] = azimuth
    return {
        "type": "telemetry",
        "timestamp": int(t_s * 1000),
        "packet": {"time": t_s * 1000.0, "altitude": alt_agl},
        "derived": derived,
    }


def _le_packet_to_ard_envelope(pkt: TelemetryPacket, origin: Origin) -> dict:
    """Inverse of the adapter: turn an LZE packet into an ARD envelope."""
    return {
        "type": "telemetry",
        "timestamp": int(pkt.t * 1000),
        "packet": {
            "time": pkt.t * 1000.0,
            "altitude": pkt.alt_baro_agl,
            "packet_id": pkt.packet_id,
            "rssi": pkt.rssi,
        },
        "derived": {
            "latitude": pkt.lat,
            "longitude": pkt.lon,
            "velocity": math.sqrt(pkt.ve ** 2 + pkt.vn ** 2 + pkt.vu ** 2),
            "azimuth_deg": (math.degrees(math.atan2(pkt.ve, pkt.vn)) + 360) % 360,
        },
    }


@pytest.fixture(scope="module")
def trained(cfg):
    res = generate_dataset(cfg, n_flights=12, prefer="fallback", seed=0, dt=0.5)
    surrogate, _ = train_surrogate(res.samples, res.flight_id, test_frac=0.25, seed=0)
    return surrogate


# --------------------------------------------------------------------------- #
# Adapter unit tests
# --------------------------------------------------------------------------- #
def test_adapter_basic_fields(origin):
    adapter = ArdTelemetryAdapter(origin)
    pkt = adapter.to_packet(_envelope(origin, t_s=3.0, e=120.0, n=-40.0, alt_agl=800.0))
    assert pkt is not None
    assert pkt.t == pytest.approx(3.0)
    assert pkt.alt_baro_agl == pytest.approx(800.0, abs=1e-6)
    # lat/lon should round-trip back to the requested ENU offset (sub-metre).
    e, n, _ = origin.geo_to_enu(pkt.lat, pkt.lon, pkt.alt_gps)
    assert e == pytest.approx(120.0, abs=1.0)
    assert n == pytest.approx(-40.0, abs=1.0)


def test_adapter_raises_naming_the_missing_field(origin):
    # Silently dropping frames looks identical to a quiet rocket, so a frame the
    # estimator cannot use must say exactly what it is missing.
    adapter = ArdTelemetryAdapter(origin)
    bad = {"packet": {"time": 1000, "altitude": 10.0}, "derived": {"velocity": 5.0}}
    with pytest.raises(ArdFrameError, match="derived.latitude"):
        adapter.to_packet(bad)

    with pytest.raises(ArdFrameError, match="no 'derived' block"):
        adapter.to_packet({"packet": {"time": 1, "altitude": 2.0}})

    with pytest.raises(ArdFrameError, match="packet.altitude"):
        adapter.to_packet({"packet": {"time": 1000},
                           "derived": {"latitude": -30.8, "longitude": 143.1}})


def test_adapter_rejects_impossible_position(origin):
    adapter = ArdTelemetryAdapter(origin)
    with pytest.raises(ArdFrameError, match="out of range"):
        adapter.to_packet({"packet": {"time": 1000, "altitude": 10.0},
                           "derived": {"latitude": 999.0, "longitude": 143.1}})


def test_replay_source_skips_bad_frames_but_keeps_good_ones(origin):
    # One corrupt frame must not kill a whole flight replay.
    good = _envelope(origin, t_s=1.0, e=1.0, n=1.0, alt_agl=100.0)
    good2 = _envelope(origin, t_s=2.0, e=2.0, n=2.0, alt_agl=200.0)
    bad = {"packet": {"time": 1500}, "derived": {}}
    pkts = list(ArdReplaySource([good, bad, good2], origin))
    assert [round(p.t) for p in pkts] == [1, 2]


def test_adapter_velocity_reconstruction(origin):
    # smoothing=1.0 -> the second fix's velocity equals the raw finite difference.
    adapter = ArdTelemetryAdapter(origin, smoothing=1.0)
    adapter.to_packet(_envelope(origin, t_s=0.0, e=0.0, n=0.0, alt_agl=0.0))
    pkt = adapter.to_packet(_envelope(origin, t_s=1.0, e=10.0, n=5.0, alt_agl=80.0))
    assert pkt.ve == pytest.approx(10.0, abs=0.5)
    assert pkt.vn == pytest.approx(5.0, abs=0.5)
    assert pkt.vu == pytest.approx(80.0, abs=0.5)


def test_adapter_first_fix_uses_scalar_speed(origin):
    adapter = ArdTelemetryAdapter(origin)
    pkt = adapter.to_packet(
        _envelope(origin, t_s=0.0, e=0.0, n=0.0, alt_agl=0.0, velocity=10.0, azimuth=90.0)
    )
    # azimuth 90deg (due east) -> horizontal speed goes into ve.
    assert pkt.ve == pytest.approx(10.0, abs=0.5)
    assert pkt.vn == pytest.approx(0.0, abs=0.5)


def test_ard_jsonl_loader_skips_junk(origin, tmp_path):
    path = tmp_path / "capture.jsonl"
    good = _envelope(origin, t_s=1.0, e=1.0, n=2.0, alt_agl=100.0)
    import json
    path.write_text(
        "not json\n"
        + json.dumps({"type": "telemetry_status"}) + "\n"   # no packet/derived
        + json.dumps(good) + "\n"
    )
    envs = ard_envelopes_from_jsonl(str(path))
    assert len(envs) == 1


# --------------------------------------------------------------------------- #
# End-to-end: ARD telemetry -> landing prediction
# --------------------------------------------------------------------------- #
def test_ard_source_drives_prediction_to_landing(cfg, origin, trained):
    """A flight replayed *through the ARD envelope shape* must still converge."""
    tr = simulate(cfg, prefer="fallback", wind_east=5.0, wind_north=-3.0, dry_mass=15.0)
    tlat, tlon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
    packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=3)
    envelopes = [_le_packet_to_ard_envelope(p, origin) for p in packets]

    source = ArdReplaySource(envelopes, origin)
    pred = LandingPredictor(cfg, trained, origin)
    last_err = None
    seen_phases = set()
    for pkt in source:
        p = pred.process(pkt)
        seen_phases.add(p.phase)
        last_err = pred.error_against_truth(p, tlat, tlon)

    # The bridge preserves enough signal to recover the phases and land close.
    assert {"boost", "main"}.issubset(seen_phases)
    assert last_err < 250.0


# --------------------------------------------------------------------------- #
# Forecast wind seed
# --------------------------------------------------------------------------- #
def test_wind_vector_convention():
    # A "westerly" (FROM 270 deg) blows toward the east: +east, ~0 north.
    e, n = wind_vector(6.0, 270.0)
    assert e == pytest.approx(6.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)


def test_wind_seed_sets_ascent_wind(cfg, origin):
    est = OnlineStateEstimator(cfg, origin=origin, wind_seed=(6.0, -4.0))
    tr = simulate(cfg, prefer="fallback", wind_east=6.0, wind_north=-4.0, dry_mass=15.0)
    packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=1)
    boost = next(est.update(p) for p in packets)  # first packet is boost
    assert boost.phase == PHASE_BOOST
    # Seeded model carries the forecast wind during ascent instead of zero.
    assert boost.wind_e == pytest.approx(6.0)
    assert boost.wind_n == pytest.approx(-4.0)


def test_default_estimator_has_zero_ascent_wind(cfg, origin):
    est = OnlineStateEstimator(cfg, origin=origin)  # no seed = default behaviour
    tr = simulate(cfg, prefer="fallback", wind_east=6.0, wind_north=-4.0, dry_mass=15.0)
    packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=1)
    boost = next(est.update(p) for p in packets)
    assert boost.wind_e == 0.0 and boost.wind_n == 0.0


# --------------------------------------------------------------------------- #
# Live uncertainty inflation
# --------------------------------------------------------------------------- #
def test_uncertainty_never_below_phase_floor(cfg, origin, trained):
    tr = simulate(cfg, prefer="fallback", wind_east=4.0, wind_north=2.0, dry_mass=15.0)
    packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=5)
    pred = LandingPredictor(cfg, trained, origin)
    for pkt in packets:
        p = pred.process(pkt)
        out = p.to_dict()
        assert "raw_spread_m" in out and "base_sigma_m" in out
        # Radius is the calibrated phase floor inflated by live scatter -> never
        # tighter than the floor, and equal to it when scatter is zero.
        assert p.uncertainty_m >= out["base_sigma_m"] - 1e-6


# --------------------------------------------------------------------------- #
# REST API source (ARD's documented /telemetry/* endpoints)
# --------------------------------------------------------------------------- #
def test_adapter_reads_link_health_from_quality_block(origin):
    # ARD puts rssi / packet_id in a "quality" block, not in "packet".
    adapter = ArdTelemetryAdapter(origin)
    env = _envelope(origin, t_s=1.0, e=5.0, n=5.0, alt_agl=100.0)
    env["quality"] = {"rssi": -73.5, "packet_id": 42, "status": "OK"}
    pkt = adapter.to_packet(env)
    assert pkt.rssi == pytest.approx(-73.5)
    assert pkt.packet_id == 42


def _serve(routes):
    """Run a throwaway HTTP server exposing ARD-shaped JSON routes."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path not in routes:
                self.send_error(404)
                return
            body = _json.dumps(routes[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_rest_source_backfills_history_and_dedupes(origin):
    # Three frames of history, and /latest keeps returning the last one --
    # the source must yield each frame exactly once.
    history = [
        _envelope(origin, t_s=float(i), e=10.0 * i, n=5.0 * i, alt_agl=100.0 * i)
        for i in range(3)
    ]
    srv, url = _serve({
        "/health": {"status": "ok"},
        "/telemetry/history": {"success": True, "data": history},
        "/telemetry/latest": {"success": True, "data": history[-1]},
    })
    try:
        src = ArdRestSource(url=url, origin=origin, poll_hz=50.0)
        assert src.health() is True
        got = []
        for pkt in src:
            got.append(pkt)
            if len(got) >= 3:
                break
        assert [round(p.t, 3) for p in got] == [0.0, 1.0, 2.0]
    finally:
        srv.shutdown()


def test_rest_source_health_false_when_backend_absent(origin):
    src = ArdRestSource(url="http://127.0.0.1:9", origin=origin)
    assert src.health() is False
