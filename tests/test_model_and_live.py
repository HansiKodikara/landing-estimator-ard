"""Surrogate training + full live pipeline behaviour.

These are the tests that pin the core promise of the project: the predicted
landing zone must get *tighter* as the flight progresses. They train a small
surrogate on the fly with the fallback engine so they need no committed model.
"""
import numpy as np
import pytest

from lze.geo import Origin
from lze.live.predictor import LandingPredictor
from lze.model.features import build_feature_vector, FEATURE_NAMES
from lze.model.surrogate import Surrogate
from lze.model.train import train_surrogate
from lze.sim import simulate
from lze.sim.dataset import generate_dataset
from lze.telemetry.replay import trajectory_to_packets
from lze.telemetry.schema import TelemetryPacket


@pytest.fixture(scope="module")
def trained(cfg):
    # Small, fast dataset with the offline engine.
    res = generate_dataset(cfg, n_flights=12, prefer="fallback", seed=0, dt=0.5)
    surrogate, report = train_surrogate(res.samples, res.flight_id, test_frac=0.25, seed=0)
    return surrogate, report


def test_feature_vector_matches_schema():
    state = dict(u=1000, vu=-20, ve=5, vn=-3, phase=2, wind_e=5, wind_n=-3)
    x = build_feature_vector(state)
    assert x.shape == (1, len(FEATURE_NAMES))


def test_training_beats_baseline(trained):
    _, report = trained
    # The learned model must clearly beat "assume zero remaining drift".
    # (This is a tiny 12-flight fallback model; the committed RocketPy model is
    # far stronger -- see README benchmarks.)
    assert report.overall_mae_m < report.baseline_mae_m * 0.8


def test_prediction_tightens_through_flight(cfg, trained):
    surrogate, _ = trained
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)

    # Average landing error per phase over several fresh flights.
    rng = np.random.default_rng(99)
    from collections import defaultdict
    errs = defaultdict(list)
    for _ in range(6):
        ws = rng.uniform(3, 9)
        wd = np.radians(rng.uniform(0, 360))
        tr = simulate(cfg, prefer="fallback",
                      wind_east=float(-ws * np.sin(wd)), wind_north=float(-ws * np.cos(wd)),
                      dry_mass=float(rng.uniform(14.5, 16.0)))
        tlat, tlon, _ = origin.enu_to_geo(tr.landing_e, tr.landing_n, 0.0)
        packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=int(rng.integers(1e6)))
        pred = LandingPredictor(cfg, surrogate, origin)
        for pkt in packets:
            p = pred.process(pkt)
            errs[p.phase].append(pred.error_against_truth(p, tlat, tlon))

    main_mae = np.mean(errs["main"])
    drogue_mae = np.mean(errs["drogue"])
    coast_mae = np.mean(errs["coast"])
    # The core promise: the recovery area tightens as the flight progresses.
    # Under the main chute it must be clearly tighter than under drogue, which in
    # turn must beat the high-uncertainty coast phase.
    assert main_mae < drogue_mae < coast_mae
    # Absolute bound is loose here because this is a 12-flight fallback model;
    # the committed RocketPy model lands this near ~45 m (see README).
    assert main_mae < 250.0


def test_predictor_output_is_serialisable(cfg, trained):
    surrogate, _ = trained
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat, lon, elev)
    pred = LandingPredictor(cfg, surrogate, origin)
    pkt = TelemetryPacket(t=1.0, lat=lat, lon=lon, alt_gps=elev + 100,
                          alt_baro_agl=100, ve=2, vn=1, vu=80, packet_id=1)
    out = pred.process(pkt).to_dict()
    for key in ["land_lat", "land_lon", "uncertainty_m", "cur_e", "cur_n", "phase"]:
        assert key in out


def test_saved_model_roundtrips_with_provenance(trained, tmp_path):
    surrogate, _ = trained
    path = tmp_path / "surrogate.joblib"
    surrogate.save(path)
    again = Surrogate.load(path)
    env = again.metadata["env"]
    for key in ["sklearn", "numpy", "python", "saved_utc", "model_class"]:
        assert env.get(key)
    # Same input must give the same answer after a save/load round trip.
    state = dict(u=300, vu=-6.5, ve=7, vn=-3, phase=3, wind_e=7, wind_n=-3)
    before, after = surrogate.predict(state), again.predict(state)
    assert before.rem_e == pytest.approx(after.rem_e)
    assert before.rem_n == pytest.approx(after.rem_n)


def test_load_refuses_model_with_stale_feature_contract(trained, tmp_path):
    # A model trained on different columns would silently predict from the
    # wrong inputs -- loading it must fail loudly instead.
    import joblib

    surrogate, _ = trained
    path = tmp_path / "stale.joblib"
    surrogate.save(path)
    blob = joblib.load(path)
    blob["metadata"]["feature_names"] = ["u", "vu"]
    joblib.dump(blob, path)
    with pytest.raises(ValueError, match="feature mismatch"):
        Surrogate.load(path)


def test_library_version_difference_warns_but_loads(trained, tmp_path):
    import joblib

    surrogate, _ = trained
    path = tmp_path / "oldver.joblib"
    surrogate.save(path)
    blob = joblib.load(path)
    blob["metadata"]["env"]["sklearn"] = "0.1.0"
    joblib.dump(blob, path)
    loaded = Surrogate.load(path)            # must not raise
    assert any("sklearn" in w for w in loaded.check_compatibility())


def test_hub_replays_history_to_late_subscriber():
    # A browser that connects (or refreshes) mid-flight must receive the whole
    # flight so far, not a stream that starts in mid-air.
    from lze.live.server import _Hub

    hub = _Hub()
    for i in range(5):
        hub.publish(f"frame-{i}")
    q = hub.subscribe()
    got = [q.get_nowait() for _ in range(5)]
    assert got == [f"frame-{i}" for i in range(5)]
    hub.publish("frame-5")
    assert q.get_nowait() == "frame-5"


def test_telemetry_packet_json_roundtrip():
    pkt = TelemetryPacket(t=5, lat=-30.8, lon=143.1, alt_gps=1000, alt_baro_agl=900,
                          ve=3, vn=-2, vu=-8, packet_id=7, rssi=-70)
    again = TelemetryPacket.from_json(pkt.to_json())
    assert again == pkt
