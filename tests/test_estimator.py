"""Online estimator: phase detection, wind inference, apogee latch."""
from lze.estimator.state import OnlineStateEstimator
from lze.sim import PHASE_BOOST, PHASE_MAIN, simulate
from lze.telemetry.replay import trajectory_to_packets


def _run(cfg, origin, wind_e, wind_n):
    tr = simulate(cfg, prefer="fallback", wind_east=wind_e, wind_north=wind_n, dry_mass=15.0)
    packets = trajectory_to_packets(tr, origin, rate_hz=1.0, seed=1)
    est = OnlineStateEstimator(cfg, origin=origin)
    states = [est.update(p) for p in packets]
    return tr, states


def test_first_state_is_boost(cfg, origin):
    _, states = _run(cfg, origin, 0.0, 0.0)
    assert states[0].phase == PHASE_BOOST


def test_apogee_latches_and_reaches_main(cfg, origin):
    _, states = _run(cfg, origin, 0.0, 0.0)
    assert any(s.apogee_seen for s in states)
    assert states[-1].phase == PHASE_MAIN


def test_wind_estimate_converges(cfg, origin):
    # Under the main chute the airframe drifts with the air, so the estimator's
    # wind estimate should approach the true wind by touchdown.
    true_e, true_n = 6.0, -4.0
    _, states = _run(cfg, origin, true_e, true_n)
    final = states[-1]
    assert abs(final.wind_e - true_e) < 3.0
    assert abs(final.wind_n - true_n) < 3.0


def test_wind_zero_before_descent(cfg, origin):
    _, states = _run(cfg, origin, 6.0, -4.0)
    # During boost the estimator must not treat rocket motion as wind.
    boost = next(s for s in states if s.phase == PHASE_BOOST)
    assert boost.wind_e == 0.0 and boost.wind_n == 0.0
