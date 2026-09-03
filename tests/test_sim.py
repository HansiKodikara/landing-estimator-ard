"""Simulator sanity: physically plausible apogee, descent, and wind drift.

Uses the fallback 3-DOF engine so the tests run anywhere (no RocketPy needed).
"""
import numpy as np

from lze.sim import PHASE_MAIN, simulate


def test_fallback_apogee_in_range(cfg):
    tr = simulate(cfg, prefer="fallback", wind_east=0.0, wind_north=0.0, dry_mass=15.0)
    apogee_ft = tr.apogee_alt / 0.3048
    # Target is ~11k ft; allow a generous band for the simplified engine.
    assert 8000 < apogee_ft < 14000


def test_touchdown_speed_under_limit(cfg):
    tr = simulate(cfg, prefer="fallback", dry_mass=15.0)
    limit = cfg.recovery["touchdown_speed_limit"]
    assert abs(tr.vu[-1]) < limit + 2.0  # within tolerance of the 11 m/s req


def test_wind_pushes_landing_downwind(cfg):
    # Wind blowing toward +east should move the landing point east.
    calm = simulate(cfg, prefer="fallback", wind_east=0.0, dry_mass=15.0)
    windy = simulate(cfg, prefer="fallback", wind_east=8.0, dry_mass=15.0)
    assert windy.landing_e > calm.landing_e + 100


def test_phase_progression_present(cfg):
    tr = simulate(cfg, prefer="fallback", dry_mass=15.0)
    phases = set(np.unique(tr.phase).tolist())
    # Every flight should reach the main-chute phase before landing.
    assert PHASE_MAIN in phases
    assert tr.t_apogee > 0 and tr.t_landing > tr.t_apogee


def test_samples_labels_are_remaining_offset(cfg):
    tr = simulate(cfg, prefer="fallback", dry_mass=15.0)
    rows = tr.to_samples()
    # Remaining offset at the final sample should be ~0 (already at landing).
    assert abs(rows["rem_e"][-1]) < 1.0
    assert abs(rows["rem_n"][-1]) < 1.0
    # First sample's remaining offset points to the landing site.
    assert abs(rows["rem_e"][0] - (tr.landing_e - tr.e[0])) < 1e-3
