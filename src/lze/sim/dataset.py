"""Monte-Carlo dataset generation from the flight simulator.

Runs many flights with randomised launch conditions (wind, rail angle, mass,
drag) and turns every trajectory into supervised samples of
``current state -> remaining offset to landing``. Because each flight
contributes samples from every phase (boost through touchdown), the surrogate
learns how the prediction *should* tighten as the rocket descends -- the core
behaviour of a live recovery assistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..config import Config
from ..geo import Origin
from . import simulate as _simulate
from .trajectory import SAMPLE_DTYPE, Trajectory


@dataclass
class DatasetResult:
    samples: np.ndarray            # structured array (SAMPLE_DTYPE)
    flight_id: np.ndarray          # (N,) per-sample flight index (for grouped split)
    trajectories: List[Trajectory]  # kept for evaluation / plotting
    engine: str


def build_estimated_samples(
    tr: Trajectory,
    cfg: Config,
    origin: Origin,
    seed: int = 0,
    forecast_wind_noise: Optional[float] = None,
) -> np.ndarray:
    """Turn a trajectory into training rows *as the live estimator would see it*.

    Replays the flight as noisy telemetry, runs :class:`OnlineStateEstimator`,
    and pairs each **estimated** state (noisy kinematics, causal wind estimate,
    detected phase) with the **true** remaining offset to landing. Training on
    these rows eliminates train/serve skew: the model learns to predict from
    exactly the information available in flight, so its accuracy genuinely
    improves as the estimator's wind/phase knowledge improves.

    If ``forecast_wind_noise`` is set (m/s std), the estimator is *seeded* with
    this flight's true wind perturbed by that much Gaussian noise -- simulating
    an imperfect launch-day forecast. The surrogate then learns to use an
    ascent-time wind hint, matching a live run that seeds the same forecast.
    Leave it ``None`` (default) for the standard model that assumes zero wind
    until the chutes reveal it.
    """
    # Local imports avoid a package import cycle (estimator/telemetry sit above
    # sim in the dependency graph).
    from ..estimator.state import OnlineStateEstimator
    from ..telemetry.replay import trajectory_to_packets

    tele = cfg.telemetry
    packets = trajectory_to_packets(
        tr,
        origin,
        rate_hz=float(tele["rate_hz"]),
        gps_h_noise=float(tele["gps_horizontal_noise_m"]),
        gps_v_noise=float(tele["gps_vertical_noise_m"]),
        baro_noise=float(tele["baro_noise_m"]),
        vel_noise=float(tele["velocity_noise_ms"]),
        seed=seed,
    )
    wind_seed = None
    if forecast_wind_noise is not None:
        true_we, true_wn = tr.wind_estimate()
        rng = np.random.default_rng(seed ^ 0x5EED)
        wind_seed = (
            float(true_we + rng.normal(0.0, forecast_wind_noise)),
            float(true_wn + rng.normal(0.0, forecast_wind_noise)),
        )
    est = OnlineStateEstimator(cfg, origin=origin, wind_seed=wind_seed)

    rows = np.empty(len(packets), dtype=SAMPLE_DTYPE)
    for i, pkt in enumerate(packets):
        s = est.update(pkt)
        rows["t"][i] = s.t
        rows["e"][i] = s.e
        rows["n"][i] = s.n
        rows["u"][i] = s.u
        rows["ve"][i] = s.ve
        rows["vn"][i] = s.vn
        rows["vu"][i] = s.vu
        rows["phase"][i] = s.phase
        rows["wind_e"][i] = s.wind_e
        rows["wind_n"][i] = s.wind_n
        # Labels: TRUE remaining offset (physical), measured from the true
        # position at this time so predicted = live_pos + rem lands correctly.
        true_e = float(np.interp(s.t, tr.t, tr.e))
        true_n = float(np.interp(s.t, tr.t, tr.n))
        rows["rem_e"][i] = tr.landing_e - true_e
        rows["rem_n"][i] = tr.landing_n - true_n
        rows["rem_t"][i] = max(0.0, tr.t_landing - s.t)
    return rows


def generate_dataset(
    cfg: Config,
    n_flights: Optional[int] = None,
    prefer: str = "auto",
    seed: int = 0,
    dt: float = 1.0,
    progress: Optional[Callable[[int, int, Trajectory], None]] = None,
    forecast_wind_noise: Optional[float] = None,
) -> DatasetResult:
    """Generate a training dataset by Monte-Carlo over launch conditions.

    ``forecast_wind_noise`` (m/s std, optional) trains a forecast-seeded model:
    each flight's estimator is seeded with its true wind plus this much noise.
    See :func:`build_estimated_samples`.
    """
    mc = cfg.monte_carlo
    rng = np.random.default_rng(seed)
    n = int(n_flights if n_flights is not None else mc["n_flights"])
    lat, lon, elev = cfg.site_origin
    origin = Origin(lat=lat, lon=lon, elevation=elev)

    trajectories: List[Trajectory] = []
    sample_blocks: List[np.ndarray] = []
    for i in range(n):
        wind_speed = rng.uniform(*mc["wind_speed_range"])
        wind_dir = np.radians(rng.uniform(*mc["wind_heading_range"]))  # FROM direction
        # Convert meteorological "from" heading into an ENU "toward" vector.
        wind_e = -wind_speed * np.sin(wind_dir)
        wind_n = -wind_speed * np.cos(wind_dir)
        dry_mass = rng.uniform(*mc["dry_mass_range"])
        drag_mult = rng.uniform(*mc["drag_multiplier_range"])
        inclination = rng.uniform(*mc["rail_inclination_range"])
        heading = rng.uniform(0.0, 360.0)

        tr = _simulate(
            cfg,
            prefer=prefer,
            wind_east=float(wind_e),
            wind_north=float(wind_n),
            dry_mass=float(dry_mass),
            drag_multiplier=float(drag_mult),
            inclination=float(inclination),
            heading=float(heading),
            dt=dt,
        )
        trajectories.append(tr)
        sample_blocks.append(
            build_estimated_samples(
                tr, cfg, origin,
                seed=seed * 100000 + i,
                forecast_wind_noise=forecast_wind_noise,
            )
        )
        if progress is not None:
            progress(i + 1, n, tr)

    samples = (
        np.concatenate(sample_blocks) if sample_blocks else np.empty(0, dtype=SAMPLE_DTYPE)
    )
    flight_id = (
        np.concatenate([np.full(len(b), i, dtype=np.int32) for i, b in enumerate(sample_blocks)])
        if sample_blocks
        else np.empty(0, dtype=np.int32)
    )
    from . import engine_name

    return DatasetResult(
        samples=samples,
        flight_id=flight_id,
        trajectories=trajectories,
        engine=engine_name(prefer),
    )
