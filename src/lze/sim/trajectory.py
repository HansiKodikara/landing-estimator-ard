"""Common flight-trajectory container shared by the RocketPy and fallback sims.

A :class:`Trajectory` is the "flight experience" the surrogate learns from. It
stores a time series of the rocket state in the launch-pad ENU frame plus the
landing point and event times, and knows how to turn itself into supervised
training samples of the form::

    current state  ->  remaining (east, north) offset to landing, remaining time

which is exactly what the live estimator needs (see project brief: "current
rocket state -> remaining offset to landing").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Flight phases. Ordered so `>=` comparisons express progression.
PHASE_BOOST = 0     # on the rail / motor burning
PHASE_COAST = 1     # unpowered ascent to apogee
PHASE_DROGUE = 2    # descending under drogue (or ballistic pre-main)
PHASE_MAIN = 3      # descending under main parachute
PHASE_NAMES = {
    PHASE_BOOST: "boost",
    PHASE_COAST: "coast",
    PHASE_DROGUE: "drogue",
    PHASE_MAIN: "main",
}


@dataclass
class Trajectory:
    """One simulated flight sampled on a uniform time grid.

    All positions/velocities are in the launch-pad ENU frame (metres, m/s):
    ``e`` = east, ``n`` = north, ``u`` = up (altitude AGL).
    """

    t: np.ndarray            # (N,) time since ignition [s]
    e: np.ndarray            # (N,) east position [m]
    n: np.ndarray            # (N,) north position [m]
    u: np.ndarray            # (N,) altitude AGL [m]
    ve: np.ndarray           # (N,) east velocity [m/s]
    vn: np.ndarray           # (N,) north velocity [m/s]
    vu: np.ndarray           # (N,) vertical velocity [m/s]
    phase: np.ndarray        # (N,) int phase id
    # Landing point in ENU (the label everything is measured against).
    landing_e: float
    landing_n: float
    # Event times [s].
    t_apogee: float
    t_landing: float
    apogee_alt: float
    # Per-flight conditions used to generate this trajectory (wind etc.).
    meta: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def wind_estimate(self) -> tuple[float, float]:
        """The (east, north) wind [m/s] that produced this flight, if known."""
        return self.meta.get("wind_e", 0.0), self.meta.get("wind_n", 0.0)

    def to_samples(self) -> "np.ndarray":
        """Return a structured array of (state, label) rows for training.

        Columns are defined by :data:`SAMPLE_DTYPE`. One row per time sample
        (excluding the final landing sample, which has zero remaining offset).
        """
        wind_e, wind_n = self.wind_estimate()
        rows = np.empty(len(self.t), dtype=SAMPLE_DTYPE)
        rows["t"] = self.t
        rows["e"] = self.e
        rows["n"] = self.n
        rows["u"] = self.u
        rows["ve"] = self.ve
        rows["vn"] = self.vn
        rows["vu"] = self.vu
        rows["phase"] = self.phase
        rows["wind_e"] = wind_e
        rows["wind_n"] = wind_n
        # Labels: remaining offset from the *current* position to landing.
        rows["rem_e"] = self.landing_e - self.e
        rows["rem_n"] = self.landing_n - self.n
        rows["rem_t"] = np.maximum(self.t_landing - self.t, 0.0)
        return rows


# Structured dtype for one training sample: features first, then labels.
SAMPLE_DTYPE = np.dtype(
    [
        # --- features (rocket state + inferred wind) ---
        ("t", "f4"),
        ("e", "f4"),
        ("n", "f4"),
        ("u", "f4"),
        ("ve", "f4"),
        ("vn", "f4"),
        ("vu", "f4"),
        ("phase", "i4"),
        ("wind_e", "f4"),
        ("wind_n", "f4"),
        # --- labels (what the surrogate predicts) ---
        ("rem_e", "f4"),
        ("rem_n", "f4"),
        ("rem_t", "f4"),
    ]
)


def stack_samples(trajectories: List[Trajectory]) -> np.ndarray:
    """Concatenate the training samples from many trajectories."""
    if not trajectories:
        return np.empty(0, dtype=SAMPLE_DTYPE)
    return np.concatenate([tr.to_samples() for tr in trajectories])
