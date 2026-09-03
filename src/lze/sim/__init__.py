"""Flight simulation package.

Exposes a single :func:`simulate` that uses RocketPy when available and falls
back to the built-in 3-DOF integrator otherwise, so callers do not need to know
which teacher produced the trajectory.
"""
from __future__ import annotations

from ..config import Config
from .rocketpy_sim import ROCKETPY_AVAILABLE
from .trajectory import (  # re-export
    PHASE_BOOST,
    PHASE_COAST,
    PHASE_DROGUE,
    PHASE_MAIN,
    PHASE_NAMES,
    SAMPLE_DTYPE,
    Trajectory,
    stack_samples,
)


def simulate(cfg: Config, prefer: str = "auto", **kwargs) -> Trajectory:
    """Simulate one flight.

    ``prefer`` selects the engine: ``"rocketpy"``, ``"fallback"``, or ``"auto"``
    (RocketPy if importable, else fallback). ``**kwargs`` are passed through
    (wind_east, wind_north, dry_mass, drag_multiplier, inclination, heading, dt).
    """
    if prefer == "fallback" or (prefer == "auto" and not ROCKETPY_AVAILABLE):
        from . import fallback

        # fallback uses out_dt for the output grid; map dt->out_dt if given.
        if "dt" in kwargs:
            kwargs["out_dt"] = kwargs.pop("dt")
        return fallback.simulate(cfg, **kwargs)

    from . import rocketpy_sim

    return rocketpy_sim.simulate(cfg, **kwargs)


def engine_name(prefer: str = "auto") -> str:
    """Human-readable name of the engine :func:`simulate` would use."""
    if prefer == "fallback" or (prefer == "auto" and not ROCKETPY_AVAILABLE):
        return "fallback-3dof"
    return "rocketpy"


__all__ = [
    "simulate",
    "engine_name",
    "ROCKETPY_AVAILABLE",
    "Trajectory",
    "stack_samples",
    "SAMPLE_DTYPE",
    "PHASE_BOOST",
    "PHASE_COAST",
    "PHASE_DROGUE",
    "PHASE_MAIN",
    "PHASE_NAMES",
]
