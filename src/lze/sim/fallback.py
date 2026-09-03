"""Offline 3-DOF point-mass fallback simulator.

Used automatically when RocketPy is not installed (see
:data:`lze.sim.rocketpy_sim.ROCKETPY_AVAILABLE`). It integrates a point mass
through boost, coast, drogue and main phases with quadratic drag, a constant
wind field, and the same recovery events as the RocketPy model, so the training
pipeline and demos still run anywhere -- CI, a bare Raspberry Pi, etc.

It is deliberately simple (forward Euler, constant Cd), so it is *not* a
substitute for RocketPy when generating the real training set, but it produces
qualitatively correct landing drift for testing the surrogate/estimator/live
stack end to end.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..config import Config
from .trajectory import (
    PHASE_BOOST,
    PHASE_COAST,
    PHASE_DROGUE,
    PHASE_MAIN,
    Trajectory,
)

G = 9.80665
RHO0 = 1.225  # sea-level air density (kg/m^3)


def _air_density(alt_asl: float) -> float:
    """Exponential-atmosphere density approximation."""
    return RHO0 * math.exp(-alt_asl / 8500.0)


def simulate(
    cfg: Config,
    wind_east: float = 0.0,
    wind_north: float = 0.0,
    dry_mass: Optional[float] = None,
    drag_multiplier: float = 1.0,
    inclination: Optional[float] = None,
    heading: Optional[float] = None,
    dt: float = 0.05,
    out_dt: float = 0.5,
) -> Trajectory:
    """Integrate one flight and return a trajectory sampled every ``out_dt`` s."""
    _, _, elevation = cfg.site_origin
    r = cfg.rocket
    m = cfg.motor
    rail = cfg.rail
    rec = cfg.recovery

    inc = math.radians(float(inclination if inclination is not None else rail["inclination"]))
    hdg = math.radians(float(heading if heading is not None else rail["heading"]))
    dry = float(dry_mass if dry_mass is not None else r["dry_mass"])
    prop = float(m["propellant_mass"])
    burn = float(m["burn_time"])
    thrust = float(m["average_thrust"])
    area = math.pi * float(r["radius"]) ** 2
    cd = float(r["drag_coefficient"]) * drag_multiplier

    main_alt = float(rec["main"]["deploy_altitude_agl"])
    drogue_cd_s = float(rec["drogue"]["cd"]) * math.pi * (float(rec["drogue"]["diameter"]) / 2) ** 2
    main_cd_s = float(rec["main"]["cd"]) * math.pi * (float(rec["main"]["diameter"]) / 2) ** 2

    # Launch direction unit vector (ENU). heading measured clockwise from north.
    dir_e = math.cos(inc) * math.sin(hdg)
    dir_n = math.cos(inc) * math.cos(hdg)
    dir_u = math.sin(inc)

    # State.
    pos = np.zeros(3)          # e, n, u (AGL)
    vel = np.zeros(3)
    t = 0.0
    apogee_reached = False
    t_apogee = 0.0
    apogee_alt = 0.0
    max_alt = 0.0

    rec_t = [0.0]
    rec_pos = [pos.copy()]
    rec_vel = [vel.copy()]
    next_out = out_dt

    wind = np.array([wind_east, wind_north, 0.0])

    for _ in range(int(600 / dt)):
        alt_asl = pos[2] + elevation
        rho = _air_density(alt_asl)
        mass = dry + prop * max(0.0, 1.0 - t / burn) + 1.5  # + inert motor mass

        # Thrust along launch direction while on the rail / burning.
        f = np.zeros(3)
        if t < burn:
            f += thrust * np.array([dir_e, dir_n, dir_u])

        # Drag opposes airspeed (velocity relative to wind).
        airspeed_vec = vel - wind
        speed = np.linalg.norm(airspeed_vec)
        descending = vel[2] < 0 and apogee_reached
        if descending and pos[2] <= main_alt:
            cd_s = main_cd_s
        elif descending:
            cd_s = drogue_cd_s
        else:
            cd_s = cd * area
        if speed > 1e-6:
            f += -0.5 * rho * cd_s * speed * airspeed_vec

        # Gravity.
        f += np.array([0.0, 0.0, -mass * G])

        acc = f / mass
        vel = vel + acc * dt
        pos = pos + vel * dt
        t += dt

        if pos[2] > max_alt:
            max_alt = pos[2]
        if not apogee_reached and vel[2] <= 0 and t > burn:
            apogee_reached = True
            t_apogee = t
            apogee_alt = pos[2]

        if t >= next_out:
            rec_t.append(t)
            rec_pos.append(pos.copy())
            rec_vel.append(vel.copy())
            next_out += out_dt

        if apogee_reached and pos[2] <= 0.0:
            rec_t.append(t)
            rec_pos.append(np.array([pos[0], pos[1], 0.0]))
            rec_vel.append(vel.copy())
            break

    ts = np.array(rec_t)
    P = np.array(rec_pos)
    V = np.array(rec_vel)
    t_landing = ts[-1]

    phase = np.empty(len(ts), dtype=np.int32)
    for i, (tt, alt, vz) in enumerate(zip(ts, P[:, 2], V[:, 2])):
        if tt <= burn:
            phase[i] = PHASE_BOOST
        elif tt <= t_apogee:
            phase[i] = PHASE_COAST
        elif alt > main_alt:
            phase[i] = PHASE_DROGUE
        else:
            phase[i] = PHASE_MAIN

    return Trajectory(
        t=ts,
        e=P[:, 0],
        n=P[:, 1],
        u=np.maximum(P[:, 2], 0.0),
        ve=V[:, 0],
        vn=V[:, 1],
        vu=V[:, 2],
        phase=phase,
        landing_e=float(P[-1, 0]),
        landing_n=float(P[-1, 1]),
        t_apogee=t_apogee,
        t_landing=t_landing,
        apogee_alt=apogee_alt,
        meta={
            "wind_e": wind_east,
            "wind_n": wind_north,
            "dry_mass": dry,
            "drag_multiplier": drag_multiplier,
            "inclination": math.degrees(inc),
            "heading": math.degrees(hdg),
        },
    )
