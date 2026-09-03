"""RocketPy flight simulation of Project Kronos.

This is the physics "teacher". It builds a RocketPy model of the Kronos vehicle
from :mod:`lze.config` and flies it under a chosen wind, returning a uniformly
sampled :class:`~lze.sim.trajectory.Trajectory` in the launch-pad ENU frame.

The surrogate model never runs RocketPy in flight -- that is the whole point.
RocketPy is used *offline* to generate the training data; the Raspberry Pi then
runs only the lightweight learned model.
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

ROCKETPY_AVAILABLE = True
try:  # pragma: no cover - exercised by environment, not unit tests
    from rocketpy import Environment, Flight, Rocket, SolidMotor
except Exception:  # noqa: BLE001 - any import failure => use fallback
    ROCKETPY_AVAILABLE = False


def _chute_cd_s(cd: float, diameter: float) -> float:
    """Parachute drag area Cd*S from coefficient and canopy diameter."""
    return cd * math.pi * (diameter / 2.0) ** 2


def build_environment(cfg: Config, wind_east: float, wind_north: float):
    """Construct a RocketPy Environment with a constant wind field."""
    lat, lon, elev = cfg.site_origin
    env = Environment(latitude=lat, longitude=lon, elevation=elev)
    # A constant custom atmosphere: wind_u toward east, wind_v toward north.
    env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=wind_east,
        wind_v=wind_north,
    )
    return env


def build_motor(cfg: Config, dry_mass_override: Optional[float] = None) -> "SolidMotor":
    """AeroTech M1845NT approximation as a RocketPy SolidMotor.

    We approximate the thrust curve as constant average thrust over the burn,
    which reproduces the published total impulse (8308 N*s). That is plenty for
    generating landing-drift training data -- the descent (which dominates the
    landing point) is insensitive to the exact boost thrust shape.
    """
    m = cfg.motor
    return SolidMotor(
        thrust_source=float(m["average_thrust"]),
        dry_mass=1.5,  # inert motor hardware mass (kg)
        dry_inertia=(0.05, 0.05, 0.005),
        nozzle_radius=float(m["nozzle_radius"]),
        grain_number=int(m["grain_number"]),
        grain_density=float(m["grain_density"]),
        grain_outer_radius=float(m["grain_outer_radius"]),
        grain_initial_inner_radius=float(m["grain_initial_inner_radius"]),
        grain_initial_height=float(m["grain_initial_height"]),
        grain_separation=float(m["grain_separation"]),
        grains_center_of_mass_position=0.0,
        center_of_dry_mass_position=0.0,
        nozzle_position=-0.35,
        burn_time=float(m["burn_time"]),
        throat_radius=float(m["throat_radius"]),
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def build_rocket(
    cfg: Config,
    motor: "SolidMotor",
    dry_mass: Optional[float] = None,
    drag_multiplier: float = 1.0,
) -> "Rocket":
    """Assemble the Kronos rocket: nose, fins, motor, drogue + main chutes."""
    r = cfg.rocket
    fins = cfg.fins
    nose = cfg.nose_cone
    rec = cfg.recovery

    dry = float(dry_mass if dry_mass is not None else r["dry_mass"])
    rocket = Rocket(
        radius=float(r["radius"]),
        mass=dry,
        inertia=(float(r["inertia_i"]), float(r["inertia_i"]), float(r["inertia_z"])),
        power_off_drag=float(r["drag_coefficient"]) * drag_multiplier,
        power_on_drag=float(r["drag_coefficient_power_on"]) * drag_multiplier,
        center_of_mass_without_motor=float(r["center_of_mass_without_motor"]),
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=-1.4)
    rocket.add_nose(length=float(nose["length"]), kind=str(nose["kind"]), position=0.0)
    rocket.add_trapezoidal_fins(
        n=int(fins["n"]),
        root_chord=float(fins["root_chord"]),
        tip_chord=float(fins["tip_chord"]),
        span=float(fins["span"]),
        sweep_length=float(fins["sweep_length"]),
        position=-float(fins["position"]),
    )
    # Recovery: drogue at apogee, main at a set AGL altitude during descent.
    rocket.add_parachute(
        name="drogue",
        cd_s=_chute_cd_s(float(rec["drogue"]["cd"]), float(rec["drogue"]["diameter"])),
        trigger="apogee",
    )
    rocket.add_parachute(
        name="main",
        cd_s=_chute_cd_s(float(rec["main"]["cd"]), float(rec["main"]["diameter"])),
        trigger=float(rec["main"]["deploy_altitude_agl"]),
    )
    return rocket


def simulate(
    cfg: Config,
    wind_east: float = 0.0,
    wind_north: float = 0.0,
    dry_mass: Optional[float] = None,
    drag_multiplier: float = 1.0,
    inclination: Optional[float] = None,
    heading: Optional[float] = None,
    dt: float = 0.5,
) -> Trajectory:
    """Fly one Kronos mission and return a uniformly sampled trajectory.

    Parameters mirror the Monte-Carlo dataset knobs (wind, mass, drag, rail
    angle). ``dt`` is the output sampling interval (s); the surrogate is trained
    on ~1-2 Hz samples to match the telemetry downlink rate.
    """
    if not ROCKETPY_AVAILABLE:
        raise RuntimeError("RocketPy is not available; use lze.sim.fallback instead")

    _, _, elevation = cfg.site_origin
    rail = cfg.rail
    inc = float(inclination if inclination is not None else rail["inclination"])
    hdg = float(heading if heading is not None else rail["heading"])

    env = build_environment(cfg, wind_east, wind_north)
    motor = build_motor(cfg)
    rocket = build_rocket(cfg, motor, dry_mass=dry_mass, drag_multiplier=drag_multiplier)

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=float(rail["length"]),
        inclination=inc,
        heading=hdg,
        max_time=600,
        terminate_on_apogee=False,
    )

    t_final = float(flight.t_final)
    t_apogee = float(flight.apogee_time)
    burn_out = float(motor.burn_out_time)
    main_alt = float(cfg.recovery["main"]["deploy_altitude_agl"])

    # Uniform time grid from ignition to touchdown.
    n = max(2, int(math.ceil(t_final / dt)) + 1)
    ts = np.linspace(0.0, t_final, n)

    e = np.array([flight.x(t) for t in ts], dtype=float)
    nn = np.array([flight.y(t) for t in ts], dtype=float)
    z_asl = np.array([flight.z(t) for t in ts], dtype=float)
    u = z_asl - elevation  # AGL
    ve = np.array([flight.vx(t) for t in ts], dtype=float)
    vn = np.array([flight.vy(t) for t in ts], dtype=float)
    vu = np.array([flight.vz(t) for t in ts], dtype=float)

    phase = _classify_phase(ts, u, vu, t_apogee, burn_out, main_alt)

    return Trajectory(
        t=ts,
        e=e,
        n=nn,
        u=np.maximum(u, 0.0),
        ve=ve,
        vn=vn,
        vu=vu,
        phase=phase,
        landing_e=float(flight.x_impact),
        landing_n=float(flight.y_impact),
        t_apogee=t_apogee,
        t_landing=t_final,
        apogee_alt=float(flight.apogee) - elevation,
        meta={
            "wind_e": wind_east,
            "wind_n": wind_north,
            "dry_mass": float(dry_mass if dry_mass is not None else cfg.rocket["dry_mass"]),
            "drag_multiplier": drag_multiplier,
            "inclination": inc,
            "heading": hdg,
        },
    )


def _classify_phase(ts, u, vu, t_apogee, burn_out, main_alt) -> np.ndarray:
    """Assign a flight phase id to each sample."""
    phase = np.empty(len(ts), dtype=np.int32)
    for i, (t, alt, vz) in enumerate(zip(ts, u, vu)):
        if t <= burn_out:
            phase[i] = PHASE_BOOST
        elif t <= t_apogee:
            phase[i] = PHASE_COAST
        elif alt > main_alt:
            phase[i] = PHASE_DROGUE
        else:
            phase[i] = PHASE_MAIN
    return phase
