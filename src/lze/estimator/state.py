"""Online state estimator.

Consumes telemetry packets one at a time and maintains the best current estimate
of the rocket state that the surrogate needs: position (ENU), velocity, altitude,
flight phase, and an inferred wind vector. This is the "System estimates the
rocket's current state" box in the pipeline.

Design notes
------------
* **Altitude** fuses GPS and barometric height (baro is lower-noise short-term).
* **Phase** follows the CDR flight state machine (boost -> coast -> apogee ->
  drogue -> main), using known rocket constants (motor burn time, main-deploy
  altitude) plus the live velocity sign.
* **Wind** is inferred from horizontal motion. Under a parachute the airframe
  drifts with the air mass, so the horizontal velocity *is* the wind; we run an
  exponential moving average that only trusts descent samples. This is why the
  landing prediction sharpens after the chutes open -- the wind estimate that
  feeds the surrogate finally becomes accurate, matching the training signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..config import Config
from ..geo import Origin
from ..sim.trajectory import PHASE_BOOST, PHASE_COAST, PHASE_DROGUE, PHASE_MAIN
from ..telemetry.schema import TelemetryPacket


@dataclass
class EstimatedState:
    """Current best-estimate state, ready to be turned into model features."""

    t: float
    e: float            # east position [m]
    n: float            # north position [m]
    u: float            # altitude AGL [m]
    ve: float
    vn: float
    vu: float
    phase: int
    wind_e: float
    wind_n: float
    apogee_seen: bool
    max_alt: float

    def to_feature_state(self) -> Dict[str, float]:
        return {
            "u": self.u,
            "vu": self.vu,
            "ve": self.ve,
            "vn": self.vn,
            "phase": self.phase,
            "wind_e": self.wind_e,
            "wind_n": self.wind_n,
        }


class OnlineStateEstimator:
    """Stateful estimator; feed it packets via :meth:`update`."""

    def __init__(
        self,
        cfg: Config,
        origin: Optional[Origin] = None,
        wind_seed: Optional[Tuple[float, float]] = None,
    ):
        self.cfg = cfg
        lat, lon, elev = cfg.site_origin
        self.origin = origin or Origin(lat=lat, lon=lon, elevation=elev)

        self._burn_time = float(cfg.motor["burn_time"])
        self._main_alt = float(cfg.recovery["main"]["deploy_altitude_agl"])

        # Optional forecast wind seed (ENU m/s). When provided the estimator
        # starts with this wind instead of zero, so the *ascent* landing
        # predictions -- which otherwise assume no drift -- are less wild. Once
        # the chutes open, measured drift takes over (the EMA in _update_wind
        # relaxes the seed toward the true, air-coupled velocity). Seeding only
        # helps if the surrogate was *also* trained with a seed (see
        # sim.dataset.build_estimated_samples); otherwise leave it None to match
        # the default, wind=0-during-ascent model.
        self._wind_seed = wind_seed

        # Running state.
        self.max_alt = 0.0
        self.apogee_seen = False
        if wind_seed is not None:
            self.wind_e, self.wind_n = float(wind_seed[0]), float(wind_seed[1])
            self._wind_initialised = True
        else:
            self.wind_e = 0.0
            self.wind_n = 0.0
            self._wind_initialised = False
        self._last: Optional[EstimatedState] = None

    # --- wind inference -----------------------------------------------------
    def _update_wind(self, phase: int, ve: float, vn: float) -> None:
        # Trust horizontal velocity as wind only while descending under a chute.
        if phase == PHASE_MAIN:
            alpha = 0.35        # main: near-terminal, horizontal vel ~= wind
        elif phase == PHASE_DROGUE:
            alpha = 0.15        # drogue: noisier but still air-coupled
        else:
            return              # boost/coast horizontal motion is not wind
        if not self._wind_initialised:
            self.wind_e, self.wind_n = ve, vn
            self._wind_initialised = True
        else:
            self.wind_e = (1 - alpha) * self.wind_e + alpha * ve
            self.wind_n = (1 - alpha) * self.wind_n + alpha * vn

    # --- phase detection ----------------------------------------------------
    def _classify_phase(self, t: float, u: float, vu: float) -> int:
        if not self.apogee_seen:
            # Ascending. Distinguish boost from coast by motor burn time.
            if t <= self._burn_time:
                return PHASE_BOOST
            return PHASE_COAST
        # Descending.
        if u > self._main_alt:
            return PHASE_DROGUE
        return PHASE_MAIN

    # --- main entry point ---------------------------------------------------
    def update(self, pkt: TelemetryPacket) -> EstimatedState:
        e, n, _u_gps = self.origin.geo_to_enu(pkt.lat, pkt.lon, pkt.alt_gps)
        # Fuse altitude: weight low-noise baro more than GPS vertical.
        u = 0.7 * pkt.alt_baro_agl + 0.3 * max(0.0, _u_gps)

        if u > self.max_alt:
            self.max_alt = u
        # Apogee latch: vertical velocity has turned negative and we have fallen
        # a little from the peak (robust to noise), OR clearly descending.
        if not self.apogee_seen and pkt.vu < 0 and (self.max_alt - u) > 5.0:
            self.apogee_seen = True

        phase = self._classify_phase(pkt.t, u, pkt.vu)
        self._update_wind(phase, pkt.ve, pkt.vn)

        state = EstimatedState(
            t=pkt.t,
            e=e,
            n=n,
            u=u,
            ve=pkt.ve,
            vn=pkt.vn,
            vu=pkt.vu,
            phase=phase,
            wind_e=self.wind_e,
            wind_n=self.wind_n,
            apogee_seen=self.apogee_seen,
            max_alt=self.max_alt,
        )
        self._last = state
        return state
