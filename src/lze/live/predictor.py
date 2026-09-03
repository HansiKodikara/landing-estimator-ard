"""Live landing-zone predictor.

Ties the pieces together::

    telemetry packet -> OnlineStateEstimator -> state -> Surrogate -> landing

For each packet it produces a :class:`LivePrediction`: the predicted landing
lat/lon, the remaining flight time, and an uncertainty radius. The radius is
taken from the surrogate's own held-out error for the current phase, so the
"likely recovery area" the dashboard draws is honestly calibrated -- wide during
boost, tight under the main chute.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..config import Config
from ..estimator.state import EstimatedState, OnlineStateEstimator
from ..geo import Origin, haversine
from ..model.surrogate import Surrogate
from ..sim.trajectory import PHASE_NAMES
from ..telemetry.schema import TelemetryPacket

# Fallback uncertainty (m) per phase if the model carries no eval metadata.
_DEFAULT_PHASE_SIGMA = {"boost": 300.0, "coast": 120.0, "drogue": 60.0, "main": 35.0}


@dataclass
class LivePrediction:
    t: float
    phase: str
    # Current rocket fix.
    cur_lat: float
    cur_lon: float
    cur_e: float
    cur_n: float
    alt_agl: float
    # Predicted landing.
    land_lat: float
    land_lon: float
    land_e: float
    land_n: float
    remaining_time_s: float
    uncertainty_m: float
    # Live wind estimate (m/s ENU) and downrange distance of predicted landing.
    wind_e: float
    wind_n: float
    downrange_m: float
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        d = self.__dict__.copy()
        d.pop("extras")
        d.update(self.extras)
        return d


class LandingPredictor:
    """Stateful live predictor. Call :meth:`process` with each packet."""

    def __init__(
        self,
        cfg: Config,
        surrogate: Surrogate,
        origin: Optional[Origin] = None,
        smoothing_alpha: float = 0.35,
        wind_seed: Optional[Tuple[float, float]] = None,
    ):
        self.cfg = cfg
        self.surrogate = surrogate
        lat, lon, elev = cfg.site_origin
        self.origin = origin or Origin(lat=lat, lon=lon, elevation=elev)
        self.estimator = OnlineStateEstimator(cfg, origin=self.origin, wind_seed=wind_seed)
        self._phase_sigma, self.sigma_calibrated = self._load_phase_sigma(surrogate)
        # EMA on the predicted landing point: damps per-frame model jitter so the
        # displayed recovery area glides toward the true point instead of hopping.
        self._alpha = smoothing_alpha
        self._sm_e: Optional[float] = None
        self._sm_n: Optional[float] = None
        # Recent *raw* (un-smoothed) landing points. Their scatter is a live,
        # data-driven read on how much the model is currently disagreeing with
        # itself -- used to inflate the recovery radius above its calibrated
        # per-phase floor when the estimate is visibly unsettled.
        self._recent: "deque[Tuple[float, float]]" = deque(maxlen=8)

    def _recent_spread(self) -> float:
        """RMS distance of recent raw landing points from their centroid [m]."""
        pts = self._recent
        if len(pts) < 3:
            return 0.0
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        var = sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / len(pts)
        return var ** 0.5

    @staticmethod
    def _load_phase_sigma(surrogate: Surrogate) -> Tuple[Dict[str, float], bool]:
        """Per-phase recovery radius, and whether it is actually calibrated.

        The honest radius is the model's own held-out error. If a model carries
        no evaluation metadata we fall back to hardcoded guesses -- which look
        identical on screen but mean nothing, so the caller is told and the
        operator warned. An uncalibrated circle is worse than no circle.
        """
        eval_meta = surrogate.metadata.get("eval", {})
        per_phase = eval_meta.get("per_phase_mae_m")
        if per_phase:
            return dict(per_phase), True
        print(
            "[predictor] WARNING: this model carries no evaluation metadata, so "
            "the recovery-zone radius is a hardcoded guess, NOT calibrated to "
            "measured error. Retrain with scripts/train_model.py to get an "
            "honest radius."
        )
        return dict(_DEFAULT_PHASE_SIGMA), False

    def process(self, pkt: TelemetryPacket) -> LivePrediction:
        state: EstimatedState = self.estimator.update(pkt)
        pred = self.surrogate.predict(state.to_feature_state())

        raw_land_e = state.e + pred.rem_e
        raw_land_n = state.n + pred.rem_n
        # Temporal smoothing (EMA). Trust new predictions a little more as the
        # rocket descends, since later states are closer to the training-tight
        # regime; but always keep some inertia to reject single-frame outliers.
        alpha = self._alpha if state.phase < 3 else min(0.6, self._alpha + 0.15)
        if self._sm_e is None:
            self._sm_e, self._sm_n = raw_land_e, raw_land_n
        else:
            self._sm_e = (1 - alpha) * self._sm_e + alpha * raw_land_e
            self._sm_n = (1 - alpha) * self._sm_n + alpha * raw_land_n
        land_e, land_n = self._sm_e, self._sm_n
        land_lat, land_lon, _ = self.origin.enu_to_geo(land_e, land_n, 0.0)

        phase_name = PHASE_NAMES.get(state.phase, "unknown")
        base_sigma = self._phase_sigma.get(phase_name, 100.0)
        # Inflate the calibrated per-phase error by the live scatter of recent
        # raw predictions (added in quadrature, so the radius only ever grows
        # above its honest floor -- it never claims to be tighter than the
        # held-out error says it should be).
        self._recent.append((raw_land_e, raw_land_n))
        spread = self._recent_spread()
        sigma = (base_sigma ** 2 + spread ** 2) ** 0.5

        return LivePrediction(
            t=state.t,
            phase=phase_name,
            cur_lat=pkt.lat,
            cur_lon=pkt.lon,
            cur_e=state.e,
            cur_n=state.n,
            alt_agl=state.u,
            land_lat=land_lat,
            land_lon=land_lon,
            land_e=land_e,
            land_n=land_n,
            remaining_time_s=pred.rem_t,
            uncertainty_m=sigma,
            wind_e=state.wind_e,
            wind_n=state.wind_n,
            downrange_m=(land_e ** 2 + land_n ** 2) ** 0.5,
            extras={
                "ve": state.ve,
                "vn": state.vn,
                "vu": state.vu,
                "rssi": pkt.rssi,
                "packet_id": float(pkt.packet_id),
                "raw_land_e": raw_land_e,
                "raw_land_n": raw_land_n,
                "raw_spread_m": spread,
                "base_sigma_m": base_sigma,
                # 0 => the radius on screen is a guess, not measured error.
                "sigma_calibrated": 1.0 if self.sigma_calibrated else 0.0,
            },
        )

    def error_against_truth(
        self, pred: LivePrediction, true_land_lat: float, true_land_lon: float
    ) -> float:
        """Great-circle distance (m) between predicted and true landing."""
        return haversine(pred.land_lat, pred.land_lon, true_land_lat, true_land_lon)
