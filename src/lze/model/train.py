"""Train and evaluate the landing-drift surrogate.

Splits the Monte-Carlo data by *flight* (not by sample) so a trajectory never
appears in both train and test -- otherwise neighbouring 1 Hz samples leak and
the metrics lie. Trains one gradient-boosted regressor per target and reports
error as a function of flight phase, which is the number that matters: the
prediction should get tighter from boost -> coast -> drogue -> main.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..sim.trajectory import PHASE_NAMES
from .features import build_feature_matrix, build_target_matrix
from .surrogate import Surrogate


@dataclass
class EvalReport:
    overall_rmse_m: float          # landing-point error (2-D) [m]
    overall_mae_m: float
    per_phase_mae_m: Dict[str, float]
    baseline_mae_m: float          # "assume it lands straight below" baseline
    n_test_samples: int

    def format(self) -> str:
        lines = [
            f"Test samples:        {self.n_test_samples:,}",
            f"Landing MAE:         {self.overall_mae_m:6.1f} m   "
            f"(baseline {self.baseline_mae_m:6.1f} m)",
            f"Landing RMSE:        {self.overall_rmse_m:6.1f} m",
            "Landing MAE by phase (should shrink toward landing):",
        ]
        for phase in ["boost", "coast", "drogue", "main"]:
            if phase in self.per_phase_mae_m:
                lines.append(f"    {phase:7s}: {self.per_phase_mae_m[phase]:6.1f} m")
        return "\n".join(lines)


def _split_by_flight(
    samples: np.ndarray, flight_id: np.ndarray, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks that keep whole flights together."""
    rng = np.random.default_rng(seed)
    flights = np.unique(flight_id)
    rng.shuffle(flights)
    n_test = max(1, int(round(len(flights) * test_frac)))
    test_flights = set(flights[:n_test].tolist())
    test_mask = np.array([f in test_flights for f in flight_id])
    return ~test_mask, test_mask


def _make_regressor() -> HistGradientBoostingRegressor:
    # Histogram GBDT: fast to train, tiny to store, sub-ms inference on a Pi.
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        min_samples_leaf=25,
        random_state=0,
    )


def train_surrogate(
    samples: np.ndarray,
    flight_id: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[Surrogate, EvalReport]:
    """Train the surrogate and evaluate landing-point error on held-out flights."""
    X = build_feature_matrix(samples)
    Y = build_target_matrix(samples)  # columns: rem_e, rem_n, rem_t

    train_mask, test_mask = _split_by_flight(samples, flight_id, test_frac, seed)

    models: Dict[str, object] = {}
    for j, target in enumerate(["rem_e", "rem_n", "rem_t"]):
        reg = _make_regressor()
        reg.fit(X[train_mask], Y[train_mask, j])
        models[target] = reg

    surrogate = Surrogate(
        models=models,
        metadata={
            "n_train_flights": int(len(np.unique(flight_id[train_mask]))),
            "n_test_flights": int(len(np.unique(flight_id[test_mask]))),
            "n_samples": int(len(samples)),
        },
    )

    report = _evaluate(surrogate, samples, X, Y, test_mask)
    return surrogate, report


def _evaluate(
    surrogate: Surrogate,
    samples: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    test_mask: np.ndarray,
) -> EvalReport:
    Xt = X[test_mask]
    st = samples[test_mask]
    pred = surrogate.predict_matrix(Xt)  # (N, 3)

    # Predicted landing = current position + predicted remaining offset.
    pred_land_e = st["e"] + pred[:, 0]
    pred_land_n = st["n"] + pred[:, 1]
    true_land_e = st["e"] + Y[test_mask, 0]
    true_land_n = st["n"] + Y[test_mask, 1]

    err = np.hypot(pred_land_e - true_land_e, pred_land_n - true_land_n)

    # Baseline: predict zero remaining drift (lands straight below current pos).
    base = np.hypot(Y[test_mask, 0], Y[test_mask, 1])

    per_phase: Dict[str, float] = {}
    for pid, name in PHASE_NAMES.items():
        m = st["phase"] == pid
        if m.any():
            per_phase[name] = float(np.mean(err[m]))

    return EvalReport(
        overall_rmse_m=float(np.sqrt(np.mean(err ** 2))),
        overall_mae_m=float(np.mean(err)),
        per_phase_mae_m=per_phase,
        baseline_mae_m=float(np.mean(base)),
        n_test_samples=int(test_mask.sum()),
    )


def flight_ids_from_trajectories(trajectories: List) -> np.ndarray:
    """Build a per-sample flight id array matching stacked sample order."""
    ids = []
    for i, tr in enumerate(trajectories):
        ids.extend([i] * len(tr))
    return np.array(ids, dtype=np.int32)
