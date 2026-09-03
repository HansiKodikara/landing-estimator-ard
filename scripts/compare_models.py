#!/usr/bin/env python3
"""Benchmark candidate model families for the landing-drift surrogate.

Answers "why this model?" with numbers instead of assertion. Every candidate is
trained and scored **identically** to the shipped surrogate: the same features,
the same flight-grouped train/test split (so no 1 Hz samples leak between
train and test), and the same metric -- 2-D landing-point error on held-out
flights, broken down by flight phase.

    python scripts/compare_models.py --dataset data/dataset.npz

Also reports train time, single-sample inference latency and serialised size,
because the model has to run on a Raspberry Pi in the field.
"""
from __future__ import annotations

import argparse
import io
import time
from typing import Dict, List, Tuple

import _bootstrap  # noqa: F401
import joblib
import numpy as np

from lze.model.features import build_feature_matrix, build_target_matrix
from lze.model.train import _split_by_flight
from lze.sim.trajectory import PHASE_NAMES

PHASES = ["boost", "coast", "drogue", "main"]


def candidates() -> Dict[str, callable]:
    """Model families worth considering for small tabular regression."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    return {
        # "Predict zero remaining drift" -- the floor any model must beat.
        "baseline (zero drift)": lambda: DummyRegressor(strategy="constant", constant=0.0),
        "ridge (linear)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "k-nearest (k=10)": lambda: make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=10)
        ),
        "decision tree": lambda: DecisionTreeRegressor(min_samples_leaf=25, random_state=0),
        "random forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1
        ),
        "extra trees": lambda: ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1
        ),
        "neural net (MLP)": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128), max_iter=600, early_stopping=True,
                random_state=0,
            ),
        ),
        # The shipped choice.
        "hist gradient boosting": lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=25, random_state=0,
        ),
    }


def evaluate(
    make, X: np.ndarray, Y: np.ndarray, samples: np.ndarray,
    train_mask: np.ndarray, test_mask: np.ndarray,
) -> Dict[str, float]:
    """Train one model per target, score landing error on held-out flights."""
    models = []
    t0 = time.perf_counter()
    for j in range(3):                      # rem_e, rem_n, rem_t
        m = make()
        m.fit(X[train_mask], Y[train_mask, j])
        models.append(m)
    train_s = time.perf_counter() - t0

    pred = np.column_stack([m.predict(X[test_mask]) for m in models])
    st = samples[test_mask]
    # Predicted landing = current position + predicted remaining offset.
    err = np.hypot(
        (st["e"] + pred[:, 0]) - (st["e"] + Y[test_mask, 0]),
        (st["n"] + pred[:, 1]) - (st["n"] + Y[test_mask, 1]),
    )

    # Single-sample latency: what the Pi actually pays, once per telemetry frame.
    one = X[test_mask][:1]
    for m in models:
        m.predict(one)
    t0 = time.perf_counter()
    for _ in range(50):
        for m in models:
            m.predict(one)
    infer_ms = (time.perf_counter() - t0) / 50 * 1000

    buf = io.BytesIO()
    joblib.dump(models, buf)

    out = {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "train_s": train_s,
        "infer_ms": infer_ms,
        "size_mb": buf.getbuffer().nbytes / 1e6,
    }
    for pid, name in PHASE_NAMES.items():
        m = st["phase"] == pid
        out[name] = float(np.mean(err[m])) if m.any() else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/dataset.npz")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = np.load(args.dataset, allow_pickle=False)
    samples, flight_id = blob["samples"], blob["flight_id"]
    X, Y = build_feature_matrix(samples), build_target_matrix(samples)
    train_mask, test_mask = _split_by_flight(samples, flight_id, args.test_frac, args.seed)

    print(f"{len(samples):,} samples · {len(np.unique(flight_id))} flights "
          f"({len(np.unique(flight_id[test_mask]))} held out for test)\n")
    hdr = (f"{'model':24s} {'MAE':>8s} {'RMSE':>8s} | "
           + " ".join(f"{p:>8s}" for p in PHASES)
           + f" | {'train':>7s} {'infer':>8s} {'size':>7s}")
    print(hdr); print("-" * len(hdr))

    results: List[Tuple[str, Dict[str, float]]] = []
    for name, make in candidates().items():
        r = evaluate(make, X, Y, samples, train_mask, test_mask)
        results.append((name, r))
        print(f"{name:24s} {r['mae']:7.1f}m {r['rmse']:7.1f}m | "
              + " ".join(f"{r[p]:7.1f}m" for p in PHASES)
              + f" | {r['train_s']:6.1f}s {r['infer_ms']:7.2f}ms {r['size_mb']:6.1f}MB")

    best = min(results, key=lambda kv: kv[1]["main"])
    print(f"\nBest under the main chute (the number that decides recovery): "
          f"{best[0]} at {best[1]['main']:.1f} m")


if __name__ == "__main__":
    main()
