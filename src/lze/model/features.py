"""Feature engineering for the landing-drift surrogate.

The surrogate predicts the **remaining** offset to landing from the *current*
rocket state, so the features deliberately exclude the absolute launch-relative
position (e, n): where the pad happens to be must not change "how much farther
will it drift from here". This keeps the model translation-invariant and makes
it generalise across launch sites.

The same feature builder is used offline (training, on a structured sample
array) and live (a single state dict from the estimator), guaranteeing the
column order matches.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..sim.trajectory import PHASE_BOOST, PHASE_COAST, PHASE_DROGUE, PHASE_MAIN

# Ordered feature names -- the contract between training and inference.
FEATURE_NAMES: List[str] = [
    "u",            # altitude AGL [m]
    "vu",           # vertical velocity [m/s]
    "ve",           # east velocity [m/s]
    "vn",           # north velocity [m/s]
    "horiz_speed",  # sqrt(ve^2 + vn^2) [m/s]
    "wind_e",       # estimated east wind [m/s]
    "wind_n",       # estimated north wind [m/s]
    "is_boost",     # phase one-hot
    "is_coast",
    "is_drogue",
    "is_main",
]

TARGET_NAMES: List[str] = ["rem_e", "rem_n", "rem_t"]


def _phase_onehot(phase: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "is_boost": (phase == PHASE_BOOST).astype(np.float32),
        "is_coast": (phase == PHASE_COAST).astype(np.float32),
        "is_drogue": (phase == PHASE_DROGUE).astype(np.float32),
        "is_main": (phase == PHASE_MAIN).astype(np.float32),
    }


def build_feature_matrix(samples: np.ndarray) -> np.ndarray:
    """Vectorised features from a structured sample array (see SAMPLE_DTYPE)."""
    onehot = _phase_onehot(samples["phase"])
    cols = {
        "u": samples["u"].astype(np.float32),
        "vu": samples["vu"].astype(np.float32),
        "ve": samples["ve"].astype(np.float32),
        "vn": samples["vn"].astype(np.float32),
        "horiz_speed": np.hypot(samples["ve"], samples["vn"]).astype(np.float32),
        "wind_e": samples["wind_e"].astype(np.float32),
        "wind_n": samples["wind_n"].astype(np.float32),
        **onehot,
    }
    return np.column_stack([cols[name] for name in FEATURE_NAMES])


def build_target_matrix(samples: np.ndarray) -> np.ndarray:
    """(N, 3) array of [rem_e, rem_n, rem_t] labels."""
    return np.column_stack([samples[name].astype(np.float32) for name in TARGET_NAMES])


def build_feature_vector(state: Dict[str, float]) -> np.ndarray:
    """Single feature row (1, D) from a live state dict.

    Required keys: ``u, vu, ve, vn, wind_e, wind_n, phase``.
    """
    phase = int(state["phase"])
    values = {
        "u": float(state["u"]),
        "vu": float(state["vu"]),
        "ve": float(state["ve"]),
        "vn": float(state["vn"]),
        "horiz_speed": float(np.hypot(state["ve"], state["vn"])),
        "wind_e": float(state.get("wind_e", 0.0)),
        "wind_n": float(state.get("wind_n", 0.0)),
        "is_boost": 1.0 if phase == PHASE_BOOST else 0.0,
        "is_coast": 1.0 if phase == PHASE_COAST else 0.0,
        "is_drogue": 1.0 if phase == PHASE_DROGUE else 0.0,
        "is_main": 1.0 if phase == PHASE_MAIN else 0.0,
    }
    return np.array([[values[name] for name in FEATURE_NAMES]], dtype=np.float32)
