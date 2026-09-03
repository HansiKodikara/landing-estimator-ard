"""The surrogate model: a fast learned approximation of RocketPy's landing drift.

Given the current rocket state (+ inferred wind), it predicts the **remaining**
east/north offset to the landing point and the remaining flight time. On the
Raspberry Pi this replaces a full physics re-simulation every second with a
single model evaluation of a few milliseconds -- comfortably inside the 1 Hz
telemetry budget (see ``scripts/check_model.py``, which measures it in place).

The model is a small set of scikit-learn regressors wrapped so that inference is
a single :meth:`predict` call returning a :class:`Prediction`.

The artifact written by :meth:`Surrogate.save` is self-describing: it carries the
feature contract it was trained against and the library versions that built it,
so the ground station can prove -- before launch -- that the file it is about to
fly matches the code loading it.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import sklearn

from .features import FEATURE_NAMES, TARGET_NAMES, build_feature_vector

# Canonical state used to verify a model behaves identically on another machine
# (see scripts/check_model.py): mid-descent under the main chute in a westerly.
REFERENCE_STATE: Dict[str, float] = {
    "u": 300.0, "vu": -6.5, "ve": 7.0, "vn": -3.0,
    "phase": 3, "wind_e": 7.0, "wind_n": -3.0,
}


@dataclass
class Prediction:
    """Surrogate output for one state."""

    rem_e: float        # remaining east offset to landing [m]
    rem_n: float        # remaining north offset to landing [m]
    rem_t: float        # remaining flight time [s]


class Surrogate:
    """Wraps trained regressors + metadata; handles save/load and inference."""

    def __init__(self, models: Dict[str, object], metadata: Dict | None = None):
        # One regressor per target (rem_e, rem_n, rem_t).
        self.models = models
        self.metadata = metadata or {}
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.target_names: List[str] = list(TARGET_NAMES)

    # --- inference ----------------------------------------------------------
    def predict(self, state: Dict[str, float]) -> Prediction:
        """Predict remaining offset/time for a single live state dict."""
        x = build_feature_vector(state)
        return Prediction(
            rem_e=float(self.models["rem_e"].predict(x)[0]),
            rem_n=float(self.models["rem_n"].predict(x)[0]),
            rem_t=float(max(0.0, self.models["rem_t"].predict(x)[0])),
        )

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """Batch predict on a prebuilt feature matrix -> (N, 3)."""
        return np.column_stack([self.models[t].predict(X) for t in self.target_names])

    # --- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise models + metadata to a single portable artifact.

        Stamps the training environment and the feature contract into the file
        so the ground station can verify, before launch, that the model it is
        about to fly is compatible with the code loading it.
        """
        meta = dict(self.metadata)
        meta["feature_names"] = list(self.feature_names)
        meta["env"] = {
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "model_class": type(next(iter(self.models.values()))).__name__,
        }
        joblib.dump({"models": self.models, "metadata": meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Surrogate":
        blob = joblib.load(path)
        sur = cls(models=blob["models"], metadata=blob.get("metadata", {}))
        sur.check_compatibility()
        return sur

    def check_compatibility(self) -> List[str]:
        """Validate a loaded model against the running code. Returns warnings.

        A **feature mismatch is fatal** -- the model would silently predict from
        the wrong columns, which is worse than not predicting at all. A library
        version difference is only a warning: pickled scikit-learn estimators
        usually load across minor versions, but not always, so the operator is
        told rather than blocked.
        """
        trained_features = self.metadata.get("feature_names")
        if trained_features is not None and list(trained_features) != list(FEATURE_NAMES):
            raise ValueError(
                "Surrogate feature mismatch -- this model was trained on a "
                f"different feature set and cannot be used.\n"
                f"  model expects: {list(trained_features)}\n"
                f"  code provides: {list(FEATURE_NAMES)}\n"
                "Retrain with scripts/train_model.py."
            )

        warnings: List[str] = []
        env = self.metadata.get("env") or {}
        if env:
            for lib, current in (("sklearn", sklearn.__version__), ("numpy", np.__version__)):
                trained = env.get(lib)
                if trained and trained.split(".")[:2] != current.split(".")[:2]:
                    warnings.append(
                        f"{lib} version differs: model trained with {trained}, "
                        f"running {current}. Predictions should still work, but "
                        f"verify with scripts/check_model.py."
                    )
        return warnings
