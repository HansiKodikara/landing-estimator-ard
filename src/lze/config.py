"""Configuration loading for the Kronos landing-zone estimator.

A thin, typed-ish wrapper around ``config/kronos.yaml`` so the rest of the code
can say ``cfg.rocket["dry_mass"]`` instead of digging through nested dicts, and
so there is a single place that knows where the config lives.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root = two levels up from this file (src/lze/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "kronos.yaml"
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Config:
    """Parsed Kronos configuration with convenience accessors."""

    raw: Dict[str, Any]
    path: Path

    # --- top-level sections -------------------------------------------------
    @property
    def launch_site(self) -> Dict[str, Any]:
        return self.raw["launch_site"]

    @property
    def rocket(self) -> Dict[str, Any]:
        return self.raw["rocket"]

    @property
    def motor(self) -> Dict[str, Any]:
        return self.raw["motor"]

    @property
    def fins(self) -> Dict[str, Any]:
        return self.raw["fins"]

    @property
    def nose_cone(self) -> Dict[str, Any]:
        return self.raw["nose_cone"]

    @property
    def recovery(self) -> Dict[str, Any]:
        return self.raw["recovery"]

    @property
    def rail(self) -> Dict[str, Any]:
        return self.raw["rail"]

    @property
    def environment(self) -> Dict[str, Any]:
        return self.raw["environment"]

    @property
    def telemetry(self) -> Dict[str, Any]:
        return self.raw["telemetry"]

    @property
    def monte_carlo(self) -> Dict[str, Any]:
        return self.raw["monte_carlo"]

    # --- derived values -----------------------------------------------------
    @property
    def site_origin(self) -> tuple[float, float, float]:
        """(lat, lon, elevation) of the launch pad -- the ENU frame origin."""
        s = self.launch_site
        return float(s["latitude"]), float(s["longitude"]), float(s["elevation"])


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from *path* (default: ``config/kronos.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=p)
