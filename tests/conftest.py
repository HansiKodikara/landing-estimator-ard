"""Pytest fixtures + path setup so ``import lze`` works without installation."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lze.config import load_config  # noqa: E402
from lze.geo import Origin  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def origin(cfg):
    lat, lon, elev = cfg.site_origin
    return Origin(lat=lat, lon=lon, elevation=elev)
