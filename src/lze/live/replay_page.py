"""Bake a predicted flight into a standalone, self-contained HTML page.

Runs the live predictor over a replayed flight, records every prediction frame,
and embeds them into the dashboard as ``window.EMBEDDED_FLIGHT``. The result is
a single HTML file that animates the landing-zone converging with no server and
no internet -- ideal for sharing a demo or reviewing a past flight.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..config import REPO_ROOT
from ..geo import Origin
from ..telemetry.schema import TelemetryPacket
from .predictor import LandingPredictor

DASHBOARD_HTML = REPO_ROOT / "dashboard" / "index.html"


def build_replay_page(
    predictor: LandingPredictor,
    packets: List[TelemetryPacket],
    origin: Origin,
    truth_land_e: Optional[float] = None,
    truth_land_n: Optional[float] = None,
    speed: float = 6.0,
) -> str:
    """Return HTML with the full predicted flight embedded for playback."""
    frames = []
    for pkt in packets:
        pred = predictor.process(pkt)
        frames.append(pred.to_dict())

    flight = {"frames": frames, "speed": speed}
    if truth_land_e is not None and truth_land_n is not None:
        t_lat, t_lon, _ = origin.enu_to_geo(truth_land_e, truth_land_n, 0.0)
        flight["truth"] = {
            "e": truth_land_e,
            "n": truth_land_n,
            "lat": t_lat,
            "lon": t_lon,
        }

    html = DASHBOARD_HTML.read_text()
    inject = f"<script>window.EMBEDDED_FLIGHT = {json.dumps(flight)};</script>\n"
    # Insert just before the main dashboard script so EMBEDDED_FLIGHT exists.
    return html.replace("<script>\n/* =", inject + "<script>\n/* =", 1)


def write_replay_page(path: str | Path, html: str) -> None:
    Path(path).write_text(html)
