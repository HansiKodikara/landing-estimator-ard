"""Geodetic <-> local ENU conversions.

The surrogate model and the physics simulator all work in a local **ENU**
(East, North, Up) tangent-plane frame centred on the launch pad, in metres.
Telemetry arrives as GPS lat/lon/alt, and the dashboard wants lat/lon back, so
these helpers convert between the two.

For the ~few-km scales of a hobby rocket flight an equirectangular (flat-earth)
approximation about the launch latitude is accurate to well under a metre,
which is far below GPS noise, so we do not need a full ECEF transform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# WGS-84 mean earth radius (m).
EARTH_RADIUS = 6_371_000.0


@dataclass(frozen=True)
class Origin:
    """Launch-pad origin of the local ENU frame."""

    lat: float          # deg
    lon: float          # deg
    elevation: float    # m ASL

    @property
    def _meters_per_deg_lat(self) -> float:
        return math.pi * EARTH_RADIUS / 180.0

    @property
    def _meters_per_deg_lon(self) -> float:
        return math.pi * EARTH_RADIUS * math.cos(math.radians(self.lat)) / 180.0

    def enu_to_geo(self, east: float, north: float, up: float = 0.0):
        """ENU offset (m) -> (lat, lon, altitude ASL)."""
        lat = self.lat + north / self._meters_per_deg_lat
        lon = self.lon + east / self._meters_per_deg_lon
        return lat, lon, self.elevation + up

    def geo_to_enu(self, lat: float, lon: float, alt: float = 0.0):
        """(lat, lon, altitude ASL) -> ENU offset (m)."""
        east = (lon - self.lon) * self._meters_per_deg_lon
        north = (lat - self.lat) * self._meters_per_deg_lat
        up = alt - self.elevation
        return east, north, up


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))


def wind_vector(speed: float, heading_from_deg: float) -> tuple[float, float]:
    """Meteorological wind ``(speed, FROM-heading)`` -> ENU ``(east, north)`` m/s.

    Wind headings are quoted as the direction the wind blows *from* (a "westerly"
    is 270 deg and blows toward the east). This returns the velocity vector of
    the air mass in the local ENU frame -- the same convention used everywhere
    the surrogate reasons about wind (dataset generation, the estimator's
    inferred wind, and the optional forecast seed).
    """
    d = math.radians(heading_from_deg)
    return -speed * math.sin(d), -speed * math.cos(d)
