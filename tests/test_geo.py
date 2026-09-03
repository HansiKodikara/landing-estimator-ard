"""ENU <-> geodetic round-trip and haversine sanity."""
import math

from lze.geo import Origin, haversine


def test_enu_geo_roundtrip(origin):
    for e, n, u in [(0, 0, 0), (500, -300, 1200), (-1234, 987, 50)]:
        lat, lon, alt = origin.enu_to_geo(e, n, u)
        e2, n2, u2 = origin.geo_to_enu(lat, lon, alt)
        assert abs(e - e2) < 1e-3
        assert abs(n - n2) < 1e-3
        assert abs(u - u2) < 1e-6


def test_enu_directions(origin):
    # +east increases longitude, +north increases latitude.
    lat_e, lon_e, _ = origin.enu_to_geo(1000, 0, 0)
    lat_n, lon_n, _ = origin.enu_to_geo(0, 1000, 0)
    assert lon_e > origin.lon and abs(lat_e - origin.lat) < 1e-6
    assert lat_n > origin.lat and abs(lon_n - origin.lon) < 1e-6


def test_haversine_matches_enu(origin):
    lat, lon, _ = origin.enu_to_geo(300, 400, 0)  # 500 m away
    d = haversine(origin.lat, origin.lon, lat, lon)
    assert math.isclose(d, 500, rel_tol=1e-3)
