import math

import numpy as np

from byakugan_app.math import geometry, geodesy


def test_pixel_angle_roundtrip():
    width, height = 1024, 512
    for u in [0, width // 4, width // 2, width - 1]:
        for v in [0, height // 4, height // 2, height - 1]:
            theta, phi = geometry.pixel_to_angles(u, v, width, height)
            u2, v2 = geometry.angles_to_pixel(theta, phi, width, height)
            assert abs(u2 - u) <= 1
            assert abs(v2 - v) <= 1


def test_enu_vector_rotation_with_bearing():
    width, height = 1024, 512
    u = width // 2
    v = height // 2
    depth_m = 10.0

    east0, north0, up0, theta, phi = geometry.enu_vector_from_pixel(u, v, width, height, depth_m, 0.0)
    # Pixel at equator should have zero elevation and point straight ahead (north axis)
    assert math.isclose(up0, 0.0, abs_tol=1e-6)
    assert math.isclose(east0, 0.0, abs_tol=1e-6)
    assert math.isclose(north0, depth_m, rel_tol=1e-6)

    east90, north90, up90, *_ = geometry.enu_vector_from_pixel(u, v, width, height, depth_m, math.pi / 2)
    assert math.isclose(north90, 0.0, abs_tol=1e-6)
    assert math.isclose(east90, depth_m, rel_tol=1e-6)
    assert math.isclose(up90, 0.0, abs_tol=1e-6)


def test_geodesy_roundtrip_origin():
    lat0, lon0, alt0 = 37.7749, -122.4194, 15.0
    lat, lon, alt = geodesy.enu_to_geodetic(0.0, 0.0, 0.0, lat0, lon0, alt0)
    assert math.isclose(lat, lat0, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(lon, lon0, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(alt, alt0, rel_tol=1e-9, abs_tol=1e-6)


def test_geodesy_small_offset_matches_flat_model():
    lat0, lon0, alt0 = 0.0, 0.0, 100.0
    east, north, up = 5.0, 5.0, 2.0
    precise = geodesy.enu_to_geodetic(east, north, up, lat0, lon0, alt0)
    approx = geodesy.enu_flat_earth(east, north, up, lat0, lon0, alt0)
    assert math.isclose(precise[0], approx[0], abs_tol=1e-6)
    assert math.isclose(precise[1], approx[1], abs_tol=1e-6)
    assert math.isclose(precise[2], approx[2], abs_tol=1e-3)
