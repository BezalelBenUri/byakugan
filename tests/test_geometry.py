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
    # Validate that no-bearing ENU matches the direct spherical projection.
    cos_phi = math.cos(phi)
    assert math.isclose(east0, depth_m * cos_phi * math.sin(theta), abs_tol=1e-6)
    assert math.isclose(north0, depth_m * cos_phi * math.cos(theta), abs_tol=1e-6)
    assert math.isclose(up0, depth_m * math.sin(phi), abs_tol=1e-6)

    east90, north90, up90, *_ = geometry.enu_vector_from_pixel(u, v, width, height, depth_m, math.pi / 2)
    # A +90 degree bearing rotates horizontal ENU axes while preserving Up.
    assert math.isclose(east90, north0, abs_tol=1e-6)
    assert math.isclose(north90, -east0, abs_tol=1e-6)
    assert math.isclose(up90, up0, abs_tol=1e-6)


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


def test_geodesy_geodetic_to_enu_roundtrip():
    lat0, lon0, alt0 = 6.62, 3.36, 54.3
    east, north, up = 12.5, -8.0, 2.4
    lat1, lon1, alt1 = geodesy.enu_to_geodetic(east, north, up, lat0, lon0, alt0)
    e2, n2, u2 = geodesy.geodetic_to_enu(lat1, lon1, alt1, lat0, lon0, alt0)
    assert math.isclose(e2, east, abs_tol=1e-4)
    assert math.isclose(n2, north, abs_tol=1e-4)
    assert math.isclose(u2, up, abs_tol=1e-4)


def test_intersect_ray_with_altitude_plane():
    theta = 0.0
    phi = -math.radians(30.0)
    camera_alt = 100.0
    target_alt = 90.0

    east, north, up, slant = geometry.intersect_ray_with_altitude_plane(
        theta,
        phi,
        bearing_rad=0.0,
        camera_alt_m=camera_alt,
        target_alt_m=target_alt,
    )

    assert math.isclose(east, 0.0, abs_tol=1e-6)
    assert up < 0.0
    assert slant > 0.0
    assert math.isclose(camera_alt + up, target_alt, abs_tol=1e-6)
    assert math.isclose(north, slant * math.cos(phi), abs_tol=1e-6)


def test_intersect_ray_with_altitude_plane_rejects_parallel_ray():
    with np.testing.assert_raises(ValueError):
        geometry.intersect_ray_with_altitude_plane(
            theta=0.0,
            phi=0.0,
            bearing_rad=0.0,
            camera_alt_m=100.0,
            target_alt_m=90.0,
        )


def test_enu_vector_with_zero_pitch_roll_matches_legacy_behavior():
    theta = math.radians(35.0)
    phi = math.radians(-10.0)
    depth = 42.0
    bearing = math.radians(15.0)
    east, north, up = geometry.enu_vector_from_angles(theta, phi, depth, bearing)
    theta_world = theta + bearing
    cos_phi = math.cos(phi)
    assert math.isclose(east, depth * cos_phi * math.sin(theta_world), abs_tol=1e-6)
    assert math.isclose(north, depth * cos_phi * math.cos(theta_world), abs_tol=1e-6)
    assert math.isclose(up, depth * math.sin(phi), abs_tol=1e-6)


def test_enu_vector_pitch_increases_up_component_for_forward_ray():
    theta = 0.0
    phi = 0.0
    depth = 10.0
    east0, north0, up0 = geometry.enu_vector_from_angles(theta, phi, depth, bearing_rad=0.0)
    east1, north1, up1 = geometry.enu_vector_from_angles(
        theta,
        phi,
        depth,
        bearing_rad=0.0,
        pitch_rad=math.radians(10.0),
    )
    assert math.isclose(east0, 0.0, abs_tol=1e-6)
    assert math.isclose(east1, 0.0, abs_tol=1e-6)
    assert up1 > up0
    assert north1 < north0


def test_triangulate_rays_closest_point_simple_intersection():
    origin_a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    origin_b = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    direction_b = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    point, residual, range_a, range_b = geometry.triangulate_rays_closest_point(
        origin_a,
        direction_a,
        origin_b,
        direction_b,
    )

    assert math.isclose(point[0], 1.0, abs_tol=1e-6)
    assert math.isclose(point[1], 0.0, abs_tol=1e-6)
    assert math.isclose(point[2], 0.0, abs_tol=1e-6)
    assert math.isclose(residual, 0.0, abs_tol=1e-6)
    assert math.isclose(range_a, 1.0, abs_tol=1e-6)
    assert math.isclose(range_b, 1.0, abs_tol=1e-6)
