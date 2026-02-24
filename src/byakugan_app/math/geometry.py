"""Geometry helpers for spherical panoramas."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

TWO_PI = 2.0 * math.pi


def pixel_to_angles(u: int, v: int, width: int, height: int) -> Tuple[float, float]:
    """Convert pixel coordinates to spherical angles.

    Returns a tuple (theta, phi) where theta is the azimuth in radians measured from
    the +X axis in the camera frame, increasing towards +Y, and phi is the elevation
    angle in radians measured from the XY plane (positive up).
    """
    u_norm = (u + 0.5) / float(width)
    v_norm = (v + 0.5) / float(height)
    theta = TWO_PI * u_norm
    phi = (math.pi / 2.0) - (math.pi * v_norm)
    return theta, phi


def angles_to_pixel(theta: float, phi: float, width: int, height: int) -> Tuple[int, int]:
    """Convert spherical angles to pixel indices (clamped to valid range)."""
    theta = theta % TWO_PI
    u_norm = theta / TWO_PI
    v_norm = (math.pi / 2.0 - phi) / math.pi

    u = int(np.clip(u_norm * width - 0.5, 0, width - 1))
    v = int(np.clip(v_norm * height - 0.5, 0, height - 1))
    return u, v


def spherical_direction(theta: float, phi: float) -> np.ndarray:
    """Return the unit direction vector for the given spherical angles."""
    cos_phi = math.cos(phi)
    return np.array(
        [
            cos_phi * math.cos(theta),
            cos_phi * math.sin(theta),
            math.sin(phi),
        ],
        dtype=np.float64,
    )


def enu_vector_from_angles(
    theta: float,
    phi: float,
    depth_m: float,
    bearing_rad: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> Tuple[float, float, float]:
    """Compute ENU vector from spherical angles and camera orientation.

    The camera local frame is defined as:
    - +X: forward
    - +Y: right
    - +Z: up

    `theta`/`phi` are measured in this local frame. The local ray is then rotated by
    roll (about +X), pitch (about +Y), and yaw/bearing (clockwise from north), and
    finally mapped into ENU.
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    x = depth_m * cos_phi * math.cos(theta)  # forward
    y = depth_m * cos_phi * math.sin(theta)  # right
    z = depth_m * sin_phi                    # up

    # Roll about forward axis.
    cos_roll = math.cos(roll_rad)
    sin_roll = math.sin(roll_rad)
    y_roll = (y * cos_roll) - (z * sin_roll)
    z_roll = (y * sin_roll) + (z * cos_roll)

    # Pitch about right axis (positive pitches camera upward).
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    x_pitch = (x * cos_pitch) - (z_roll * sin_pitch)
    z_pitch = (x * sin_pitch) + (z_roll * cos_pitch)

    # Yaw/bearing maps local forward/right axes into North/East.
    sin_bearing = math.sin(bearing_rad)
    cos_bearing = math.cos(bearing_rad)
    east = (x_pitch * sin_bearing) + (y_roll * cos_bearing)
    north = (x_pitch * cos_bearing) - (y_roll * sin_bearing)
    up = z_pitch
    return east, north, up


def intersect_ray_with_altitude_plane(
    theta: float,
    phi: float,
    bearing_rad: float,
    camera_alt_m: float,
    target_alt_m: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
    *,
    epsilon: float = 1e-9,
) -> Tuple[float, float, float, float]:
    """Intersect a panorama ray with a horizontal plane at fixed altitude.

    This is a depth-free fallback for georeferencing when no depth map is available.
    The returned ENU vector originates at the camera position and points to the
    intersection with the plane `altitude == target_alt_m`.

    Args:
        theta: Camera-frame azimuth angle in radians.
        phi: Elevation angle in radians.
        bearing_rad: Camera bearing in radians.
        camera_alt_m: Camera altitude in meters.
        target_alt_m: Altitude of the horizontal plane to intersect, in meters.
        pitch_rad: Camera pitch angle in radians.
        roll_rad: Camera roll angle in radians.
        epsilon: Numerical tolerance used to detect near-parallel rays.

    Returns:
        Tuple ``(east, north, up, slant_range_m)``.

    Raises:
        ValueError: If the ray is parallel to the plane or intersects behind camera.
    """
    east_u, north_u, up_u = enu_vector_from_angles(
        theta,
        phi,
        1.0,
        bearing_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
    )
    if abs(up_u) <= epsilon:
        raise ValueError("Selected ray is parallel to the altitude plane.")

    scale = (target_alt_m - camera_alt_m) / up_u
    if scale <= 0.0:
        raise ValueError("Selected ray does not intersect the altitude plane in front of camera.")

    east = east_u * scale
    north = north_u * scale
    up = up_u * scale
    return east, north, up, float(scale)


def enu_vector_from_pixel(
    u: int,
    v: int,
    width: int,
    height: int,
    depth_m: float,
    bearing_rad: float,
    pitch_rad: float = 0.0,
    roll_rad: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """Convenience wrapper returning ENU vector and angles for a pixel."""
    theta, phi = pixel_to_angles(u, v, width, height)
    east, north, up = enu_vector_from_angles(
        theta,
        phi,
        depth_m,
        bearing_rad,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
    )
    return east, north, up, theta, phi


def triangulate_rays_closest_point(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
    *,
    parallel_epsilon: float = 1e-9,
) -> Tuple[np.ndarray, float, float, float]:
    """Triangulate a 3D point from two rays using closest-point midpoint.

    Args:
        origin_a: First ray origin, shape ``(3,)``.
        direction_a: First ray direction (does not need to be unit length).
        origin_b: Second ray origin, shape ``(3,)``.
        direction_b: Second ray direction (does not need to be unit length).
        parallel_epsilon: Minimum denominator magnitude before treating rays as
            near-parallel and non-triangulatable.

    Returns:
        Tuple ``(point, residual_m, range_a_m, range_b_m)`` where:
        - ``point`` is the midpoint between the two closest points.
        - ``residual_m`` is the closest-point separation.
        - ``range_a_m`` and ``range_b_m`` are slant distances from each origin.

    Raises:
        ValueError: If a direction is degenerate or rays are near-parallel.
    """
    oa = np.asarray(origin_a, dtype=np.float64).reshape(3)
    ob = np.asarray(origin_b, dtype=np.float64).reshape(3)
    da = np.asarray(direction_a, dtype=np.float64).reshape(3)
    db = np.asarray(direction_b, dtype=np.float64).reshape(3)

    norm_a = float(np.linalg.norm(da))
    norm_b = float(np.linalg.norm(db))
    if norm_a <= parallel_epsilon or norm_b <= parallel_epsilon:
        raise ValueError("Triangulation direction vector is degenerate.")
    da /= norm_a
    db /= norm_b

    w0 = oa - ob
    a = float(np.dot(da, da))
    b = float(np.dot(da, db))
    c = float(np.dot(db, db))
    d = float(np.dot(da, w0))
    e = float(np.dot(db, w0))

    denom = (a * c) - (b * b)
    if abs(denom) <= parallel_epsilon:
        raise ValueError("Rays are near-parallel; triangulation is unstable.")

    ta = ((b * e) - (c * d)) / denom
    tb = ((a * e) - (b * d)) / denom

    pa = oa + (ta * da)
    pb = ob + (tb * db)
    midpoint = 0.5 * (pa + pb)
    residual = float(np.linalg.norm(pa - pb))
    return midpoint, residual, float(ta), float(tb)
