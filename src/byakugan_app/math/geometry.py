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


def enu_vector_from_angles(theta: float, phi: float, depth_m: float, bearing_rad: float) -> Tuple[float, float, float]:
    """Compute ENU vector from spherical angles, depth, and bearing.

    The Insta360 Pro2 is assumed level (no pitch/roll). The incoming `theta` is the
    camera-frame azimuth. We rotate by the camera bearing to align with world axes.
    """
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    theta_world = (theta + bearing_rad) % TWO_PI
    east = depth_m * cos_phi * math.sin(theta_world)
    north = depth_m * cos_phi * math.cos(theta_world)
    up = depth_m * sin_phi
    return east, north, up


def enu_vector_from_pixel(
    u: int,
    v: int,
    width: int,
    height: int,
    depth_m: float,
    bearing_rad: float,
) -> Tuple[float, float, float, float, float]:
    """Convenience wrapper returning ENU vector and angles for a pixel."""
    theta, phi = pixel_to_angles(u, v, width, height)
    east, north, up = enu_vector_from_angles(theta, phi, depth_m, bearing_rad)
    return east, north, up, theta, phi
