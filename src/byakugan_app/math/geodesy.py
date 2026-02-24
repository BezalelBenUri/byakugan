"""Geodesy utilities for ENU/ECEF and geodetic conversions."""
from __future__ import annotations

import functools
import math
from typing import Tuple

import numpy as np
from pyproj import CRS, Transformer

WGS84_GEODETIC = CRS.from_epsg(4979)  # lon, lat, h
WGS84_ECEF = CRS.from_epsg(4978)


@functools.lru_cache(maxsize=2)
def _geodetic_to_ecef_transformer() -> Transformer:
    return Transformer.from_crs(WGS84_GEODETIC, WGS84_ECEF, always_xy=True)


@functools.lru_cache(maxsize=2)
def _ecef_to_geodetic_transformer() -> Transformer:
    return Transformer.from_crs(WGS84_ECEF, WGS84_GEODETIC, always_xy=True)


def geodetic_to_ecef(lon_deg: float, lat_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert lon/lat/alt to ECEF coordinates."""
    transformer = _geodetic_to_ecef_transformer()
    x, y, z = transformer.transform(lon_deg, lat_deg, alt_m)
    return x, y, z


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to lon/lat/alt."""
    transformer = _ecef_to_geodetic_transformer()
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt


def enu_to_ecef(east: float, north: float, up: float, lat0: float, lon0: float) -> Tuple[float, float, float]:
    """Convert ENU offsets to ECEF deltas relative to a reference point.

    The transform uses the WGS84 local tangent frame with axes:
    East, North, Up -> X, Y, Z in Earth-Centred Earth-Fixed coordinates.
    """
    lat = math.radians(lat0)
    lon = math.radians(lon0)

    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)

    # ENU->ECEF rotation matrix.
    # Reference: standard local tangent frame transform.
    t = np.array(
        [
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ],
        dtype=float,
    )
    ecef_delta = t @ np.array([east, north, up], dtype=float)
    return tuple(float(v) for v in ecef_delta)


def ecef_to_enu(dx: float, dy: float, dz: float, lat0: float, lon0: float) -> Tuple[float, float, float]:
    """Convert ECEF deltas to ENU offsets at the reference latitude/longitude."""
    lat = math.radians(lat0)
    lon = math.radians(lon0)

    slat = math.sin(lat)
    clat = math.cos(lat)
    slon = math.sin(lon)
    clon = math.cos(lon)

    east = (-slon * dx) + (clon * dy)
    north = (-slat * clon * dx) + (-slat * slon * dy) + (clat * dz)
    up = (clat * clon * dx) + (clat * slon * dy) + (slat * dz)
    return float(east), float(north), float(up)


def enu_to_geodetic(
    east: float,
    north: float,
    up: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> Tuple[float, float, float]:
    """Convert ENU offsets to geodetic coordinates using WGS84."""
    x0, y0, z0 = geodetic_to_ecef(origin_lon, origin_lat, origin_alt)
    dx, dy, dz = enu_to_ecef(east, north, up, origin_lat, origin_lon)
    lat, lon, alt = ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)
    return lat, lon, alt


def geodetic_to_enu(
    target_lat: float,
    target_lon: float,
    target_alt: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> Tuple[float, float, float]:
    """Convert geodetic target coordinates to ENU relative to origin."""
    x0, y0, z0 = geodetic_to_ecef(origin_lon, origin_lat, origin_alt)
    xt, yt, zt = geodetic_to_ecef(target_lon, target_lat, target_alt)
    return ecef_to_enu(xt - x0, yt - y0, zt - z0, origin_lat, origin_lon)


def enu_flat_earth(
    east: float,
    north: float,
    up: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> Tuple[float, float, float]:
    """Approximate ENU?LLA using a flat Earth assumption (for testing)."""
    delta_lat = north / 111_111.0
    delta_lon = east / (111_111.0 * math.cos(math.radians(origin_lat)))
    return (
        origin_lat + delta_lat,
        origin_lon + delta_lon,
        origin_alt + up,
    )
