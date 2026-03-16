"""Robust multi-view triangulation helpers.

This module extends the pairwise ray intersection utilities with robust
estimators that remain stable when one or more observations are noisy or
slightly mismatched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import numpy as np
from scipy.optimize import least_squares

from . import geometry

try:  # pragma: no cover - optional runtime dependency
    import pyceres

    PYCERES_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    pyceres = None
    PYCERES_AVAILABLE = False


@dataclass(slots=True, frozen=True)
class RobustTriangulationResult:
    """Result bundle returned by :func:`triangulate_observations_robust`.

    Attributes
    ----------
    point_ecef:
        Estimated world point in ECEF coordinates, shape ``(3,)``.
    residual_rms_m:
        Weighted RMS orthogonal distance from the point to the inlier rays.
    ranges_m:
        Signed slant ranges from each observation origin to the estimated point.
    orth_distances_m:
        Orthogonal distances from each observation ray to the estimated point.
    inlier_mask:
        Boolean array marking which rays were retained after robust fitting.
    iterations:
        Number of IRLS refinement iterations executed.
    """

    point_ecef: np.ndarray
    residual_rms_m: float
    ranges_m: np.ndarray
    orth_distances_m: np.ndarray
    inlier_mask: np.ndarray
    iterations: int


@dataclass(slots=True, frozen=True)
class BundleAdjustmentResult:
    """Bundle-adjusted point and pose perturbations with Gaussian priors.

    Attributes
    ----------
    point_ecef:
        Optimized point in ECEF.
    adjusted_origins:
        Per-observation optimized camera origins in ECEF.
    adjusted_directions:
        Per-observation optimized ray directions in ECEF (unit vectors).
    yaw_deltas_rad:
        Per-observation yaw perturbation around local Up in radians.
    origin_deltas_m:
        Per-observation ECEF translation perturbation in metres.
    residual_rms_m:
        Weighted RMS orthogonal distance after optimization.
    solver_backend:
        Solver backend that produced this result (``"ceres"`` or ``"scipy"``).
    """

    point_ecef: np.ndarray
    adjusted_origins: np.ndarray
    adjusted_directions: np.ndarray
    yaw_deltas_rad: np.ndarray
    origin_deltas_m: np.ndarray
    residual_rms_m: float
    solver_backend: str


def triangulate_observations_robust(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    ransac_threshold_m: float = 2.0,
    huber_scale_m: float = 1.0,
    max_ransac_trials: int = 300,
    max_irls_iters: int = 10,
    min_inliers: int = 2,
) -> RobustTriangulationResult:
    """Robustly triangulate a 3D point from two or more rays.

    The solver runs a deterministic RANSAC seed search, then refines the point
    with IRLS (Huber) and a final non-linear least-squares pass.
    """
    origins, directions, base_weights = _prepare_inputs(origins, directions, weights)
    if origins.shape[0] < 2:
        raise ValueError("At least two observations are required for triangulation.")

    threshold = max(0.05, float(ransac_threshold_m))
    huber = max(0.05, float(huber_scale_m))
    candidate_mask = _ransac_inlier_mask(
        origins,
        directions,
        base_weights,
        threshold_m=threshold,
        max_trials=max_ransac_trials,
        min_inliers=max(2, int(min_inliers)),
    )
    if int(np.sum(candidate_mask)) < 2:
        candidate_mask = np.ones(origins.shape[0], dtype=bool)

    point, iter_count = _irls_refine_point(
        origins,
        directions,
        base_weights,
        candidate_mask,
        huber_scale_m=huber,
        max_iters=max_irls_iters,
    )
    point = _nonlinear_refine_point(
        point,
        origins,
        directions,
        base_weights,
        candidate_mask,
        f_scale=huber,
    )

    orth = _orthogonal_distances(point, origins, directions)
    inlier_mask = candidate_mask & (orth <= max(threshold, 2.5 * huber))
    if int(np.sum(inlier_mask)) < 2:
        inlier_mask = candidate_mask.copy()

    weighted = np.clip(base_weights * inlier_mask.astype(np.float64), 1e-9, None)
    residual_rms = float(np.sqrt(np.average(np.square(orth), weights=weighted)))
    ranges = np.sum(directions * (point[None, :] - origins), axis=1)

    return RobustTriangulationResult(
        point_ecef=point.astype(np.float64),
        residual_rms_m=residual_rms,
        ranges_m=ranges.astype(np.float64),
        orth_distances_m=orth.astype(np.float64),
        inlier_mask=inlier_mask.astype(bool),
        iterations=iter_count,
    )


def bundle_adjust_point_with_pose_priors(
    *,
    point_seed: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    heading_sigma_deg: Optional[np.ndarray] = None,
    position_sigma_m: Optional[np.ndarray] = None,
    max_nfev: int = 120,
    solver_backend: Literal["auto", "ceres", "scipy"] = "auto",
) -> BundleAdjustmentResult:
    """Run local bundle adjustment over one point and per-observation pose deltas.

    The optimization variables are:
    - one 3D point in ECEF,
    - one small yaw perturbation per observation,
    - one small 3D ECEF translation perturbation per observation.

    Priors are applied to yaw and translation perturbations using
    heading/position uncertainty estimates from metadata.
    """
    o, d, base_weights = _prepare_inputs(origins, directions, weights)
    n = o.shape[0]
    if n < 2:
        raise ValueError("Bundle adjustment requires at least two observations.")

    point0 = np.asarray(point_seed, dtype=np.float64).reshape(3)
    heading_sigma = _sanitize_sigma_vector(heading_sigma_deg, n, default=3.0, floor=0.25)
    position_sigma = _sanitize_sigma_vector(position_sigma_m, n, default=2.0, floor=0.2)
    up_axes = _compute_up_axes(o)

    backend = _resolve_bundle_adjustment_backend(solver_backend)
    if backend == "ceres":
        try:
            return _solve_bundle_adjustment_ceres(
                point_seed=point0,
                origins=o,
                directions=d,
                weights=base_weights,
                heading_sigma_deg=heading_sigma,
                position_sigma_m=position_sigma,
                up_axes=up_axes,
                max_iterations=max_nfev,
            )
        except Exception:
            if solver_backend == "ceres":
                raise
            # Robust fallback when Ceres is unavailable or fails numerically.
            return _solve_bundle_adjustment_scipy(
                point_seed=point0,
                origins=o,
                directions=d,
                weights=base_weights,
                heading_sigma_deg=heading_sigma,
                position_sigma_m=position_sigma,
                up_axes=up_axes,
                max_nfev=max_nfev,
            )

    return _solve_bundle_adjustment_scipy(
        point_seed=point0,
        origins=o,
        directions=d,
        weights=base_weights,
        heading_sigma_deg=heading_sigma,
        position_sigma_m=position_sigma,
        up_axes=up_axes,
        max_nfev=max_nfev,
    )


def _resolve_bundle_adjustment_backend(
    backend: Literal["auto", "ceres", "scipy"],
) -> Literal["ceres", "scipy"]:
    """Resolve requested BA backend with availability-aware fallback."""
    if backend == "scipy":
        return "scipy"
    if backend == "ceres":
        if not PYCERES_AVAILABLE:
            raise RuntimeError("Ceres backend requested but pyceres is not installed.")
        return "ceres"
    if PYCERES_AVAILABLE:
        return "ceres"
    return "scipy"


def _compute_up_axes(origins: np.ndarray) -> np.ndarray:
    """Compute stable local Up axes for each observation origin."""
    origin_norms = np.linalg.norm(origins, axis=1, keepdims=True)
    up_axes = np.divide(
        origins,
        np.clip(origin_norms, 1e-12, None),
        out=np.zeros_like(origins),
    )
    tiny_mask = origin_norms.reshape(-1) <= 1e-9
    if np.any(tiny_mask):
        up_axes[tiny_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return up_axes


def _solve_bundle_adjustment_scipy(
    *,
    point_seed: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    heading_sigma_deg: np.ndarray,
    position_sigma_m: np.ndarray,
    up_axes: np.ndarray,
    max_nfev: int,
) -> BundleAdjustmentResult:
    """Solve local BA with SciPy robust least squares."""
    n = origins.shape[0]
    x0 = np.zeros(3 + n + (3 * n), dtype=np.float64)
    x0[:3] = point_seed
    sqrt_w = np.sqrt(np.clip(weights, 1e-9, None))
    heading_sigma_rad = np.radians(heading_sigma_deg)

    def residuals(x: np.ndarray) -> np.ndarray:
        point = x[:3]
        yaw = x[3 : 3 + n]
        origin_delta = x[3 + n :].reshape(n, 3)
        origin_opt = origins + origin_delta

        direction_opt = np.zeros_like(directions)
        for idx in range(n):
            direction_opt[idx] = _rotate_vector_axis_angle(directions[idx], up_axes[idx], yaw[idx])

        cross = np.cross(direction_opt, point[None, :] - origin_opt, axis=1)
        data_term = (cross * sqrt_w[:, None]).reshape(-1)
        yaw_prior = yaw / np.clip(heading_sigma_rad, 1e-6, None)
        pos_prior = (origin_delta / np.clip(position_sigma_m[:, None], 1e-6, None)).reshape(-1)
        return np.concatenate([data_term, yaw_prior, pos_prior], axis=0)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max(40, int(max_nfev)),
    )
    x_opt = result.x if result.success else x0
    return _build_bundle_adjustment_result(
        point=x_opt[:3],
        yaw=x_opt[3 : 3 + n],
        origin_delta=x_opt[3 + n :].reshape(n, 3),
        origins=origins,
        directions=directions,
        up_axes=up_axes,
        weights=weights,
        solver_backend="scipy",
    )


def _solve_bundle_adjustment_ceres(
    *,
    point_seed: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    heading_sigma_deg: np.ndarray,
    position_sigma_m: np.ndarray,
    up_axes: np.ndarray,
    max_iterations: int,
) -> BundleAdjustmentResult:
    """Solve local BA with Ceres through pyceres bindings."""
    if not PYCERES_AVAILABLE or pyceres is None:
        raise RuntimeError("pyceres is unavailable in this environment.")

    n = origins.shape[0]
    point = np.asarray(point_seed, dtype=np.float64).reshape(3).copy()
    yaw_blocks = [np.zeros(1, dtype=np.float64) for _ in range(n)]
    origin_delta_blocks = [np.zeros(3, dtype=np.float64) for _ in range(n)]

    problem = pyceres.Problem()
    for idx in range(n):
        obs_cost = _CeresObservationCost(
            origin=origins[idx],
            direction=directions[idx],
            up_axis=up_axes[idx],
            sqrt_weight=math.sqrt(max(float(weights[idx]), 1e-9)),
        )
        problem.add_residual_block(
            obs_cost,
            pyceres.SoftLOneLoss(1.0),
            [point, yaw_blocks[idx], origin_delta_blocks[idx]],
        )
        problem.add_residual_block(
            _CeresYawPriorCost(sigma_rad=max(math.radians(float(heading_sigma_deg[idx])), 1e-6)),
            pyceres.TrivialLoss(),
            [yaw_blocks[idx]],
        )
        problem.add_residual_block(
            _CeresOriginPriorCost(sigma_m=max(float(position_sigma_m[idx]), 1e-6)),
            pyceres.TrivialLoss(),
            [origin_delta_blocks[idx]],
        )

    options = pyceres.SolverOptions()
    options.max_num_iterations = max(20, int(max_iterations))
    options.minimizer_progress_to_stdout = False
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    yaw = np.array([block[0] for block in yaw_blocks], dtype=np.float64)
    origin_delta = np.stack(origin_delta_blocks, axis=0).astype(np.float64)
    return _build_bundle_adjustment_result(
        point=point,
        yaw=yaw,
        origin_delta=origin_delta,
        origins=origins,
        directions=directions,
        up_axes=up_axes,
        weights=weights,
        solver_backend="ceres",
    )


def _build_bundle_adjustment_result(
    *,
    point: np.ndarray,
    yaw: np.ndarray,
    origin_delta: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    up_axes: np.ndarray,
    weights: np.ndarray,
    solver_backend: str,
) -> BundleAdjustmentResult:
    """Assemble BA result object and residual diagnostics."""
    origin_opt = origins + origin_delta
    direction_opt = np.zeros_like(directions)
    for idx in range(directions.shape[0]):
        direction_opt[idx] = _rotate_vector_axis_angle(directions[idx], up_axes[idx], yaw[idx])
    orth = _orthogonal_distances(point, origin_opt, direction_opt)
    residual_rms = float(np.sqrt(np.average(np.square(orth), weights=np.clip(weights, 1e-9, None))))
    return BundleAdjustmentResult(
        point_ecef=np.asarray(point, dtype=np.float64).reshape(3),
        adjusted_origins=origin_opt.astype(np.float64),
        adjusted_directions=direction_opt.astype(np.float64),
        yaw_deltas_rad=np.asarray(yaw, dtype=np.float64).reshape(-1),
        origin_deltas_m=origin_delta.astype(np.float64),
        residual_rms_m=residual_rms,
        solver_backend=solver_backend,
    )


def _prepare_inputs(
    origins: np.ndarray,
    directions: np.ndarray,
    weights: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize triangulation inputs."""
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if o.ndim != 2 or d.ndim != 2 or o.shape != d.shape or o.shape[1] != 3:
        raise ValueError("origins and directions must be matching (N,3) arrays.")
    norms = np.linalg.norm(d, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("One or more direction vectors are degenerate.")
    d_unit = d / norms[:, None]
    if weights is None:
        w = np.ones(o.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != o.shape[0]:
            raise ValueError("weights must have one entry per ray observation.")
        w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 1e-12:
        raise ValueError("All observation weights are zero.")
    return o, d_unit, w


def _orthogonal_distances(point: np.ndarray, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return orthogonal distances from point to each ray."""
    return np.linalg.norm(np.cross(point[None, :] - origins, directions, axis=1), axis=1)


def _ransac_inlier_mask(
    origins: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
    *,
    threshold_m: float,
    max_trials: int,
    min_inliers: int,
) -> np.ndarray:
    """Find a robust inlier set by sampling ray pairs."""
    n = origins.shape[0]
    pair_indices: list[tuple[int, int]]
    if n <= 16:
        pair_indices = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    else:
        rng = np.random.default_rng(17)
        pair_indices = []
        for _ in range(max_trials):
            i, j = rng.choice(n, size=2, replace=False)
            if i > j:
                i, j = j, i
            pair_indices.append((int(i), int(j)))

    best_score = -math.inf
    best_mask = np.ones(n, dtype=bool)
    for i, j in pair_indices:
        try:
            point, _, range_i, range_j = geometry.triangulate_rays_closest_point(
                origins[i],
                directions[i],
                origins[j],
                directions[j],
            )
        except ValueError:
            continue
        if range_i <= 0.0 and range_j <= 0.0:
            continue
        distances = _orthogonal_distances(point, origins, directions)
        mask = distances <= threshold_m
        inlier_count = int(np.sum(mask))
        if inlier_count < min_inliers:
            continue
        inlier_weight = float(np.sum(weights[mask]))
        weighted_error = float(np.average(distances[mask], weights=np.clip(weights[mask], 1e-9, None)))
        score = inlier_weight - weighted_error
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _irls_refine_point(
    origins: np.ndarray,
    directions: np.ndarray,
    base_weights: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    huber_scale_m: float,
    max_iters: int,
) -> tuple[np.ndarray, int]:
    """Iteratively reweight ray residuals with a Huber-like kernel."""
    active_weights = np.clip(base_weights * inlier_mask.astype(np.float64), 0.0, None)
    point, _, _ = geometry.triangulate_rays_weighted_least_squares(
        origins,
        directions,
        weights=np.clip(active_weights, 1e-9, None),
    )
    for iteration in range(1, max(1, max_iters) + 1):
        distances = _orthogonal_distances(point, origins, directions)
        robust = np.ones_like(distances)
        far_mask = distances > huber_scale_m
        robust[far_mask] = huber_scale_m / np.maximum(distances[far_mask], 1e-9)
        refined_weights = np.clip(active_weights * robust, 0.0, None)
        point_next, _, _ = geometry.triangulate_rays_weighted_least_squares(
            origins,
            directions,
            weights=np.clip(refined_weights, 1e-9, None),
        )
        if float(np.linalg.norm(point_next - point)) <= 1e-5:
            return point_next, iteration
        point = point_next
    return point, max(1, max_iters)


def _nonlinear_refine_point(
    point_seed: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    base_weights: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    f_scale: float,
) -> np.ndarray:
    """Refine triangulated point with robust non-linear least squares."""
    weights = np.clip(base_weights * inlier_mask.astype(np.float64), 0.0, None)
    active = weights > 0.0
    if int(np.sum(active)) < 2:
        return point_seed

    o = origins[active]
    d = directions[active]
    w = np.sqrt(np.clip(weights[active], 1e-9, None))

    def residuals(x: np.ndarray) -> np.ndarray:
        point = x.reshape(3)
        cross = np.cross(d, point[None, :] - o, axis=1)
        return (cross * w[:, None]).reshape(-1)

    result = least_squares(
        residuals,
        np.asarray(point_seed, dtype=np.float64).reshape(3),
        method="trf",
        loss="soft_l1",
        f_scale=max(0.05, float(f_scale)),
        max_nfev=120,
    )
    if not result.success or result.x.shape[0] != 3:
        return point_seed
    return result.x.astype(np.float64)


class _CeresObservationCost(pyceres.CostFunction if PYCERES_AVAILABLE else object):
    """Ceres residual for one observation's orthogonal point-to-ray error."""

    def __init__(self, *, origin: np.ndarray, direction: np.ndarray, up_axis: np.ndarray, sqrt_weight: float) -> None:
        if not PYCERES_AVAILABLE or pyceres is None:
            raise RuntimeError("pyceres is unavailable.")
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 1, 3])
        self._origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self._direction = np.asarray(direction, dtype=np.float64).reshape(3)
        self._up_axis = np.asarray(up_axis, dtype=np.float64).reshape(3)
        self._sqrt_weight = max(float(sqrt_weight), 1e-9)

    def _residual(self, point: np.ndarray, yaw: float, origin_delta: np.ndarray) -> np.ndarray:
        origin_opt = self._origin + origin_delta
        direction_opt = _rotate_vector_axis_angle(self._direction, self._up_axis, yaw)
        cross = np.cross(direction_opt, point - origin_opt)
        return cross * self._sqrt_weight

    def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802
        point = np.asarray(parameters[0], dtype=np.float64).reshape(3)
        yaw = float(parameters[1][0])
        origin_delta = np.asarray(parameters[2], dtype=np.float64).reshape(3)
        residual = self._residual(point, yaw, origin_delta)
        residuals[:3] = residual
        if jacobians is None:
            return True

        if jacobians[0] is not None:
            jac = _numeric_jacobian(
                lambda value: self._residual(np.asarray(value, dtype=np.float64).reshape(3), yaw, origin_delta),
                point.copy(),
                epsilon=1e-7,
            )
            jacobians[0][:] = jac.reshape(-1)
        if jacobians[1] is not None:
            jac = _numeric_jacobian(
                lambda value: self._residual(point, float(np.asarray(value)[0]), origin_delta),
                np.array([yaw], dtype=np.float64),
                epsilon=1e-7,
            )
            jacobians[1][:] = jac.reshape(-1)
        if jacobians[2] is not None:
            jac = _numeric_jacobian(
                lambda value: self._residual(point, yaw, np.asarray(value, dtype=np.float64).reshape(3)),
                origin_delta.copy(),
                epsilon=1e-7,
            )
            jacobians[2][:] = jac.reshape(-1)
        return True


class _CeresYawPriorCost(pyceres.CostFunction if PYCERES_AVAILABLE else object):
    """Ceres Gaussian prior on yaw perturbation."""

    def __init__(self, *, sigma_rad: float) -> None:
        if not PYCERES_AVAILABLE or pyceres is None:
            raise RuntimeError("pyceres is unavailable.")
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])
        self._inv_sigma = 1.0 / max(float(sigma_rad), 1e-9)

    def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802
        yaw = float(parameters[0][0])
        residuals[0] = yaw * self._inv_sigma
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = self._inv_sigma
        return True


class _CeresOriginPriorCost(pyceres.CostFunction if PYCERES_AVAILABLE else object):
    """Ceres Gaussian prior on 3D origin perturbation."""

    def __init__(self, *, sigma_m: float) -> None:
        if not PYCERES_AVAILABLE or pyceres is None:
            raise RuntimeError("pyceres is unavailable.")
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3])
        self._inv_sigma = 1.0 / max(float(sigma_m), 1e-9)

    def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802
        delta = np.asarray(parameters[0], dtype=np.float64).reshape(3)
        residuals[:3] = delta * self._inv_sigma
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = np.eye(3, dtype=np.float64).reshape(-1) * self._inv_sigma
        return True


def _sanitize_sigma_vector(
    sigma: Optional[np.ndarray],
    length: int,
    *,
    default: float,
    floor: float,
) -> np.ndarray:
    """Normalize optional per-observation sigma vectors."""
    if sigma is None:
        values = np.full(length, default, dtype=np.float64)
    else:
        values = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if values.shape[0] != length:
            raise ValueError("Sigma vector length must match number of observations.")
        values = np.where(np.isfinite(values) & (values > 0.0), values, default)
    return np.clip(values, floor, None)


def _rotate_vector_axis_angle(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 3D vector around an axis via Rodrigues' formula."""
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm_axis = float(np.linalg.norm(a))
    if norm_axis <= 1e-12 or abs(float(angle_rad)) <= 1e-12:
        return v
    unit_axis = a / norm_axis
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    return (
        (v * cos_a)
        + (np.cross(unit_axis, v) * sin_a)
        + (unit_axis * np.dot(unit_axis, v) * (1.0 - cos_a))
    )


def _numeric_jacobian(
    function,
    x0: np.ndarray,
    *,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Compute central-difference Jacobian for small parameter blocks."""
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    y0 = np.asarray(function(x), dtype=np.float64).reshape(-1)
    jac = np.zeros((y0.shape[0], x.shape[0]), dtype=np.float64)
    for idx in range(x.shape[0]):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += epsilon
        xm[idx] -= epsilon
        yp = np.asarray(function(xp), dtype=np.float64).reshape(-1)
        ym = np.asarray(function(xm), dtype=np.float64).reshape(-1)
        jac[:, idx] = (yp - ym) / (2.0 * epsilon)
    return jac
