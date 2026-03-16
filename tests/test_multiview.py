from __future__ import annotations

import numpy as np

from byakugan_app.math import multiview


def test_triangulate_observations_robust_handles_outlier_ray():
    true_point = np.array([15.0, -2.0, 4.0], dtype=np.float64)
    origins = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, -1.5, 2.0],
        ],
        dtype=np.float64,
    )
    directions = true_point[None, :] - origins
    directions[3] = np.array([1.0, -0.2, 0.05], dtype=np.float64)  # intentional mismatch
    weights = np.array([1.0, 1.0, 1.0, 0.3], dtype=np.float64)

    result = multiview.triangulate_observations_robust(
        origins,
        directions,
        weights=weights,
        ransac_threshold_m=0.8,
        huber_scale_m=0.4,
    )

    assert np.linalg.norm(result.point_ecef - true_point) < 0.5
    assert np.count_nonzero(result.inlier_mask) >= 3
    assert np.all(np.isfinite(result.ranges_m))
    assert result.iterations >= 1


def test_triangulate_observations_robust_requires_two_rays():
    origins = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    try:
        multiview.triangulate_observations_robust(origins, directions)
    except ValueError as exc:
        assert "At least two observations" in str(exc)
    else:  # pragma: no cover - defensive assertion style
        raise AssertionError("Expected ValueError for single observation.")


def test_bundle_adjust_point_with_pose_priors_refines_point():
    true_point = np.array([22.0, 4.0, 1.5], dtype=np.float64)
    origins = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.5, 0.1],
            [11.0, -0.6, 0.2],
        ],
        dtype=np.float64,
    )
    directions = true_point[None, :] - origins
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # Inject small angular and origin perturbation.
    directions[1] += np.array([0.01, -0.015, 0.005], dtype=np.float64)
    directions[1] /= np.linalg.norm(directions[1])
    noisy_origins = origins.copy()
    noisy_origins[2] += np.array([0.25, -0.12, 0.08], dtype=np.float64)

    robust = multiview.triangulate_observations_robust(
        noisy_origins,
        directions,
        weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    ba = multiview.bundle_adjust_point_with_pose_priors(
        point_seed=robust.point_ecef,
        origins=noisy_origins,
        directions=directions,
        weights=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        heading_sigma_deg=np.array([2.0, 2.0, 2.0], dtype=np.float64),
        position_sigma_m=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        solver_backend="scipy",
    )

    assert np.linalg.norm(ba.point_ecef - true_point) < 1.0
    assert np.isfinite(ba.residual_rms_m)
    assert ba.solver_backend == "scipy"


def test_bundle_adjust_point_with_pose_priors_uses_ceres_when_available():
    if not multiview.PYCERES_AVAILABLE:
        return
    true_point = np.array([12.0, -3.0, 2.0], dtype=np.float64)
    origins = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.3, 0.2],
            [8.0, -0.4, 0.3],
        ],
        dtype=np.float64,
    )
    directions = true_point[None, :] - origins
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    robust = multiview.triangulate_observations_robust(origins, directions)
    ba = multiview.bundle_adjust_point_with_pose_priors(
        point_seed=robust.point_ecef,
        origins=origins,
        directions=directions,
        weights=np.ones(3, dtype=np.float64),
        solver_backend="ceres",
    )
    assert np.isfinite(ba.point_ecef).all()
    assert ba.solver_backend == "ceres"
