"""Score candidate experiments and suggest which one to try next."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

try:
    from .config import get_objective_sign
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import get_objective_sign

Bounds = Sequence[tuple[float, float]]


def _validate_bounds(bounds: Bounds) -> list[tuple[float, float]]:
    """Check that the search space is valid for the chosen input variables."""
    validated_bounds = list(bounds)
    if not validated_bounds:
        raise ValueError("bounds must define at least one feature range.")

    for index, (lower, upper) in enumerate(validated_bounds):
        if upper <= lower:
            raise ValueError(
                f"bounds[{index}] must have upper > lower; received {(lower, upper)}."
            )
    return validated_bounds


def _build_grid(bounds: Bounds, resolution: int) -> np.ndarray:
    """Create a grid of candidate experiments inside the chosen bounds."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    validated_bounds = _validate_bounds(bounds)
    axes = [np.linspace(lower, upper, resolution) for lower, upper in validated_bounds]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([axis.ravel() for axis in mesh])


def _validate_training_inputs(
    X_train: np.ndarray, y_train: np.ndarray, bounds: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Make sure the optimizer inputs are shaped consistently."""
    X_array = np.asarray(X_train, dtype=float)
    y_array = np.asarray(y_train, dtype=float).reshape(-1)

    if X_array.ndim != 2:
        raise ValueError("X_train must be a 2D array of observed experiments.")
    if len(X_array) != len(y_array):
        raise ValueError("X_train and y_train must have the same number of rows.")
    if X_array.shape[1] != len(bounds):
        raise ValueError(
            "The number of bounds must match the number of configured input features."
        )

    return X_array, y_array


def _observed_point_mask(X_grid: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Mark grid points that have already been tested in the lab data."""
    observed = np.asarray(X_train, dtype=float)
    if observed.size == 0:
        return np.zeros(len(X_grid), dtype=bool)
    if observed.ndim != 2 or observed.shape[1] != X_grid.shape[1]:
        raise ValueError("X_train must have the same feature dimension as X_grid.")

    return np.any(
        np.all(np.isclose(X_grid[:, None, :], observed[None, :, :]), axis=2),
        axis=1,
    )


def calculate_ei(
    gp_model, X_grid: np.ndarray, y_best: float, xi: float = 0.01
) -> np.ndarray:
    """Give each candidate a score based on possible improvement."""
    candidates = np.asarray(X_grid, dtype=float)
    if candidates.ndim != 2 or len(candidates) == 0:
        raise ValueError("X_grid must be a non-empty 2D array of candidate points.")

    objective_sign = get_objective_sign()
    mu, sigma = gp_model.predict(candidates, return_std=True)
    mu_objective = objective_sign * mu
    y_best_objective = objective_sign * y_best

    ei = np.zeros_like(mu_objective)
    mask = sigma > 0

    improvement = mu_objective[mask] - y_best_objective - xi
    z_scores = improvement / sigma[mask]
    ei[mask] = improvement * norm.cdf(z_scores) + sigma[mask] * norm.pdf(z_scores)
    return ei


def get_next_experiment(
    gp_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    bounds: Bounds,
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the single best new experiment from the candidate grid."""
    validated_bounds = _validate_bounds(bounds)
    X_array, y_array = _validate_training_inputs(X_train, y_train, validated_bounds)

    objective_sign = get_objective_sign()
    X_grid = _build_grid(validated_bounds, resolution)
    incumbent = float(y_array[np.argmax(objective_sign * y_array)])
    ei = calculate_ei(gp_model, X_grid, y_best=incumbent)
    ei[_observed_point_mask(X_grid, X_array)] = 0.0

    if np.max(ei) <= 0:
        raise ValueError(
            "No new candidate improved on the current best observation within the grid."
        )

    return X_grid[np.argmax(ei)], X_grid, ei


def get_batch_experiments(
    gp_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    bounds: Bounds,
    k: int = 3,
    min_distance: float = 0.1,
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return several strong candidates while keeping them spread out."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")

    validated_bounds = _validate_bounds(bounds)
    X_array, y_array = _validate_training_inputs(X_train, y_train, validated_bounds)

    objective_sign = get_objective_sign()
    X_grid = _build_grid(validated_bounds, resolution)
    incumbent = float(y_array[np.argmax(objective_sign * y_array)])
    ei = calculate_ei(gp_model, X_grid, y_best=incumbent)
    ei_original = ei.copy()
    ei[_observed_point_mask(X_grid, X_array)] = 0.0

    ranges = np.array([upper - lower for lower, upper in validated_bounds], dtype=float)
    minimums = np.array([lower for lower, _ in validated_bounds], dtype=float)
    X_norm = (X_grid - minimums) / ranges

    batch: list[np.ndarray] = []
    for _ in range(k):
        if np.max(ei) <= 0:
            break
        best_index = int(np.argmax(ei))
        batch.append(X_grid[best_index])
        # Zero out nearby points so the batch is not clustered in one area.
        distances = np.linalg.norm(X_norm - X_norm[best_index], axis=1)
        ei[distances < min_distance] = 0.0

    if not batch:
        return np.empty((0, X_grid.shape[1])), X_grid, ei_original

    return np.vstack(batch), X_grid, ei_original
