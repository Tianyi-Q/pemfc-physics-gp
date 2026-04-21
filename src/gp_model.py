"""Fit and query the Gaussian Process model used by the pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

_N_FEATURES = 2
_Y_STD_FLOOR = 1e-8


def _build_kernel() -> ConstantKernel | Matern | WhiteKernel:
    """Create the kernel that tells the GP how to model the data."""
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=[1.0, 1.0],
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
    )


class FuelCellSurrogate:
    """A small wrapper around scikit-learn's GP model for this project."""

    def __init__(self, n_restarts_optimizer: int = 10, random_state: int = 0):
        # Scale inputs and outputs so the GP fits more reliably.
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=_build_kernel(),
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False,
            random_state=random_state,
        )
        self._is_fitted = False

    @staticmethod
    def _as_feature_matrix(X: np.ndarray, *, name: str) -> np.ndarray:
        """Make sure feature data looks like a 2-column numeric table."""
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            if array.shape[0] != _N_FEATURES:
                raise ValueError(
                    f"{name} must contain {_N_FEATURES} features per observation."
                )
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != _N_FEATURES:
            raise ValueError(
                f"{name} must have shape (n_samples, {_N_FEATURES})."
            )
        return array

    @staticmethod
    def _as_target_vector(y: np.ndarray, *, name: str) -> np.ndarray:
        """Make sure target data is a flat numeric array."""
        array = np.asarray(y, dtype=float).reshape(-1)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        return array

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FuelCellSurrogate":
        """Train the model on the cleaned experiment data."""
        X_array = self._as_feature_matrix(X_train, name="X_train")
        y_array = self._as_target_vector(y_train, name="y_train")

        if len(X_array) != len(y_array):
            raise ValueError("X_train and y_train must have the same number of rows.")
        if len(y_array) < 2:
            raise ValueError("At least two observations are required to fit the GP.")
        if np.std(y_array) < _Y_STD_FLOOR:
            raise ValueError(
                "All voltages are identical (std ~= 0). Need distinct readings "
                "to fit a meaningful response surface."
            )

        X_scaled = self.scaler_X.fit_transform(X_array)
        y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).ravel()
        self.gp.fit(X_scaled, y_scaled)
        self._is_fitted = True
        return self

    def predict(
        self, X_query: np.ndarray, return_std: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict voltage for one or more candidate experiments."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X_array = self._as_feature_matrix(X_query, name="X_query")
        X_scaled = self.scaler_X.transform(X_array)

        if return_std:
            mu_scaled, sigma_scaled = self.gp.predict(X_scaled, return_std=True)
            mu = self.scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
            # Convert uncertainty back to the original voltage scale.
            sigma = sigma_scaled * self.scaler_y.scale_[0]
            return mu, sigma

        mu_scaled = self.gp.predict(X_scaled, return_std=False)
        return self.scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()

    def loo_cross_validate(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Test the model by holding out one row at a time."""
        X_array = self._as_feature_matrix(X, name="X")
        y_array = self._as_target_vector(y, name="y")

        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same number of observations.")
        if len(y_array) < 3:
            raise ValueError(
                "Leave-one-out validation requires at least three observations."
            )

        y_true = np.empty(len(y_array), dtype=float)
        y_pred = np.empty(len(y_array), dtype=float)
        y_std = np.empty(len(y_array), dtype=float)

        for index, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X_array)):
            # Refit on all rows except one, then predict the held-out row.
            fold = FuelCellSurrogate(
                n_restarts_optimizer=self.gp.n_restarts_optimizer,
                random_state=self.gp.random_state,
            )
            try:
                fold.fit(X_array[train_idx], y_array[train_idx])
            except ValueError as exc:
                raise ValueError(
                    "LOO fold failed because the training subset is degenerate. "
                    "Add more varied observations before running diagnostics."
                ) from exc

            mu, sigma = fold.predict(X_array[test_idx], return_std=True)
            y_true[index] = y_array[test_idx][0]
            y_pred[index] = mu[0]
            y_std[index] = sigma[0]

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return y_true, y_pred, y_std, rmse
