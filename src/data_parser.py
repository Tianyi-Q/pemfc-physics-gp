"""Read raw experiment CSV files, clean them, and save training arrays."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from .config import FEATURE_COLUMNS, TARGET_COLUMN, get_expected_columns
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import FEATURE_COLUMNS, TARGET_COLUMN, get_expected_columns


def _coerce_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Keep only the required columns and drop rows that are unusable."""
    expected_columns = get_expected_columns()
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"{source} is missing required column(s): {missing_text}")

    numeric = frame.loc[:, expected_columns].apply(pd.to_numeric, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty:
        warnings.warn(f"Skipping {source}: no valid rows after numeric cleaning.")

    return clean


def _resolve_processed_dir(
    raw_data_dir: Path, processed_dir: str | Path | None
) -> Path:
    """Choose where the cleaned NumPy files should be written."""
    if processed_dir is not None:
        return Path(processed_dir)
    return raw_data_dir.parent / "processed"


def load_and_sanitize_data(
    raw_data_dir: str | Path, processed_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read every CSV in the raw-data folder and turn it into clean model input.

    The function:
    - looks for CSV files in ``raw_data_dir``
    - keeps only the configured input and target columns
    - converts values to numbers when possible
    - drops rows with missing or invalid values
    - saves the final arrays to ``processed_dir``

    Returns ``X`` with the configured input columns and ``y`` with the target values.
    """
    raw_dir = Path(raw_data_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}.")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            frame = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            warnings.warn(f"Skipping empty file: {csv_path}")
            continue

        if frame.empty:
            warnings.warn(f"Skipping empty CSV: {csv_path}")
            continue

        clean_frame = _coerce_frame(frame, csv_path)
        if not clean_frame.empty:
            frames.append(clean_frame)

    if not frames:
        raise ValueError(
            "No usable rows were found after validation. Expected numeric columns: "
            + ", ".join(get_expected_columns())
        )

    # Combine all usable rows into one table before splitting inputs and target.
    master = pd.concat(frames, ignore_index=True)
    X = master.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = master.loc[:, TARGET_COLUMN].to_numpy(dtype=float)

    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS) or len(X) != len(y):
        raise ValueError("Parsed data has inconsistent feature/target dimensions.")

    processed_path = _resolve_processed_dir(raw_dir, processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    np.save(processed_path / "X_train.npy", X)
    np.save(processed_path / "y_train.npy", y)

    return X, y


if __name__ == "__main__":
    X, y = load_and_sanitize_data("data/raw", "data/processed")
    print(f"Loaded {len(y)} observations.")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        print(
            f"{feature_name} range: "
            f"[{X[:, index].min():.4f}, {X[:, index].max():.4f}]"
        )
    print(f"{TARGET_COLUMN} range: [{y.min():.4f}, {y.max():.4f}]")
    print("Saved X_train.npy and y_train.npy to data/processed/")
