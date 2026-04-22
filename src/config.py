"""Central settings for input features, target metric, labels, and bounds."""

from __future__ import annotations

from typing import Final

FEATURE_COLUMNS: Final[tuple[str, ...]] = ("TiO2_Loading", "RH")
TARGET_COLUMN: Final[str] = "Voltage_1.5A"
OPTIMIZATION_GOAL: Final[str] = "maximize"

FEATURE_LABELS: Final[dict[str, str]] = {
    "TiO2_Loading": "TiO2 Loading (mg/cm^2)",
    "RH": "Relative Humidity (%)",
}
TARGET_LABEL: Final[str] = "Voltage at 1.5 A cm^-2 (V)"

FEATURE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "TiO2_Loading": (0.0, 0.1),
    "RH": (20.0, 100.0),
}


def get_expected_columns() -> tuple[str, ...]:
    """Return the columns the pipeline expects to load from the CSV files."""
    return (*FEATURE_COLUMNS, TARGET_COLUMN)



def get_feature_count() -> int:
    """Return how many input variables the current model should use."""
    return len(FEATURE_COLUMNS)



def get_feature_labels() -> tuple[str, ...]:
    """Return human-readable labels for the active feature columns."""
    return tuple(FEATURE_LABELS.get(column, column) for column in FEATURE_COLUMNS)



def get_target_label() -> str:
    """Return the human-readable label for the active target column."""
    return TARGET_LABEL or TARGET_COLUMN



def get_feature_bounds() -> list[tuple[float, float]]:
    """Return optimizer bounds in the same order as FEATURE_COLUMNS."""
    missing = [column for column in FEATURE_COLUMNS if column not in FEATURE_BOUNDS]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(
            "Missing FEATURE_BOUNDS entry for: " + missing_text
        )
    return [FEATURE_BOUNDS[column] for column in FEATURE_COLUMNS]



def get_objective_sign() -> float:
    """
    Return the sign used by the optimizer.

    ``maximize`` keeps the target unchanged.
    ``minimize`` flips the sign so the optimizer can still search for a maximum.
    """
    if OPTIMIZATION_GOAL == "maximize":
        return 1.0
    if OPTIMIZATION_GOAL == "minimize":
        return -1.0
    raise ValueError(
        "OPTIMIZATION_GOAL must be either 'maximize' or 'minimize'."
    )
