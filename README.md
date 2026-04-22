# Bayesian Optimization for PEM Fuel Cells with Gaussian Processes

This repository helps you choose which PEMFC gas diffusion layer experiment to run next.
It loads CSV data, fits a Gaussian Process surrogate model, and ranks candidate experiments with Expected Improvement.

Note: the repo name says physics-guided but the control equations aren't implemented yet. I want to iron out the details before starting to encode the physics

## One-Stop Configuration

The main knobs now live in `src/config.py`.

Edit that file when you want to change:

- `FEATURE_COLUMNS`: which input variables are active
- `TARGET_COLUMN`: which metric the model predicts
- `OPTIMIZATION_GOAL`: whether the optimizer should maximize or minimize the target
- `FEATURE_LABELS` and `TARGET_LABEL`: notebook-friendly names for plots and summaries
- `FEATURE_BOUNDS`: optimizer search ranges, in the same order as the active features

If you add a new feature such as `Catalyst`, add it in all relevant config sections so the notebooks can both verify it and optimize over it.

## Quick Start

Create a Conda environment, activate it, and install the required packages:

```bash
conda create -n pemfc-gp python=3.11
conda activate pemfc-gp
pip install -r requirements.txt
```

Then start Jupyter from the project root:

```bash
jupyter notebook
```

## Data Requirements

Put your CSV files inside `data/raw/`.
Every CSV must contain the active feature columns plus the active target column declared in `src/config.py`.

With the current defaults, that means:

- `TiO2_Loading`
- `RH`
- `Voltage_1.5A`

## Notebook Workflow

Most users should run the notebooks rather than calling the `src/` modules directly.

1. Open `notebooks/01_data_check.ipynb` to verify which features and target are active, inspect per-feature ranges, and look for obvious data issues.
2. Open `notebooks/02_active_loop.ipynb` to fit the GP, inspect the learned kernel, and generate experiment suggestions.

The notebooks now print:

- the active feature list
- the active target
- `X_train` and `y_train` shapes
- per-feature min, max, unique-count, mean, and scaling information
- whether any feature is constant in the training data
- whether the configured optimizer bounds are complete

Important limitation: the backend can read any configured feature count, but the contour-style notebook visualization still only works when exactly two active features are configured.

## Project Layout

```text
.
├── data/
│   ├── raw/          # original CSV files
│   └── processed/    # generated NumPy arrays
├── notebooks/        # notebook workflow
├── src/              # core Python code
│   └── config.py     # one-stop feature, target, label, and bounds settings
└── requirements.txt  # Python dependencies
```

## Mathematical Background

The model assumes an unknown function `f(x)` that maps the chosen inputs to the chosen target.
A Gaussian Process provides both a prediction and an uncertainty estimate at any candidate point.

The current surrogate uses:

```math
k(x, x') = C \cdot k_{\mathrm{Matern}\,5/2}(x, x') + k_{\mathrm{white}}(x, x')
```

and the optimizer ranks candidates with Expected Improvement.

## Current Limitations

- The contour visualization is intentionally limited to exactly two active features.
- The optimizer still uses a dense grid, so higher-dimensional searches will get expensive quickly.
- A constant feature can be loaded and verified, but it will not add information to the GP until the data vary along that dimension.
- There is still no automated test suite.
