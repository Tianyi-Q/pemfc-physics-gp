# Bayesian Optimization for PEM Fuel Cells with Gaussian Processes

This project helps you choose which PEMFC gas diffusion layer experiment to run next with Bayesian Optimization. It is also the groundwork for a summer internship, you can find the details on my personal site: https://tianyi-q.github.io/ .

## What This Project Does

You provide past lab data as CSV files. The code cleans the data, fits a Gaussian Process model, and scores possible new experiments so you can focus on the most promising next step.

The pipeline does three things:

1. Read your raw experiment data.
2. Learn the relationship between the inputs and the measured voltage.
3. Suggest the next experiment to test.

This is useful because PEMFC characterization can be slow and expensive, so it helps to use previous experiments as efficiently as possible. Also, the metric I'm currently using is the voltage at a fixed current. You may modify it to reflect your optimization needs.s

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

If package installation fails on a very new Python version, try Python 3.11 or 3.12.

## Data Requirements

Put your CSV files inside `data/raw/`.

Each CSV must contain these columns exactly:

- `TiO2_Loading`
- `RH`
- `Voltage_1.5A`

Right now the model uses two input variables: `TiO2_Loading` and `RH`. More inputs could be added later if needed.

## Recommended Run Order

Most users should run the notebooks, not the `src/` files directly.

1. Put your CSV files into `data/raw/`.
2. Open `notebooks/01_data_check.ipynb` if you want to inspect the cleaned data first.
3. Open `notebooks/02_active_loop.ipynb` to run the main pipeline.

Inside the pipeline, the modules are used in this order:

1. `src/data_parser.py`
2. `src/gp_model.py`
3. `src/optimizer.py`

## Project Layout

```text
.
├── data/
│   ├── raw/          # original CSV files
│   └── processed/    # generated NumPy arrays
├── notebooks/        # notebook workflow
├── src/              # core Python code
└── requirements.txt  # Python dependencies
```

## What Each File Does

- `src/data_parser.py`: reads the CSV files, checks the required columns, removes bad rows, and saves clean arrays.
- `src/gp_model.py`: trains the Gaussian Process model and makes predictions.
- `src/optimizer.py`: scores candidate experiments and recommends the next one to try.
- `notebooks/01_data_check.ipynb`: optional notebook for checking the cleaned data.
- `notebooks/02_active_loop.ipynb`: main notebook for fitting the model and getting recommendations.

## Mathematical Background

This section explains the main ideas behind the model in plain language.

GitHub renders the equations below with MathJax. Some local Markdown previews may still show the raw LaTeX instead of formatted math.

### 1. Gaussian Process Regression

A Gaussian Process, or GP, is a way to model an unknown function from data.

In this project, the function maps the input variables to voltage:

```math
f(x) = f(\mathrm{TiO2\_Loading}, \mathrm{RH})
```

The measured voltage is modeled as:

```math
y = f(x) + \epsilon
```

where $\epsilon$ represents measurement noise of the sensor.

The GP assumption is:

```math
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
```

where:

- $m(x)$ is the mean function
- $k(x, x')$ is the kernel, which measures how similar two experiments are

After training, the GP gives a prediction and an uncertainty at a new point $x_*$:

```math
\mu(x_*) = k_*^\top K^{-1} y

\sigma^2(x_*) = k(x_*, x_*) - k_*^\top K^{-1} k_*
```

In plain language:

- $\mu(x_*)$ is the predicted voltage
- $\sigma(x_*)$ is the uncertainty of that prediction

That uncertainty is what makes Gaussian Processes useful for experiment planning.

### 2. Bayesian Optimization

Bayesian Optimization uses the GP as a cheap stand-in for the real experiment.

The loop is:

1. Fit a GP to the experiments you already ran.
2. Score many possible new experiments.
3. Choose the next experiment using both prediction and uncertainty.

This project uses the Expected Improvement, or EI, acquisition function:

```math
\mathrm{EI}(x) = \left(\mu(x) - y_{\mathrm{best}} - \xi\right)\Phi(Z) + \sigma(x)\phi(Z)

Z = \frac{\mu(x) - y_{\mathrm{best}} - \xi}{\sigma(x)}
```

where:

- $y_{\mathrm{best}}$ is the best voltage seen so far
- $\Phi$ is the standard normal cumulative distribution function
- $\phi$ is the standard normal probability density function
- $\xi$ is a small exploration parameter

In plain language:

- points with high predicted voltage get rewarded
- points with high uncertainty can also get rewarded
- EI balances exploitation and exploration

### 3. Kernel Used In This Project

The kernel in this repo is:

```math
k(x, x') = C \cdot k_{\mathrm{Mat\acute{e}rn}\,5/2}(x, x') + k_{\mathrm{white}}(x, x')
```

In code, that corresponds to:

- `ConstantKernel`
- `Matern(..., nu=2.5)`
- `WhiteKernel`

The Matérn-5/2 part is:

```math
k_{\mathrm{Mat\acute{e}rn}\,5/2}(x, x')
=
\left(1 + \sqrt{5}\,r + \frac{5}{3}r^2\right)e^{-\sqrt{5}r}
```

where:

```math
r = \sqrt{\sum_{i=1}^{2}\frac{(x_i - x_i')^2}{\ell_i^2}}
```

and $\ell_i$ are the learned length scales for each input dimension.

This means the model can learn that voltage may change more quickly with one variable than with the other.

The white-noise term is:

```math
k_{\mathrm{white}}(x, x') = \sigma_n^2 \delta(x, x')
```

This helps the model handle noisy experimental data.

In plain language, the full kernel says:

- nearby experiments should often behave similarly
- the response surface should be smooth, but not unrealistically perfect
- measurements may contain noise

## Direct Python Usage

If you want to call the modules yourself:

```python
from src.data_parser import load_and_sanitize_data
from src.gp_model import FuelCellSurrogate
from src.optimizer import get_next_experiment

X_train, y_train = load_and_sanitize_data("data/raw", "data/processed")
model = FuelCellSurrogate().fit(X_train, y_train)

bounds = [
    (X_train[:, 0].min(), X_train[:, 0].max()),
    (X_train[:, 1].min(), X_train[:, 1].max()),
]

next_point, _, _ = get_next_experiment(model, X_train, y_train, bounds)
print(next_point)
```

## Current Limits

- The optimizer searches over a grid of candidate points instead of a fully continuous space.
- There is no automated test suite yet.
- The notebook is still the main interface.
