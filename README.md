# AutoML for Trend Prediction

**Combined Algorithm Selection and Hyperparameter Optimisation (CASH) for Neural Network-based Financial Trend Prediction**

[![CI](https://github.com/adamlewison/py-performance/actions/workflows/ci.yml/badge.svg)](https://github.com/adamlewison/py-performance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project frames neural network architecture selection and hyperparameter tuning as a unified black-box optimisation problem — known as **CASH** — and applies population-based metaheuristics to solve it efficiently on financial time-series data.

Rather than predicting raw price values, the system first converts each price series into a compact sequence of **piecewise linear trend segments** (slope, duration) using a sliding-window OLS regression. A neural network — either an LSTM or MLP — is then trained to predict the next trend segment from the last four, effectively learning the rhythm of market movements.

The CASH optimiser simultaneously searches over:

| Dimension | Search space |
|---|---|
| Architecture | LSTM vs MLP |
| Learning rate | {0.1, 0.01, 0.001, 0.0001, 0.00001} |
| Dropout rate | {0.0 … 0.9} in steps of 0.1 |
| Hidden layers / size | 1–5 layers, 2–200 units each |
| LSTM depth | 2–5 layers |

Two experiments are provided:

| Script | Purpose |
|---|---|
| `exp.py` | Grid search over DE mutation factor *F* and crossover rate *CR* to study their effect on CASH performance |
| `algorithm_comparison.py` | Head-to-head comparison of Random Search, Pattern Search, and Differential Evolution as CASH optimisers |

---

## Method

```
Raw price CSV  →  Sliding-window OLS  →  Trend segments (slope, length)
                                                  ↓
                              Train / Val / Test split (60 / 20 / 20 %)
                                                  ↓
                           CASH via population-based metaheuristic
                             (Differential Evolution / Pattern Search)
                                                  ↓
                           Best (architecture, hyperparameters) → test MSE
```

The **sliding window** algorithm fits a least-squares line to an expanding window anchored at the current position. When the residual error exceeds a threshold, the previous fit is recorded as a trend segment and a new anchor is set. This yields a compact, noise-reduced representation of the price series.

---

## Datasets

Place CSV files in the `data/` directory. Each file must contain at minimum a `Date` and `Close` column.

| File | Exchange | Description |
|---|---|---|
| `NYSE.csv` | New York Stock Exchange | Composite index close prices |
| `NASDAQ.csv` | NASDAQ | Composite index close prices |
| `STX40.csv` | JSE STX40 | South African top-40 index |

---

## Installation

```bash
git clone https://github.com/adamlewison/py-performance.git
cd py-performance
pip install -r requirements.txt
```

For CPU-only PyTorch (recommended for most users):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Usage

### DE Hyperparameter Grid Search

Runs a full factorial grid search over population size, mutation factor *F*, and crossover rate *CR* across all datasets:

```bash
python exp.py
```

Run on a single dataset with a custom budget and epoch count:

```bash
python exp.py NYSE 180,25
```

### Algorithm Comparison

Compares Random Search, Pattern Search, and Differential Evolution:

```bash
python algorithm_comparison.py
python algorithm_comparison.py NASDAQ 540,50
```

### Arguments

Both scripts accept the same positional arguments:

```
python exp.py [dataset[,dataset,...]] [budget,num_epochs]
```

| Argument | Default | Description |
|---|---|---|
| `dataset` | `NYSE,NASDAQ,STX40` | Comma-separated list, or `all` |
| `budget` | 180 / 540 | Total function evaluations |
| `num_epochs` | 25 / 50 | Training epochs per evaluation |

Results are written to `Experiment Results (timestamp)/` as CSV files:
`Results.csv`, `Rows.csv`, `Params.csv`, `Predictions.csv`, `Summary.csv`, `SDA.csv`.

---

## Project Structure

```
py-performance/
├── exp.py                    # DE grid search (main experiment)
├── algorithm_comparison.py   # Random / PS / DE comparison
├── helpers.py                # TrendLine and StraightLine primitives
├── main.py                   # Standalone sliding-window demo
├── speed-test.py             # Modular constraint benchmark
├── app/
│   ├── TrendLine.py          # TrendLine class
│   └── FileStorage.py        # Simple file-based storage
├── data/                     # CSV price data (not tracked)
├── tests/
│   └── test_core.py          # Unit tests
├── requirements.txt
└── pyproject.toml
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=.
```

---

## Results

Each experiment exports a per-iteration convergence trace (`Results.csv`) and the best model's test-set predictions (`Predictions.csv`). The `SDA.csv` file reports:

- **Slope RMSE** — accuracy of the predicted trend direction
- **Duration RMSE** — accuracy of the predicted trend length
- **Average RMSE** — combined metric

---

## License

MIT — see [LICENSE](LICENSE).
