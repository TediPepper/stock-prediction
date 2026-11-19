"""Metrics, naive baseline and plots saved under reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless environment: never open a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import REPORTS_DIR

METRIC_COLUMNS = ["Model", "RMSE", "MAE", "MAPE", "DirectionalAccuracy"]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute the four required metrics on price-scale values."""
    actual = np.asarray(actual, dtype="float64").ravel()
    predicted = np.asarray(predicted, dtype="float64").ravel()
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    direction = float(
        np.mean((np.diff(actual) > 0) == (np.diff(predicted) > 0)) * 100
    )
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 4),
        "DirectionalAccuracy": round(direction, 4),
    }


def naive_baseline(actual: np.ndarray) -> dict:
    """Tomorrow's price equals today's price, evaluated on the same period."""
    return compute_metrics(actual[1:], actual[:-1])


def plot_history(history, name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Save the training/validation loss curves."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_title(f"{name} loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.legend()
    fig.tight_layout()
    output = reports_dir / f"{name}_history.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def plot_predictions(
    dates, actual: np.ndarray, predicted: np.ndarray, name: str,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save the actual-vs-predicted chart on the test period."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual", linewidth=1)
    ax.plot(dates, predicted, label="Predicted", linewidth=1, alpha=0.85)
    ax.set_title(f"{name}: actual vs predicted")
    ax.set_xlabel("date")
    ax.set_ylabel("Close (USD)")
    ax.legend()
    fig.tight_layout()
    output = reports_dir / f"{name}_predictions.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output


def upsert_result(row: dict, path: Path = REPORTS_DIR / "results.csv") -> Path:
    """Add or replace one model row in the shared results table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        table = pd.read_csv(path)
        table = table[table["Model"] != row["Model"]]
    else:
        table = pd.DataFrame(columns=METRIC_COLUMNS)
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table = table[METRIC_COLUMNS]
    table.to_csv(path, index=False)
    return path
