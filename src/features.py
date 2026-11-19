"""Scaling, overlapping windows and technical indicators for the AAPL series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60
TRAIN_FRACTION = 0.8
FEATURES = ["Close", "Volume", "RSI", "EMA20", "EMA50"]


def make_windows(
    data: np.ndarray, target_col: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (window, n_features) samples; a window ending at i predicts day i."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i, target_col])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def make_windows_multi(
    X_data: np.ndarray, y_data: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build windows for shifted next-day targets; sample i-1 targets day i."""
    X, y = [], []
    for i in range(window, len(X_data) + 1):
        X.append(X_data[i - window : i])
        y.append(y_data[i - 1])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Compute RSI with pandas_ta, falling back to plain pandas if needed."""
    try:
        import pandas_ta as ta

        result = ta.rsi(series, length=length)
        if result is not None:
            return result
    except Exception:  # the beta release may fail on some pandas builds
        pass
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Compute EMA with pandas_ta, falling back to plain pandas if needed."""
    try:
        import pandas_ta as ta

        result = ta.ema(series, length=length)
        if result is not None:
            return result
    except Exception:  # the beta release may fail on some pandas builds
        pass
    return series.ewm(span=length, adjust=False).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add RSI/EMA columns and the next-day Close target, dropping NaN rows."""
    out = frame.copy()
    out["RSI"] = rsi(out["Close"], length=14)
    out["EMA20"] = ema(out["Close"], length=20)
    out["EMA50"] = ema(out["Close"], length=50)
    out = out.dropna()
    out["Target"] = out["Close"].shift(-1)
    return out.dropna()


def prepare_univariate(
    frame: pd.DataFrame, window: int = WINDOW, train_fraction: float = TRAIN_FRACTION
) -> dict:
    """Scale Close with train statistics only and split windows chronologically."""
    split_idx = int(len(frame) * train_fraction)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(frame.iloc[:split_idx][["Close"]])
    scaled = scaler.transform(frame[["Close"]])

    X, y = make_windows(scaled, target_col=0, window=window)
    split_w = split_idx - window

    return {
        "X_train": X[:split_w],
        "y_train": y[:split_w],
        "X_test": X[split_w:],
        "y_test": y[split_w:],
        "scaler": scaler,
        "split_idx": split_idx,
        "train_target_end": frame.index[split_idx - 1],
        "test_target_start": frame.index[split_idx],
        "test_dates": frame.index[split_idx:],
    }


def prepare_multivariate(
    frame: pd.DataFrame,
    window: int = WINDOW,
    train_fraction: float = TRAIN_FRACTION,
    split_date: pd.Timestamp | None = None,
) -> dict:
    """Scale features/target on train rows only and build next-day-target windows."""
    featured = add_indicators(frame)
    if split_date is None:
        split_idx = int(len(featured) * train_fraction)
    else:
        split_idx = int(featured.index.searchsorted(pd.Timestamp(split_date)))

    # Row i targets day i+1, so row split_idx-1 targets the first test day.
    # Fit on rows before that one to keep test targets out of training.
    train_rows = featured.iloc[: split_idx - 1]
    feat_scaler = MinMaxScaler().fit(train_rows[FEATURES])
    target_scaler = MinMaxScaler().fit(train_rows[["Target"]])

    X_all = feat_scaler.transform(featured[FEATURES])
    y_all = target_scaler.transform(featured[["Target"]]).ravel()
    X, y = make_windows_multi(X_all, y_all, window)
    split_w = split_idx - window

    return {
        "X_train": X[:split_w],
        "y_train": y[:split_w],
        "X_test": X[split_w:],
        "y_test": y[split_w:],
        "feature_scaler": feat_scaler,
        "target_scaler": target_scaler,
        "split_idx": split_idx,
        "train_target_end": featured.index[split_idx - 1],
        "test_target_start": featured.index[split_idx],
        "test_dates": featured.index[split_idx:],
    }


def main() -> None:
    """Run the transformation phase from the terminal."""
    from src.data import load_prices

    frame = load_prices()
    uni = prepare_univariate(frame)
    multi = prepare_multivariate(frame, split_date=uni["test_target_start"])

    print("Univariate:")
    print(f"  train X/y: {uni['X_train'].shape} / {uni['y_train'].shape}")
    print(f"  test  X/y: {uni['X_test'].shape} / {uni['y_test'].shape}")
    print(
        "  scaler fitted on train Close: "
        f"min={uni['scaler'].data_min_[0]:.4f}, max={uni['scaler'].data_max_[0]:.4f} "
        f"(full min={frame['Close'].min():.4f}, max={frame['Close'].max():.4f})"
    )
    print(f"  last train target: {uni['train_target_end'].date()}")
    print(f"  first test target: {uni['test_target_start'].date()}")

    print("Multivariate:")
    print(f"  features: {FEATURES}")
    print(f"  train X/y: {multi['X_train'].shape} / {multi['y_train'].shape}")
    print(f"  test  X/y: {multi['X_test'].shape} / {multi['y_test'].shape}")
    print(f"  NaN cells in features: {int(add_indicators(frame)[FEATURES].isna().sum().sum())}")
    print(f"  last train target: {multi['train_target_end'].date()}")
    print(f"  first test target: {multi['test_target_start'].date()}")
    assert uni["train_target_end"] < uni["test_target_start"]
    assert multi["train_target_end"] < multi["test_target_start"]


if __name__ == "__main__":
    main()
