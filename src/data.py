"""Download, cache and sanity-check the daily AAPL price series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless environment: never open a display
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_PATH = DATA_DIR / "aapl.csv"

TICKER = "AAPL"
START = "2012-01-01"
END = "2026-11-01"


def download_prices(
    ticker: str = TICKER, start: str = START, end: str = END
) -> pd.DataFrame:
    """Download adjusted daily OHLCV data and normalise its columns."""
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"No data returned for {ticker} between {start} and {end}")

    # Older yfinance releases return MultiIndex columns; flatten to field names.
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.to_datetime(frame.index)
    frame = frame.dropna().sort_index()  # chronological order, never shuffled
    return frame


def load_prices(cache_path: Path = CACHE_PATH, force: bool = False) -> pd.DataFrame:
    """Read the cached CSV when available, otherwise download and cache it."""
    if cache_path.exists() and not force:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(frame)} rows from cache: {cache_path.relative_to(ROOT)}")
        return frame

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = download_prices()
    frame.to_csv(cache_path)
    print(f"Downloaded and cached {len(frame)} rows: {cache_path.relative_to(ROOT)}")
    return frame


def sanity_checks(frame: pd.DataFrame) -> None:
    """Print the checks recommended in the guide before any modelling."""
    print(f"shape: {frame.shape}")
    print(f"date range: {frame.index.min().date()} -> {frame.index.max().date()}")
    print("missing values per column:")
    print(frame.isna().sum().to_string())
    print(f"index monotonic increasing: {frame.index.is_monotonic_increasing}")
    print("describe:")
    print(frame.describe().round(2).to_string())

    for name, series in (
        ("Close", frame["Close"]),
        ("Returns", frame["Close"].pct_change().dropna()),
    ):
        result = adfuller(series.dropna(), result_object=True)
        print(f"ADF p-value ({name}): {result.pvalue:.4g}")


def plot_overview(frame: pd.DataFrame, reports_dir: Path = REPORTS_DIR) -> Path:
    """Save the price/volume overview plot instead of showing it."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frame.index, frame["Close"], color="steelblue", linewidth=1)
    axes[0].set_title(f"{TICKER} adjusted close")
    axes[0].set_ylabel("Close (USD)")
    axes[1].plot(frame.index, frame["Volume"], color="gray", linewidth=0.8)
    axes[1].set_title("Volume")
    axes[1].set_ylabel("Volume")
    fig.tight_layout()
    output = reports_dir / "price_volume.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"saved plot: {output.relative_to(ROOT)}")
    return output


def main() -> None:
    """Run the data phase from the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the CSV cache exists"
    )
    args = parser.parse_args()

    frame = load_prices(force=args.force)
    sanity_checks(frame)
    plot_overview(frame)


if __name__ == "__main__":
    main()
