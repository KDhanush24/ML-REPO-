"""
Exploratory Data Analysis (EDA) module for Stock Return Prediction.

Generates static PNG charts (both on disk under `outputs/graphs/` and as
in-memory PNG bytes for one-click download in the Streamlit website) from
any validated and preprocessed stock dataset:
1. Historical Closing Price Trend (`1_closing_price_trend.png`)
2. Trading Volume Over Time (`2_trading_volume_trend.png`)
3. Daily Stock Return Distribution (`3_daily_return_distribution.png`)
4. Correlation Heatmap (`4_correlation_heatmap.png`)
5. Daily Return Boxplot (`5_daily_return_boxplot.png`)
6. Closing Price Distribution (`6_closing_price_distribution.png` & `6_returns_histogram.png`)
7. Actual Versus Predicted Returns (`7_actual_vs_predicted_returns.png`)
"""

import io
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import (
    CLEAN_DATASET_PATH,
    FEATURE_COLUMNS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    run_preprocessing,
)

GRAPHS_DIR = OUTPUTS_DIR / "graphs"
PREDICTIONS_CSV_PATH = OUTPUTS_DIR / "predictions.csv"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
    }
)

EDA_INTERPRETATIONS: Dict[str, str] = {
    "closing_price_trend": (
        "The closing-price chart shows how the stock price changed during the selected historical "
        "period. Because nominal stock prices drift over long periods (non-stationarity), our Machine "
        "Learning models predict daily percentage returns rather than raw price levels."
    ),
    "trading_volume_trend": (
        "Trading volume measures the number of shares exchanged on each trading day. The 30-day moving "
        "average highlights sustained shifts in market participation, while sharp single-day spikes often "
        "coincide with major corporate or macroeconomic announcements."
    ),
    "daily_return_distribution": (
        "Daily stock returns are centered near 0% with a bell-shaped distribution. However, compared to a "
        "perfect normal curve, stock returns display heavier tails—meaning large positive or negative single-day "
        "moves occur more frequently than a standard normal distribution would predict."
    ),
    "correlation_heatmap": (
        "The correlation heatmap displays linear relationships (from -1.0 to +1.0) among features and the "
        "next-day target return. Raw price columns (`open`, `high`, `low`, `close`) are strongly correlated "
        "with one another, which motivates using regularized regression (Ridge and Lasso). Note that "
        "correlation measures linear association only and does not prove causation."
    ),
    "daily_return_boxplot": (
        "The boxplot displays the median return (center line), interquartile range (middle 50% of trading days "
        "inside the box), and outlier sessions beyond the whiskers. Most daily fluctuations lie within a narrow "
        "band around 0%, while volatile trading sessions appear as outlier points."
    ),
    "closing_price_distribution": (
        "The closing-price distribution histogram shows how frequently the stock traded within different price "
        "ranges across the selected historical period, reflecting the different price regimes the stock passed through."
    ),
    "returns_histogram": (
        "Comparing 1-day returns and 5-day historical returns shows that both are centered near 0%, with 5-day "
        "returns exhibiting a wider spread because multi-day price movements accumulate over five sessions."
    ),
    "actual_vs_predicted": (
        "This chart compares the actual next-trading-day returns in the chronological test set against the "
        "returns predicted by Ridge and Lasso Regression. Because daily stock returns contain substantial "
        "unpredictable noise, regularized models produce conservative estimates near the historical mean return."
    ),
}


def load_clean_data() -> pd.DataFrame:
    """Load the default cleaned dataset, running preprocessing if needed."""
    if not CLEAN_DATASET_PATH.exists():
        df, _ = run_preprocessing(save_outputs=True)
        return df

    df = pd.read_csv(CLEAN_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _save_fig_and_bytes(fig: plt.Figure, out_path: Optional[Path] = None) -> bytes:
    """Render a Matplotlib figure to PNG bytes and optionally save to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    png_bytes = buf.getvalue()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)
    plt.close(fig)
    return png_bytes


def plot_closing_price_trend(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 1: Historical Closing Price Trend."""
    out_path = save_dir / "1_closing_price_trend.png"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df["date"], df["close"], color="#1f4e78", linewidth=1.3, label="Closing Price")
    ax.fill_between(df["date"], df["close"], color="#1f4e78", alpha=0.10)
    ax.set_title("1. Historical Closing Price Trend")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Closing Price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_trading_volume_trend(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 2: Trading Volume Over Time."""
    out_path = save_dir / "2_trading_volume_trend.png"
    vol_millions = df["volume"] / 1e6
    vol_ma30 = vol_millions.rolling(window=30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(
        df["date"],
        vol_millions,
        color="#7f8c8d",
        alpha=0.45,
        linewidth=0.7,
        label="Daily Volume (Millions)",
    )
    ax.plot(
        df["date"],
        vol_ma30,
        color="#d35400",
        linewidth=1.4,
        label="30-Day Moving Average Volume",
    )
    ax.set_title("2. Trading Volume Over Time")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Volume (Million Shares)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_daily_return_distribution(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 3: Daily Stock Return Distribution."""
    out_path = save_dir / "3_daily_return_distribution.png"
    if "daily_return" in df.columns:
        returns_pct = df["daily_return"].dropna() * 100.0
    else:
        returns_pct = df["close"].pct_change().dropna() * 100.0
    mean_ret = float(returns_pct.mean()) if not returns_pct.empty else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.histplot(
        returns_pct,
        bins=70,
        kde=True,
        stat="density",
        color="#2e86de",
        edgecolor="white",
        alpha=0.65,
        ax=ax,
    )
    ax.axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.2, label="Zero Return (0.00%)")
    ax.axvline(
        mean_ret,
        color="#e74c3c",
        linestyle="-",
        linewidth=1.4,
        label=f"Mean Daily Return ({mean_ret:+.2f}%)",
    )
    ax.set_title("3. Daily Stock Return Distribution (%)")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 4: Correlation Heatmap."""
    out_path = save_dir / "4_correlation_heatmap.png"
    available_cols = [
        c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns
    ]
    if len(available_cols) < 2:
        available_cols = [
            c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
        ]
    corr_matrix = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7.8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("4. Feature & Return Correlation Heatmap")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_daily_return_boxplot(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 5: Daily Return Boxplot."""
    out_path = save_dir / "5_daily_return_boxplot.png"
    daily_ret = (
        df["daily_return"] * 100.0
        if "daily_return" in df.columns
        else df["close"].pct_change() * 100.0
    )
    intraday_ret = (
        df["intraday_return"] * 100.0
        if "intraday_return" in df.columns
        else (df["close"] / df["open"] - 1.0) * 100.0
    )
    box_data = {
        "Daily Return (%)": daily_ret,
        "Intraday Return (%)": intraday_ret,
    }
    if TARGET_COLUMN in df.columns:
        box_data["Next-Day Target Return (%)"] = df[TARGET_COLUMN] * 100.0

    box_df = pd.DataFrame(box_data)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.boxplot(
        data=box_df,
        orient="h",
        palette="Set2",
        fliersize=2.5,
        width=0.45,
        ax=ax,
    )
    ax.axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.1)
    ax.set_title("5. Daily Return Boxplot (%)")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Return Metric")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_closing_price_distribution(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Graph 6: Closing Price Distribution."""
    out_path = save_dir / "6_closing_price_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.histplot(
        df["close"],
        bins=60,
        kde=True,
        color="#16a085",
        edgecolor="white",
        alpha=0.75,
        ax=ax,
    )
    ax.axvline(
        float(df["close"].mean()),
        color="#e74c3c",
        linestyle="--",
        linewidth=1.3,
        label=f"Mean Close ({df['close'].mean():,.2f})",
    )
    ax.set_title("6. Closing Price Distribution")
    ax.set_xlabel("Closing Price")
    ax.set_ylabel("Frequency (Trading Days)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_returns_histogram(df: pd.DataFrame, save_dir: Path = GRAPHS_DIR) -> Path:
    """Additional Graph: Side-by-side histogram of 1-day vs 5-day historical returns."""
    out_path = save_dir / "6_returns_histogram.png"
    daily_ret = (
        df["daily_return"] * 100.0
        if "daily_return" in df.columns
        else df["close"].pct_change() * 100.0
    )
    ret_5d = (
        df["return_5d"] * 100.0
        if "return_5d" in df.columns
        else (df["close"] / df["close"].shift(5) - 1.0) * 100.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    axes[0].hist(daily_ret.dropna(), bins=65, color="#1abc9c", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.1)
    axes[0].set_title("Histogram of 1-Day Returns (%)")
    axes[0].set_xlabel("1-Day Return (%)")
    axes[0].set_ylabel("Frequency (Trading Days)")

    axes[1].hist(ret_5d.dropna(), bins=65, color="#e67e22", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.1)
    axes[1].set_title("Histogram of 5-Day Historical Returns (%)")
    axes[1].set_xlabel("5-Day Return (%)")
    axes[1].set_ylabel("Frequency (Trading Days)")

    fig.suptitle("Frequency Histograms of Stock Returns", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def plot_actual_vs_predicted_returns(
    predictions_df: Optional[pd.DataFrame] = None,
    save_dir: Path = GRAPHS_DIR,
    last_n_days: int = 120,
) -> Optional[Path]:
    """Graph 7: Actual Versus Predicted Returns on the chronological test set."""
    if predictions_df is None:
        if not PREDICTIONS_CSV_PATH.exists():
            return None
        predictions_df = pd.read_csv(PREDICTIONS_CSV_PATH)

    if predictions_df is None or predictions_df.empty:
        return None

    out_path = save_dir / "7_actual_vs_predicted_returns.png"
    plot_df = predictions_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.tail(last_n_days)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(
        plot_df["date"],
        plot_df["actual_return"] * 100.0,
        color="#2c3e50",
        alpha=0.55,
        linewidth=1.1,
        label="Actual Next-Day Return (%)",
    )
    ax.plot(
        plot_df["date"],
        plot_df["ridge_predicted_return"] * 100.0,
        color="#2980b9",
        linewidth=1.6,
        label="Ridge Predicted Return (%)",
    )
    ax.plot(
        plot_df["date"],
        plot_df["lasso_predicted_return"] * 100.0,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.6,
        label="Lasso Predicted Return (%)",
    )
    ax.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax.set_title(f"7. Actual vs. Predicted Next-Day Returns (Last {len(plot_df)} Test Days)")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Next-Day Return (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig_and_bytes(fig, out_path)
    return out_path


def generate_eda_png_bytes(
    df: pd.DataFrame, predictions_df: Optional[pd.DataFrame] = None
) -> Dict[str, bytes]:
    """
    Generate all EDA static charts for the active dataset, save them to
    `outputs/graphs/`, and return a dictionary of `{filename: png_bytes}`
    for direct download in the Streamlit app.
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_closing_price_trend(df, GRAPHS_DIR),
        plot_trading_volume_trend(df, GRAPHS_DIR),
        plot_daily_return_distribution(df, GRAPHS_DIR),
        plot_correlation_heatmap(df, GRAPHS_DIR),
        plot_daily_return_boxplot(df, GRAPHS_DIR),
        plot_closing_price_distribution(df, GRAPHS_DIR),
        plot_returns_histogram(df, GRAPHS_DIR),
    ]
    if predictions_df is not None and not predictions_df.empty:
        avp_path = plot_actual_vs_predicted_returns(predictions_df, GRAPHS_DIR)
        if avp_path is not None:
            paths.append(avp_path)

    return {p.name: p.read_bytes() for p in paths if p is not None and p.exists()}


def run_eda(df: Optional[pd.DataFrame] = None) -> List[Path]:
    """Generate all EDA visualizations and return the list of saved image paths."""
    if df is None:
        df = load_clean_data()

    saved_paths: List[Path] = [
        plot_closing_price_trend(df),
        plot_trading_volume_trend(df),
        plot_daily_return_distribution(df),
        plot_correlation_heatmap(df),
        plot_daily_return_boxplot(df),
        plot_closing_price_distribution(df),
        plot_returns_histogram(df),
    ]

    pred_plot = plot_actual_vs_predicted_returns()
    if pred_plot is not None:
        saved_paths.append(pred_plot)

    return saved_paths


def main() -> None:
    """CLI entry point for running `python -m src.eda`."""
    print("=" * 60)
    print("Stock Return Prediction - Exploratory Data Analysis (EDA)")
    print("=" * 60)
    df = load_clean_data()
    saved_paths = run_eda(df)
    print(f"Loaded cleaned dataset with {len(df):,} observations.")
    print(f"Generated {len(saved_paths)} static visualizations in '{GRAPHS_DIR.relative_to(PROJECT_ROOT)}':")
    for path in saved_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()