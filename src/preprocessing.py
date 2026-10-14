"""
Data loading, validation, cleaning, and preprocessing module
for Stock Return Prediction Using Ridge & Lasso Regression.

Supports:
- Automatic column normalization and mapping (e.g., `Date`, `Open`, `High`,
  `Low`, `Close`, `Adj Close`, `Volume` -> `date`, `open`, `high`, `low`,
  `close`, `adj_close`, `volume`).
- Treating `adj_close` as optional while requiring `date`, `open`, `high`,
  `low`, `close`, and `volume`.
- Chronological sorting, duplicate removal, missing-value removal, and
  filtering of invalid OHLC prices or negative trading volumes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data_loader import (
    DEFAULT_RAW_FILENAME,
    REQUIRED_STANDARD_COLUMNS,
    normalize_and_map_columns,
)
from src.feature_engineering import (
    COLUMN_DESCRIPTIONS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    TARGET_COLUMN,
    engineer_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLEAN_DATASET_PATH = DATASET_DIR / "clean_reliance_stock.csv"
QUALITY_SUMMARY_PATH = OUTPUTS_DIR / "data_quality_summary.json"

# Required columns for standard OHLCV datasets (`adj_close` is optional)
REQUIRED_COLUMNS: List[str] = list(REQUIRED_STANDARD_COLUMNS)

BASE_NUMERIC_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

BASE_PRICE_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
]


def find_raw_dataset(dataset_dir: Union[str, Path] = DATASET_DIR) -> Path:
    """
    Automatically locate the raw CSV file inside the dataset directory,
    explicitly ignoring generated cleaned CSV files.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    preferred = dataset_path / DEFAULT_RAW_FILENAME
    if preferred.exists():
        return preferred

    candidates = [
        p
        for p in sorted(dataset_path.glob("*.csv"))
        if not p.name.lower().startswith("clean")
        and p.name != CLEAN_DATASET_PATH.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No raw CSV dataset found in {dataset_path}. "
            f"Expected '{DEFAULT_RAW_FILENAME}'."
        )
    return candidates[0]


def load_raw_dataset(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load a raw stock dataset from CSV and verify basic non-emptiness.
    """
    csv_path = Path(filepath) if filepath is not None else find_raw_dataset()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The dataset file is empty: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"The loaded dataset contains 0 rows: {csv_path}")

    return df


def validate_columns(
    df: pd.DataFrame, required_cols: List[str] = REQUIRED_STANDARD_COLUMNS
) -> pd.DataFrame:
    """
    Normalize column names and verify that all required standardized columns exist.
    Returns the column-normalized DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None.")

    mapped_df, _, _ = normalize_and_map_columns(df)
    missing_cols = [col for col in required_cols if col not in mapped_df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )
    return mapped_df


def clean_dataset(
    df: pd.DataFrame, min_valid_rows: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean a raw stock market DataFrame.

    Steps performed:
    - Normalize and map column names (`Date` -> `date`, `Adjusted Close` -> `adj_close`, etc.).
    - Verify required columns (`date`, `open`, `high`, `low`, `close`, `volume`);
      treat `adj_close` as optional.
    - Convert `date` to timezone-naive datetime consistently.
    - Remove exact duplicate rows and duplicate trading dates.
    - Convert numeric columns to float/numeric types.
    - Remove rows with missing values (no fabrication of stock prices).
    - Filter out invalid rows (non-positive prices, negative volume, or
      inconsistent OHLC bounds such as `high < low`).
    - Sort chronologically from earliest to latest trading date.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None.")

    mapped_df, col_mapping, missing_req = normalize_and_map_columns(df)
    if missing_req:
        raise ValueError(
            f"Dataset is missing required columns: {missing_req}. "
            f"Expected columns: {REQUIRED_STANDARD_COLUMNS}"
        )

    work_df = mapped_df.copy()
    original_rows = int(len(work_df))
    original_cols = int(work_df.shape[1])

    has_adj_close = "adj_close" in work_df.columns
    numeric_cols = BASE_NUMERIC_COLUMNS + (["adj_close"] if has_adj_close else [])
    price_cols = BASE_PRICE_COLUMNS + (["adj_close"] if has_adj_close else [])
    checked_cols = ["date"] + numeric_cols

    # Count initial missing values and duplicate rows before cleaning
    initial_missing_per_col = {
        col: int(work_df[col].isna().sum()) for col in checked_cols
    }
    initial_missing_total = int(sum(initial_missing_per_col.values()))
    exact_duplicates = int(work_df.duplicated().sum())

    # 1. Convert date column consistently and strip timezone info while preserving local date
    if pd.api.types.is_numeric_dtype(work_df["date"]):
        raise ValueError("The 'date' column contains numeric values instead of valid calendar dates.")

    parsed_dates = pd.to_datetime(work_df["date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(parsed_dates):
        parsed_dates = pd.to_datetime(work_df["date"], errors="coerce", utc=True)

    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)

    work_df["date"] = parsed_dates
    invalid_date_rows = int(work_df["date"].isna().sum())
    work_df = work_df.dropna(subset=["date"]).copy()

    # 2. Remove exact duplicates and duplicate dates
    before_dedup = len(work_df)
    work_df = work_df.drop_duplicates()
    work_df = work_df.drop_duplicates(subset=["date"], keep="first")
    duplicates_removed = int(before_dedup - len(work_df))

    # 3. Convert numeric columns safely
    for col in numeric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    missing_numeric_rows = int(work_df[numeric_cols].isna().any(axis=1).sum())
    work_df = work_df.dropna(subset=numeric_cols).copy()

    # 4. Detect and remove invalid prices (<= 0), negative volume (< 0), or inconsistent OHLC
    if not work_df.empty:
        invalid_price_mask = (work_df[price_cols] <= 0).any(axis=1)
        invalid_volume_mask = work_df["volume"] < 0
        invalid_ohlc_mask = (
            (work_df["high"] < work_df["low"])
            | (work_df["high"] < work_df["open"] - 1e-6)
            | (work_df["high"] < work_df["close"] - 1e-6)
            | (work_df["low"] > work_df["open"] + 1e-6)
            | (work_df["low"] > work_df["close"] + 1e-6)
        )
        invalid_mask = invalid_price_mask | invalid_volume_mask | invalid_ohlc_mask
        invalid_value_rows = int(invalid_mask.sum())
        work_df = work_df.loc[~invalid_mask].copy()
    else:
        invalid_value_rows = 0

    if work_df.empty:
        raise ValueError("No valid observations remain after data cleaning.")

    if len(work_df) < min_valid_rows:
        raise ValueError(
            f"Dataset has only {len(work_df)} valid rows after cleaning, "
            f"which is fewer than the minimum required ({min_valid_rows} rows)."
        )

    # 5. Sort chronologically
    work_df = work_df.sort_values("date", ascending=True).reset_index(drop=True)
    work_df["volume"] = work_df["volume"].round().astype(np.int64)

    # Keep only standardized columns in a clean order
    kept_cols = ["date", "open", "high", "low", "close"] + (
        ["adj_close"] if has_adj_close else []
    ) + ["volume"]
    work_df = work_df[kept_cols].copy()

    cleaned_rows = int(len(work_df))
    total_removed_rows = int(original_rows - cleaned_rows)

    summary: Dict[str, Any] = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed_cleaning": total_removed_rows,
        "num_columns_raw": original_cols,
        "has_adj_close": has_adj_close,
        "column_mapping": col_mapping,
        "missing_values_total": initial_missing_total,
        "missing_values_by_column": initial_missing_per_col,
        "missing_numeric_rows_removed": missing_numeric_rows,
        "invalid_date_rows_removed": invalid_date_rows,
        "duplicate_rows_detected": max(exact_duplicates, duplicates_removed),
        "duplicate_rows_removed": duplicates_removed,
        "invalid_price_or_volume_rows_removed": invalid_value_rows,
        "earliest_date": work_df["date"].min().strftime("%Y-%m-%d"),
        "latest_date": work_df["date"].max().strftime("%Y-%m-%d"),
        "latest_close_price": float(work_df["close"].iloc[-1]),
        "average_close_price": float(work_df["close"].mean()),
        "min_close_price": float(work_df["close"].min()),
        "max_close_price": float(work_df["close"].max()),
    }

    return work_df, summary


def run_preprocessing(
    raw_filepath: Optional[Union[str, Path]] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete preprocessing and feature engineering pipeline on a CSV file.
    """
    raw_path = Path(raw_filepath) if raw_filepath is not None else find_raw_dataset()
    raw_df = load_raw_dataset(raw_path)

    cleaned_df, summary = clean_dataset(raw_df, min_valid_rows=30)
    featured_df = engineer_features(cleaned_df, include_target=True)

    rows_removed_lags_and_target = int(len(cleaned_df) - len(featured_df))
    summary["rows_after_feature_engineering"] = int(len(featured_df))
    summary["rows_removed_for_lags_and_final_target"] = rows_removed_lags_and_target
    summary["total_rows_removed"] = int(summary["original_rows"] - len(featured_df))
    summary["num_columns_final"] = int(featured_df.shape[1])
    summary["feature_columns"] = FEATURE_COLUMNS
    summary["target_column"] = TARGET_COLUMN
    summary["raw_dataset_filename"] = raw_path.name
    summary["clean_dataset_filename"] = CLEAN_DATASET_PATH.name
    summary["earliest_modeling_date"] = featured_df["date"].min().strftime("%Y-%m-%d")
    summary["latest_modeling_date"] = featured_df["date"].max().strftime("%Y-%m-%d")
    summary["latest_raw_date"] = cleaned_df["date"].max().strftime("%Y-%m-%d")

    if save_outputs:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        featured_df.to_csv(CLEAN_DATASET_PATH, index=False)
        with open(QUALITY_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return featured_df, summary


def main() -> None:
    """CLI entry point for running `python -m src.preprocessing`."""
    print("=" * 60)
    print("Stock Return Prediction - Data Preprocessing & Feature Engineering")
    print("=" * 60)

    featured_df, summary = run_preprocessing(save_outputs=True)

    print(f"Raw Dataset File           : {summary['raw_dataset_filename']}")
    print(f"Original Rows              : {summary['original_rows']:,}")
    print(f"Cleaned Valid Rows         : {summary['cleaned_rows']:,}")
    print(f"Rows Removed (Cleaning)    : {summary['rows_removed_cleaning']:,}")
    print(f"Duplicate Rows Detected    : {summary['duplicate_rows_detected']:,}")
    print(f"Missing Values Detected    : {summary['missing_values_total']:,}")
    print(f"Rows Removed (Lags/Target) : {summary['rows_removed_for_lags_and_final_target']:,}")
    print(f"Final Modeling Dataset     : {summary['rows_after_feature_engineering']:,} rows x {summary['num_columns_final']} columns")
    print(f"Raw Date Range             : {summary['earliest_date']} to {summary['latest_date']}")
    print(f"Modeling Date Range        : {summary['earliest_modeling_date']} to {summary['latest_modeling_date']}")
    print("=" * 60)


if __name__ == "__main__":
    main()