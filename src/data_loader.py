"""
Data loading, column-name normalization, and dataset validation module
for the Stock Return Prediction Using Ridge & Lasso Regression web application.

Supports:
1. Loading any uploaded CSV file (BytesIO / UploadedFile / file path).
2. Normalizing and mapping common stock-dataset column name variations
   (such as `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`,
   `Adjusted Close`, `Volume`) to standardized names.
3. Validating that required columns (`date`, `open`, `high`, `low`, `close`,
   `volume`) exist, while treating `adj_close` as optional.
4. Computing comprehensive raw dataset analysis metrics for Page 2.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_RAW_FILENAME = "RELIANCE.NS_1973-05-08_2025-03-01.csv"

# Required standardized columns (`adj_close` is optional)
REQUIRED_STANDARD_COLUMNS: List[str] = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_STANDARD_COLUMNS: List[str] = ["adj_close"]

# Acceptable aliases for each standardized column name (after lower/underscore normalization)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": [
        "date",
        "datetime",
        "timestamp",
        "trading_date",
        "trade_date",
        "time",
        "day",
    ],
    "open": [
        "open",
        "open_price",
        "opening_price",
        "first",
        "open_inr",
    ],
    "high": [
        "high",
        "high_price",
        "highest_price",
        "max",
        "max_price",
    ],
    "low": [
        "low",
        "low_price",
        "lowest_price",
        "min",
        "min_price",
    ],
    "close": [
        "close",
        "close_price",
        "closing_price",
        "last",
        "last_price",
        "settle",
        "close_last",
    ],
    "adj_close": [
        "adj_close",
        "adjusted_close",
        "adjclose",
        "adj_close_price",
        "adjusted_closing_price",
        "adj_closing_price",
    ],
    "volume": [
        "volume",
        "vol",
        "traded_volume",
        "total_trade_quantity",
        "shares_traded",
        "no_of_shares",
        "quantity",
    ],
}


def _clean_col_token(col_name: str) -> str:
    """Normalize a column header to lowercase alphanumeric tokens separated by underscores."""
    cleaned = str(col_name).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def normalize_and_map_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Normalize column names and map recognized variations to standardized names:
    `date`, `open`, `high`, `low`, `close`, `volume`, and optional `adj_close`.

    Returns:
        mapped_df: Copy of DataFrame with standardized column names.
        column_mapping: Dict mapping original column name -> standardized column name.
        missing_required: List of required standardized columns that could not be found.
    """
    if df is None or df.empty and len(df.columns) == 0:
        return pd.DataFrame(), {}, list(REQUIRED_STANDARD_COLUMNS)

    alias_to_standard: Dict[str, str] = {}
    for std_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_to_standard[alias] = std_col

    column_mapping: Dict[str, str] = {}
    used_standard_cols = set()
    new_columns: List[str] = []

    for orig_col in df.columns:
        token = _clean_col_token(orig_col)
        if token in alias_to_standard and alias_to_standard[token] not in used_standard_cols:
            std_name = alias_to_standard[token]
            column_mapping[str(orig_col)] = std_name
            used_standard_cols.add(std_name)
            new_columns.append(std_name)
        else:
            new_columns.append(token if token else str(orig_col))

    mapped_df = df.copy()
    mapped_df.columns = new_columns

    missing_required = [
        req for req in REQUIRED_STANDARD_COLUMNS if req not in mapped_df.columns
    ]
    return mapped_df, column_mapping, missing_required


def validate_uploaded_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate whether an uploaded DataFrame is a valid stock-market dataset.

    Checks:
    1. Non-empty DataFrame with at least 1 row and required columns (`date`, `open`,
       `high`, `low`, `close`, `volume`).
    2. `date` column contains parseable dates (not arbitrary strings/numbers).
    3. Price columns contain parseable numeric values.
    4. Sufficient observations exist (>= 30 rows recommended for lagged features + train/test split).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or (df.empty and len(df.columns) == 0):
        return {
            "is_valid": False,
            "missing_required_columns": list(REQUIRED_STANDARD_COLUMNS),
            "column_mapping": {},
            "has_adj_close": False,
            "row_count": 0,
            "col_count": 0,
            "errors": ["The uploaded CSV file is empty or contains no columns."],
            "warnings": [],
        }

    mapped_df, column_mapping, missing_required = normalize_and_map_columns(df)
    row_count = int(len(df))
    col_count = int(df.shape[1])

    if row_count == 0:
        errors.append("The uploaded CSV file contains column headers but 0 data rows.")

    if missing_required:
        errors.append(
            f"Missing required stock-market columns: {missing_required}. "
            f"An uploaded CSV must contain columns corresponding to: "
            f"{REQUIRED_STANDARD_COLUMNS} (optional: 'adj_close')."
        )
    else:
        # Check that date column has valid dates (and is not just small integers)
        raw_date_series = mapped_df["date"]
        if pd.api.types.is_numeric_dtype(raw_date_series):
            errors.append(
                "The 'date' column contains plain numbers instead of valid calendar dates."
            )
        else:
            parsed_dates = pd.to_datetime(raw_date_series, errors="coerce", utc=True)
            valid_dates_count = int(parsed_dates.notna().sum())
            if valid_dates_count == 0:
                errors.append(
                    "None of the values in the 'date' column could be parsed as valid calendar dates."
                )
            elif valid_dates_count < 30:
                errors.append(
                    f"Only {valid_dates_count} valid date rows were found. "
                    "At least 30 valid trading days are required to engineer lagged features and split into training/testing sets."
                )

        # Check that close column has valid positive numeric prices
        close_numeric = pd.to_numeric(mapped_df["close"], errors="coerce")
        valid_positive_close = int((close_numeric > 0).sum())
        if valid_positive_close == 0:
            errors.append(
                "The 'close' column does not contain any valid positive stock prices."
            )
        elif valid_positive_close < 30:
            errors.append(
                f"Only {valid_positive_close} positive closing prices were found. "
                "At least 30 valid observations are required to train and test the models."
            )

    has_adj_close = "adj_close" in mapped_df.columns
    if not has_adj_close and not missing_required:
        warnings.append(
            "Optional column 'adj_close' was not found in the uploaded file. "
            "The application will proceed using the required OHLCV columns."
        )

    return {
        "is_valid": len(errors) == 0,
        "missing_required_columns": missing_required,
        "column_mapping": column_mapping,
        "has_adj_close": has_adj_close,
        "row_count": row_count,
        "col_count": col_count,
        "errors": errors,
        "warnings": warnings,
    }


def read_uploaded_csv(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "uploaded_dataset.csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a CSV file from bytes, a file-like object, or a filesystem path,
    compute its dataset hash and file size, and validate its structure.
    """
    raw_bytes: bytes
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Please upload a '.csv' file."
            )
        raw_bytes = path.read_bytes()
        filename = path.name
    elif isinstance(file_source, bytes):
        raw_bytes = file_source
    elif hasattr(file_source, "getvalue"):
        raw_bytes = file_source.getvalue()
        filename = getattr(file_source, "name", filename)
    elif hasattr(file_source, "read"):
        content = file_source.read()
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
        filename = getattr(file_source, "name", filename)
    else:
        raise ValueError("Unsupported file source provided to read_uploaded_csv.")

    if not str(filename).lower().endswith(".csv"):
        raise ValueError(
            f"Unsupported file format for '{filename}'. Only CSV (.csv) files are accepted."
        )

    file_size_bytes = len(raw_bytes)
    if file_size_bytes == 0:
        raise ValueError(f"The uploaded CSV file '{filename}' is empty (0 bytes).")

    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The uploaded CSV file '{filename}' contains no parseable data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV structure in '{filename}': {exc}") from exc
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
        except Exception as exc:
            raise ValueError(f"Could not decode CSV file '{filename}': {exc}") from exc

    validation = validate_uploaded_dataframe(df)
    validation["filename"] = filename
    validation["file_size_bytes"] = file_size_bytes
    validation["file_size_kb"] = round(file_size_bytes / 1024.0, 2)
    validation["dataset_hash"] = dataset_hash

    return df, validation


def compute_raw_dataset_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the complete dataset analysis metrics required for Page 2
    without modifying the original uploaded DataFrame.
    """
    mapped_df, col_map, missing_req = normalize_and_map_columns(df)
    if missing_req:
        raise ValueError(
            f"Cannot analyze dataset because required columns are missing: {missing_req}"
        )

    total_rows = int(len(df))
    total_cols = int(df.shape[1])

    # Parse dates safely for summary
    parsed_dates = pd.to_datetime(mapped_df["date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(parsed_dates):
        parsed_dates = pd.to_datetime(mapped_df["date"], errors="coerce", utc=True)
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)

    valid_dates = parsed_dates.dropna()
    earliest_date = (
        valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "N/A"
    )
    latest_date = (
        valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "N/A"
    )

    # Closing price statistics
    close_num = pd.to_numeric(mapped_df["close"], errors="coerce")
    valid_close = close_num[close_num > 0].dropna()
    valid_close_count = int(len(valid_close))
    avg_close = float(valid_close.mean()) if valid_close_count > 0 else 0.0
    min_close = float(valid_close.min()) if valid_close_count > 0 else 0.0
    max_close = float(valid_close.max()) if valid_close_count > 0 else 0.0

    # Missing values and duplicate rows on the original DataFrame
    missing_by_col = df.isna().sum()
    missing_total = int(missing_by_col.sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_dates = int(valid_dates.duplicated().sum())

    col_info_df = pd.DataFrame(
        [
            {
                "Original Column Name": str(orig_col),
                "Standardized Name": col_map.get(str(orig_col), _clean_col_token(orig_col)),
                "Data Type": str(df[orig_col].dtype),
                "Non-Null Count": int(df[orig_col].notna().sum()),
                "Missing Count": int(df[orig_col].isna().sum()),
                "Missing (%)": round(float(df[orig_col].isna().mean() * 100.0), 2),
            }
            for orig_col in df.columns
        ]
    )

    # Numeric statistical summary
    numeric_work = pd.DataFrame()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in mapped_df.columns:
            numeric_work[col] = pd.to_numeric(mapped_df[col], errors="coerce")
    stats_summary_df = numeric_work.describe().T

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "missing_values_total": missing_total,
        "duplicate_rows": max(duplicate_rows, duplicate_dates),
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_dates": duplicate_dates,
        "valid_closing_prices": valid_close_count,
        "average_closing_price": avg_close,
        "minimum_closing_price": min_close,
        "maximum_closing_price": max_close,
        "column_info_df": col_info_df,
        "stats_summary_df": stats_summary_df,
        "column_mapping": col_map,
    }
