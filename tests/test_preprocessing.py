"""
Automated unit and integration tests for CSV upload loading, column
normalization, dataset validation, data cleaning, and edge cases
(`src/data_loader.py` and `src/preprocessing.py`).
"""

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    compute_raw_dataset_analysis,
    normalize_and_map_columns,
    read_uploaded_csv,
    validate_uploaded_dataframe,
)
from src.preprocessing import (
    DEFAULT_RAW_FILENAME,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    engineer_features,
    find_raw_dataset,
    load_raw_dataset,
)
from src.train import chronological_train_test_split


def make_synthetic_stock_df(n_rows: int = 35, include_adj_close: bool = True) -> pd.DataFrame:
    """Create a clean, deterministic synthetic stock DataFrame for unit tests."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    close = np.linspace(100.0, 134.0, n_rows)
    data = {
        "date": [d.strftime("%Y-%m-%d 00:00:00+05:30") for d in dates],
        "open": close - 0.5,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "volume": np.arange(10000, 10000 + n_rows * 100, 100),
    }
    if include_adj_close:
        data["adj_close"] = close * 0.98
    return pd.DataFrame(data)


def test_find_and_load_real_raw_dataset() -> None:
    """Verify the raw Reliance CSV file is found (not the cleaned file) and loads cleanly."""
    raw_path = find_raw_dataset()
    assert raw_path.name == DEFAULT_RAW_FILENAME
    assert not raw_path.name.startswith("clean")

    df = load_raw_dataset(raw_path)
    assert len(df) == 7324
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_column_normalization_and_mapping_variations() -> None:
    """
    Verify that variations like 'Date', 'Open', 'High', 'Low', 'Close',
    'Adjusted Close', and 'Volume' are mapped to standard names, and that
    'adj_close' is optional.
    """
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    close = np.linspace(200.0, 240.0, 40)
    custom_df = pd.DataFrame(
        {
            "Trading Date": dates.strftime("%Y-%m-%d"),
            "Open Price": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Adjusted Close": close * 0.99,
            "Traded Volume": np.full(40, 50000),
        }
    )

    mapped_df, col_map, missing = normalize_and_map_columns(custom_df)
    assert missing == []
    assert col_map["Trading Date"] == "date"
    assert col_map["Open Price"] == "open"
    assert col_map["Adjusted Close"] == "adj_close"
    assert col_map["Traded Volume"] == "volume"

    # Also test without Adjusted Close (optional)
    no_adj_df = custom_df.drop(columns=["Adjusted Close"])
    val_info = validate_uploaded_dataframe(no_adj_df)
    assert val_info["is_valid"] is True
    assert val_info["has_adj_close"] is False
    cleaned_no_adj, summary_no_adj = clean_dataset(no_adj_df)
    assert len(cleaned_no_adj) == 40
    assert summary_no_adj["has_adj_close"] is False


def test_csv_upload_and_invalid_dataset_messages() -> None:
    """Verify CSV upload from memory works for valid datasets and produces helpful errors for invalid ones."""
    valid_df = make_synthetic_stock_df(40, include_adj_close=False)
    csv_bytes = valid_df.to_csv(index=False).encode("utf-8")
    loaded_df, val = read_uploaded_csv(io.BytesIO(csv_bytes), filename="custom_stock.csv")

    assert val["is_valid"] is True
    assert val["row_count"] == 40
    assert len(loaded_df) == 40

    analysis = compute_raw_dataset_analysis(loaded_df)
    assert analysis["total_rows"] == 40
    assert analysis["valid_closing_prices"] == 40

    # Unrelated CSV (missing price/volume columns)
    unrelated_csv = b"student_id,name,marks\n1,Alice,95\n2,Bob,88\n"
    _, bad_val = read_uploaded_csv(unrelated_csv, filename="students.csv")
    assert bad_val["is_valid"] is False
    assert len(bad_val["missing_required_columns"]) > 0
    assert any("Missing required stock-market columns" in e for e in bad_val["errors"])


def test_clean_dataset_sorts_and_normalizes_dates() -> None:
    """Verify reverse-ordered dates are sorted chronologically and stripped of timezone."""
    df = make_synthetic_stock_df(12).iloc[::-1].reset_index(drop=True)
    cleaned_df, summary = clean_dataset(df)

    assert len(cleaned_df) == 12
    assert cleaned_df["date"].is_monotonic_increasing
    assert getattr(cleaned_df["date"].dt, "tz", None) is None
    assert summary["original_rows"] == 12
    assert summary["cleaned_rows"] == 12


def test_target_variable_and_feature_engineering_no_leakage() -> None:
    """
    Verify target_return = (next_close / current_close - 1), final row is dropped,
    no missing values remain, and changing a future price does NOT alter past features.
    """
    raw_df = make_synthetic_stock_df(15)
    cleaned_df, _ = clean_dataset(raw_df)
    feat_df = engineer_features(cleaned_df, include_target=True)

    assert len(feat_df) == 9
    assert not feat_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()

    idx_in_clean = 5
    expected_daily_ret = cleaned_df.loc[idx_in_clean, "close"] / cleaned_df.loc[idx_in_clean - 1, "close"] - 1.0
    expected_return_5d = cleaned_df.loc[idx_in_clean, "close"] / cleaned_df.loc[idx_in_clean - 5, "close"] - 1.0
    expected_target = cleaned_df.loc[idx_in_clean + 1, "close"] / cleaned_df.loc[idx_in_clean, "close"] - 1.0

    assert np.isclose(feat_df.loc[0, "daily_return"], expected_daily_ret)
    assert np.isclose(feat_df.loc[0, "return_5d"], expected_return_5d)
    assert np.isclose(feat_df.loc[0, TARGET_COLUMN], expected_target)


def test_chronological_split_preserves_time_order() -> None:
    """Verify training observations strictly precede testing observations."""
    df = make_synthetic_stock_df(35)
    cleaned_df, _ = clean_dataset(df)
    feat_df = engineer_features(cleaned_df)

    train_df, test_df = chronological_train_test_split(feat_df, train_ratio=0.80)
    assert len(train_df) + len(test_df) == len(feat_df)
    assert train_df["date"].max() < test_df["date"].min()


def test_edge_case_missing_columns() -> None:
    """Verify missing required columns raise ValueError."""
    bad_df = pd.DataFrame({"date": ["2024-01-01"], "close": [100.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(bad_df)


def test_edge_case_empty_dataset(tmp_path: Path) -> None:
    """Verify empty DataFrame or empty CSV raises ValueError."""
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="empty"):
        clean_dataset(empty_df)

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_raw_dataset(empty_csv)


def test_edge_case_invalid_dates_missing_values_and_negative_prices() -> None:
    """
    Verify rows with invalid dates, NaN numeric values, non-positive prices,
    negative volume, or duplicate dates are removed and accurately counted.
    """
    df = make_synthetic_stock_df(15)
    df.loc[1, "date"] = "not-a-valid-date"
    df.loc[2, "close"] = np.nan
    df.loc[3, "close"] = -50.0
    df.loc[4, "open"] = 0.0
    df.loc[5, "volume"] = -100
    df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned_df, summary = clean_dataset(df_with_dup)

    assert summary["invalid_date_rows_removed"] == 1
    assert summary["duplicate_rows_removed"] == 1
    assert summary["missing_numeric_rows_removed"] == 1
    assert summary["invalid_price_or_volume_rows_removed"] == 3
    assert len(cleaned_df) == 10
    assert (cleaned_df["close"] > 0).all()
    assert (cleaned_df["volume"] >= 0).all()
