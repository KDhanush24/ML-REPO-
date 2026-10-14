"""
Automated tests for feature engineering and target variable calculation
(`src/feature_engineering.py`).

Verifies:
1. Target variable definition: `target_return = (next_close / current_close) - 1`.
2. Removal of the final observation where next-day target is unobserved.
3. No future data leakage in input features.
4. Input feature order matching `FEATURE_COLUMNS`.
5. Error handling on empty or invalid DataFrames.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    TARGET_COLUMN,
    engineer_features,
)


def make_test_df(n_rows: int = 20) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing feature engineering."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    close = np.linspace(50.0, 70.0, n_rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.linspace(1000, 5000, n_rows).astype(int),
        }
    )


def test_target_formula_and_row_count() -> None:
    """Verify target calculation formula and removal of warmup rows + final row."""
    df = make_test_df(20)
    feat_df = engineer_features(df, include_target=True)

    # 20 rows - 5 warmup rows (return_5d) - 1 final row (target_return) = 14 rows
    assert len(feat_df) == 14
    assert TARGET_COLUMN in feat_df.columns
    assert list(feat_df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS

    # Check exact formula on row 0 of feat_df (which is row 5 of df)
    expected_target = df.loc[6, "close"] / df.loc[5, "close"] - 1.0
    assert np.isclose(feat_df.loc[0, TARGET_COLUMN], expected_target)

    # Check that target_return is never inside FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_no_future_data_leakage() -> None:
    """Modifying a future closing price must NOT change past feature values."""
    df1 = make_test_df(20)
    df2 = df1.copy()
    # Change the final observation's price
    df2.loc[len(df2) - 1, "close"] = 999.0
    df2.loc[len(df2) - 1, "high"] = 1005.0

    feat1 = engineer_features(df1, include_target=True)
    feat2 = engineer_features(df2, include_target=True)

    # All rows before the last row must have identical features AND target
    pd.testing.assert_frame_equal(feat1.iloc[:-1], feat2.iloc[:-1])
    # The last row must have identical input features
    pd.testing.assert_series_equal(
        feat1.iloc[-1][FEATURE_COLUMNS], feat2.iloc[-1][FEATURE_COLUMNS]
    )


def test_feature_descriptions_complete() -> None:
    """Every engineered feature must have a simple-English description."""
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_DESCRIPTIONS
        assert len(FEATURE_DESCRIPTIONS[col]) > 10


def test_empty_or_too_small_dataframe_error() -> None:
    """Empty or insufficient DataFrame raises a clear ValueError."""
    with pytest.raises(ValueError, match="empty"):
        engineer_features(pd.DataFrame())

    tiny_df = make_test_df(4)
    with pytest.raises(ValueError, match="insufficient rows"):
        engineer_features(tiny_df)
