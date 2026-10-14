"""
Feature engineering and target variable module for Stock Return Prediction.

Constructs leakage-free technical features available at the market close
of trading day t to predict the next trading day's return (day t+1):

    target_return = (close_{t+1} / close_t) - 1

The final observation is removed because its next-day closing price is
unavailable.
"""

from typing import Dict, List

import pandas as pd

from src.data_loader import REQUIRED_STANDARD_COLUMNS, normalize_and_map_columns

# Exact list and order of input features used for training and prediction
FEATURE_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "intraday_return",
    "daily_range",
    "lag_return_1",
    "return_5d",
]

TARGET_COLUMN: str = "target_return"

# Simple-English descriptions of every engineered feature
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "open": "Opening stock price at the start of the current trading day.",
    "high": "Highest stock price reached during the current trading day.",
    "low": "Lowest stock price reached during the current trading day.",
    "close": "Closing stock price at the end of the current trading day.",
    "volume": "Total number of shares traded during the current trading day.",
    "daily_return": (
        "Current day's close-to-close return: (close_t / close_{t-1}) - 1. "
        "Measures how much the stock price changed compared to the previous trading day's close."
    ),
    "intraday_return": (
        "Current day's open-to-close return: (close_t / open_t) - 1. "
        "Measures the percentage price movement within the current trading session."
    ),
    "daily_range": (
        "Intraday price spread: high_t - low_t. "
        "Captures same-day price volatility."
    ),
    "lag_return_1": (
        "Previous trading day's daily return: (close_{t-1} / close_{t-2}) - 1. "
        "Provides 1-day short-term momentum context without looking into the future."
    ),
    "return_5d": (
        "Five-day historical return: (close_t / close_{t-5}) - 1. "
        "Measures the stock's price trend over the past 5 trading days (approximately 1 trading week)."
    ),
}

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "date": "Trading date in chronological order (timezone-normalized).",
    "open": FEATURE_DESCRIPTIONS["open"],
    "high": FEATURE_DESCRIPTIONS["high"],
    "low": FEATURE_DESCRIPTIONS["low"],
    "close": FEATURE_DESCRIPTIONS["close"],
    "adj_close": "Adjusted closing price accounting for corporate actions such as splits and dividends (optional).",
    "volume": FEATURE_DESCRIPTIONS["volume"],
    "daily_return": FEATURE_DESCRIPTIONS["daily_return"],
    "intraday_return": FEATURE_DESCRIPTIONS["intraday_return"],
    "daily_range": FEATURE_DESCRIPTIONS["daily_range"],
    "lag_return_1": FEATURE_DESCRIPTIONS["lag_return_1"],
    "return_5d": FEATURE_DESCRIPTIONS["return_5d"],
    "target_return": (
        "Next trading day's return: (close_{t+1} / close_t) - 1. "
        "This is the target variable the Machine Learning models learn to predict."
    ),
}


def engineer_features(
    df: pd.DataFrame, include_target: bool = True
) -> pd.DataFrame:
    """
    Create leakage-free features from historical price and volume data
    available at the end of the current trading day (day t).

    Target definition:
        df["target_return"] = df["close"].shift(-1) / df["close"] - 1

    Removes initial warmup rows (where lagged returns are NaN) and the final
    row (where next-day target_return is NaN).
    """
    if df is None or df.empty:
        raise ValueError("Cannot engineer features on an empty DataFrame.")

    mapped_df, _, missing_req = normalize_and_map_columns(df)
    if missing_req:
        raise ValueError(
            f"Dataset is missing required columns: {missing_req}. "
            f"Expected columns: {REQUIRED_STANDARD_COLUMNS}"
        )

    feat_df = mapped_df.copy()
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    if getattr(feat_df["date"].dt, "tz", None) is not None:
        feat_df["date"] = feat_df["date"].dt.tz_localize(None)
    feat_df = feat_df.sort_values("date", ascending=True).reset_index(drop=True)

    # Current trading day features (available at market close of day t)
    feat_df["daily_return"] = feat_df["close"] / feat_df["close"].shift(1) - 1.0
    feat_df["intraday_return"] = feat_df["close"] / feat_df["open"] - 1.0
    feat_df["daily_range"] = feat_df["high"] - feat_df["low"]

    # Historical lagged return features (available at market close of day t)
    feat_df["lag_return_1"] = feat_df["daily_return"].shift(1)
    feat_df["return_5d"] = feat_df["close"] / feat_df["close"].shift(5) - 1.0

    if include_target:
        # Next-trading-day return target (day t to day t+1)
        feat_df[TARGET_COLUMN] = (
            feat_df["close"].shift(-1) / feat_df["close"] - 1.0
        )
        required_valid_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    else:
        required_valid_cols = FEATURE_COLUMNS

    feat_df = feat_df.dropna(subset=required_valid_cols).reset_index(drop=True)

    if feat_df.empty:
        raise ValueError(
            "Dataset has insufficient rows after generating 5-day lagged features and next-day target."
        )

    return feat_df
