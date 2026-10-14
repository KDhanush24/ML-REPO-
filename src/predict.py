"""
Prediction and inference module for Stock Return Prediction Using Ridge & Lasso Regression.

Supports:
- Mode A: Predicting the next trading day's return using the latest historical
  observations from any uploaded and cleaned stock dataset.
- Mode B: Predicting from manual user inputs with strict validation of OHLC
  consistency, positive prices, and non-negative trading volume.
"""

import json
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    clean_dataset,
    load_raw_dataset,
)
from src.train import LASSO_MODEL_PATH, METADATA_PATH, RIDGE_MODEL_PATH

REQUIRED_DISCLAIMER: str = (
    "Machine Learning predictions are estimates based on historical data. "
    "Actual stock returns may differ substantially. This application is for "
    "educational purposes and is not investment advice."
)


def load_trained_model(model_choice: str = "Ridge Regression") -> Pipeline:
    """
    Load the saved Ridge or Lasso scikit-learn Pipeline from disk.
    """
    normalized = model_choice.strip().lower()
    if "ridge" in normalized:
        model_path = RIDGE_MODEL_PATH
        display_name = "Ridge Regression"
    elif "lasso" in normalized:
        model_path = LASSO_MODEL_PATH
        display_name = "Lasso Regression"
    else:
        raise ValueError(
            f"Unsupported model selection '{model_choice}'. "
            "Choose 'Ridge Regression' or 'Lasso Regression'."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained {display_name} model file not found at '{model_path}'. "
            "Please train the models first by running `python -m src.train`."
        )

    pipeline: Pipeline = joblib.load(model_path)
    return pipeline


def load_model_metadata() -> Dict[str, Any]:
    """Load saved model training and evaluation metadata if available."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata file not found at '{METADATA_PATH}'. "
            "Please run `python -m src.train` first."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_prediction_inputs(
    cleaned_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Derive the latest historical observation and required lagged closing prices
    from the uploaded cleaned dataset (or the default Reliance dataset if None).
    """
    if cleaned_df is None:
        raw_df = load_raw_dataset()
        cleaned_df, _ = clean_dataset(raw_df)
    else:
        cleaned_df = cleaned_df.sort_values("date", ascending=True).reset_index(drop=True)

    if len(cleaned_df) < 6:
        raise ValueError(
            "At least 6 historical trading days are required to compute the 5-day lagged return feature."
        )

    latest_row = cleaned_df.iloc[-1]
    prev_1_row = cleaned_df.iloc[-2]
    prev_2_row = cleaned_df.iloc[-3]
    prev_5_row = cleaned_df.iloc[-6]

    open_val = float(latest_row["open"])
    high_val = float(latest_row["high"])
    low_val = float(latest_row["low"])
    close_val = float(latest_row["close"])
    volume_val = float(latest_row["volume"])

    prev_close = float(prev_1_row["close"])
    prev_2_close = float(prev_2_row["close"])
    close_5d_ago = float(prev_5_row["close"])

    daily_return = close_val / prev_close - 1.0
    intraday_return = close_val / open_val - 1.0
    daily_range = high_val - low_val
    lag_return_1 = prev_close / prev_2_close - 1.0
    return_5d = close_val / close_5d_ago - 1.0

    return {
        "reference_date": pd.to_datetime(latest_row["date"]).strftime("%Y-%m-%d"),
        "open": round(open_val, 4),
        "high": round(high_val, 4),
        "low": round(low_val, 4),
        "close": round(close_val, 4),
        "volume": int(volume_val),
        "prev_close": round(prev_close, 4),
        "prev_2_close": round(prev_2_close, 4),
        "close_5d_ago": round(close_5d_ago, 4),
        "daily_return": float(daily_return),
        "intraday_return": float(intraday_return),
        "daily_range": float(daily_range),
        "lag_return_1": float(lag_return_1),
        "return_5d": float(return_5d),
    }


def validate_and_build_features(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate user inputs and construct a 1-row DataFrame containing
    `FEATURE_COLUMNS` in the exact order used during training.
    """
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Prediction inputs must be a non-empty dictionary.")

    required_base = ["open", "high", "low", "close", "volume"]
    for key in required_base:
        if key not in inputs or inputs[key] is None:
            raise ValueError(f"Missing required input field: '{key}'.")
        try:
            val = float(inputs[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Input '{key}' must be a valid number.") from exc
        if not np.isfinite(val):
            raise ValueError(f"Input '{key}' must be a finite number.")

    open_p = float(inputs["open"])
    high_p = float(inputs["high"])
    low_p = float(inputs["low"])
    close_p = float(inputs["close"])
    volume = float(inputs["volume"])

    if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
        raise ValueError(
            "All stock prices (open, high, low, close) must be strictly positive (> 0)."
        )
    if volume < 0:
        raise ValueError("Trading volume cannot be negative (< 0).")
    if high_p < low_p:
        raise ValueError(
            f"High price ({high_p}) cannot be less than Low price ({low_p})."
        )
    if high_p < max(open_p, close_p) or low_p > min(open_p, close_p):
        raise ValueError(
            "High price must be >= max(open, close) and Low price must be <= min(open, close)."
        )

    for past_col in ["prev_close", "prev_2_close", "close_5d_ago"]:
        if past_col in inputs and inputs[past_col] is not None:
            try:
                p_val = float(inputs[past_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Input '{past_col}' must be a valid number.") from exc
            if not np.isfinite(p_val):
                raise ValueError(f"Input '{past_col}' must be a finite number.")
            if p_val <= 0:
                raise ValueError(
                    f"Historical closing price '{past_col}' must be strictly positive (> 0)."
                )

    intraday_return = (
        float(inputs["intraday_return"])
        if "intraday_return" in inputs and inputs["intraday_return"] is not None
        else (close_p / open_p - 1.0)
    )
    daily_range = (
        float(inputs["daily_range"])
        if "daily_range" in inputs and inputs["daily_range"] is not None
        else (high_p - low_p)
    )

    if daily_range < 0:
        raise ValueError("Daily range (high - low) cannot be negative.")

    if "daily_return" in inputs and inputs["daily_return"] is not None:
        daily_return = float(inputs["daily_return"])
    elif "prev_close" in inputs and inputs["prev_close"] is not None:
        prev_close = float(inputs["prev_close"])
        if prev_close <= 0:
            raise ValueError("Previous day closing price (prev_close) must be > 0.")
        daily_return = close_p / prev_close - 1.0
    else:
        raise ValueError(
            "Either 'daily_return' or 'prev_close' must be provided to compute today's return."
        )

    if "lag_return_1" in inputs and inputs["lag_return_1"] is not None:
        lag_return_1 = float(inputs["lag_return_1"])
    elif (
        "prev_close" in inputs
        and inputs["prev_close"] is not None
        and "prev_2_close" in inputs
        and inputs["prev_2_close"] is not None
    ):
        prev_close = float(inputs["prev_close"])
        prev_2_close = float(inputs["prev_2_close"])
        if prev_close <= 0 or prev_2_close <= 0:
            raise ValueError("Historical closing prices must be strictly positive (> 0).")
        lag_return_1 = prev_close / prev_2_close - 1.0
    else:
        raise ValueError(
            "Either 'lag_return_1' or ('prev_close' and 'prev_2_close') must be provided."
        )

    if "return_5d" in inputs and inputs["return_5d"] is not None:
        return_5d = float(inputs["return_5d"])
    elif "close_5d_ago" in inputs and inputs["close_5d_ago"] is not None:
        close_5d_ago = float(inputs["close_5d_ago"])
        if close_5d_ago <= 0:
            raise ValueError("Closing price 5 days ago (close_5d_ago) must be > 0.")
        return_5d = close_p / close_5d_ago - 1.0
    else:
        raise ValueError(
            "Either 'return_5d' or 'close_5d_ago' must be provided."
        )

    for ret_name, ret_val in [
        ("daily_return", daily_return),
        ("intraday_return", intraday_return),
        ("lag_return_1", lag_return_1),
        ("return_5d", return_5d),
    ]:
        if not np.isfinite(ret_val):
            raise ValueError(f"Feature '{ret_name}' must be a finite number.")
        if ret_val <= -1.0:
            raise ValueError(
                f"Feature '{ret_name}' ({ret_val:.4f}) cannot be <= -1.0 (-100%)."
            )

    feature_dict = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": volume,
        "daily_return": daily_return,
        "intraday_return": intraday_return,
        "daily_range": daily_range,
        "lag_return_1": lag_return_1,
        "return_5d": return_5d,
    }

    return pd.DataFrame([feature_dict])[FEATURE_COLUMNS]


def interpret_prediction(
    predicted_return: float, current_close: float, model_name: str
) -> Dict[str, Any]:
    """
    Create a simple-English interpretation of the predicted next-trading-day return.
    """
    pct = predicted_return * 100.0
    estimated_next_close = current_close * (1.0 + predicted_return)

    if pct > 0.0:
        direction = "Positive Return Estimate"
        summary = (
            f"The model estimates a positive return of approximately "
            f"{abs(pct):.2f}% ({pct:+.2f}%) for the next trading day "
            f"(implied next closing price near {estimated_next_close:,.2f} from {current_close:,.2f})."
        )
    elif pct < 0.0:
        direction = "Negative Return Estimate"
        summary = (
            f"The model estimates a negative return of approximately "
            f"{pct:.2f}% for the next trading day "
            f"(implied next closing price near {estimated_next_close:,.2f} from {current_close:,.2f})."
        )
    else:
        direction = "Neutral / Flat Return Estimate"
        summary = (
            f"The model estimates a flat return of approximately 0.00% "
            f"for the next trading day (around {estimated_next_close:,.2f})."
        )

    return {
        "direction": direction,
        "summary": summary,
        "estimated_next_close": float(estimated_next_close),
        "disclaimer": REQUIRED_DISCLAIMER,
    }


def predict_next_day_return(
    inputs: Dict[str, Any],
    model_choice: str = "Ridge Regression",
    pipeline: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """
    Validate inputs, build the exact training feature vector, and run inference
    using the selected trained Pipeline.
    """
    normalized = model_choice.strip().lower()
    display_model_name = (
        "Ridge Regression" if "ridge" in normalized else "Lasso Regression"
    )

    feature_df = validate_and_build_features(inputs)

    if pipeline is None:
        pipeline = load_trained_model(display_model_name)

    pred_value = float(pipeline.predict(feature_df)[0])
    pred_pct = float(pred_value * 100.0)
    current_close = float(feature_df["close"].iloc[0])

    interpretation_info = interpret_prediction(
        predicted_return=pred_value,
        current_close=current_close,
        model_name=display_model_name,
    )

    return {
        "selected_model": display_model_name,
        "predicted_return": pred_value,
        "predicted_return_pct": pred_pct,
        "estimated_next_close": interpretation_info["estimated_next_close"],
        "direction": interpretation_info["direction"],
        "interpretation": interpretation_info["summary"],
        "disclaimer": interpretation_info["disclaimer"],
        "prediction_horizon": "1 Trading Day Ahead (Next Trading Session, t+1)",
        "features": feature_df.iloc[0].to_dict(),
        "feature_dataframe": feature_df,
    }


def main() -> None:
    """CLI entry point for `python -m src.predict`."""
    print("=" * 60)
    print("Stock Return Prediction - Sample Next-Trading-Day Inference")
    print("=" * 60)
    defaults = get_default_prediction_inputs()
    print(f"Latest Historical Trading Date in Dataset: {defaults['reference_date']}")
    print(f"Reference Closing Price                  : {defaults['close']:,.2f}")
    print("-" * 60)
    for model_name in ["Ridge Regression", "Lasso Regression"]:
        res = predict_next_day_return(defaults, model_choice=model_name)
        print(f"Model                 : {res['selected_model']}")
        print(f"Predicted Next Return : {res['predicted_return_pct']:+.4f}% ({res['predicted_return']:+.6f})")
        print(f"Estimated Next Close  : {res['estimated_next_close']:,.2f}")
        print(f"Interpretation        : {res['interpretation']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
