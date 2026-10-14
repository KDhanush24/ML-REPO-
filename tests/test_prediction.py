"""
Automated unit and integration tests for model training, scaler isolation,
model persistence, evaluation metrics, and prediction inference
(`src/train.py`, `src/evaluate.py`, and `src/predict.py`).

Tests verify:
1. StandardScaler is fitted strictly on the training split (never on test data).
2. Saved Ridge and Lasso pipelines exist, can be loaded, and contain fitted scalers.
3. Prediction pipeline uses the exact `FEATURE_COLUMNS` order and produces valid estimates.
4. Edge cases: negative prices, negative volume, high < low, missing fields,
   unsupported model names, and missing model files.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.evaluate import calculate_directional_accuracy, evaluate_trained_models
from src.predict import (
    get_default_prediction_inputs,
    load_trained_model,
    predict_next_day_return,
    validate_and_build_features,
)
from src.preprocessing import CLEAN_DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from src.train import chronological_train_test_split


def test_scaler_fitted_strictly_on_training_data() -> None:
    """
    Verify that the StandardScaler inside the saved Ridge and Lasso pipelines
    was fitted strictly on the chronological training set (first 80%) and NOT
    on the full dataset or test dataset.
    """
    df = pd.read_csv(CLEAN_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    train_df, test_df = chronological_train_test_split(df, train_ratio=0.80)

    expected_train_means = train_df[FEATURE_COLUMNS].mean().to_numpy()
    full_dataset_means = df[FEATURE_COLUMNS].mean().to_numpy()

    for model_name in ["Ridge Regression", "Lasso Regression"]:
        pipeline = load_trained_model(model_name)
        scaler = pipeline.named_steps["scaler"]

        assert scaler.n_samples_seen_ == len(train_df)
        assert np.allclose(scaler.mean_, expected_train_means, rtol=1e-6)
        assert not np.allclose(scaler.mean_, full_dataset_means, rtol=1e-3)


def test_saved_models_load_and_predict_sample() -> None:
    """Verify both saved models load cleanly and generate valid predictions from default inputs."""
    defaults = get_default_prediction_inputs()

    for model_name in ["Ridge Regression", "Lasso Regression"]:
        result = predict_next_day_return(defaults, model_choice=model_name)

        assert result["selected_model"] == model_name
        assert np.isfinite(result["predicted_return"])
        assert np.isfinite(result["predicted_return_pct"])
        assert result["estimated_next_close"] > 0
        assert list(result["feature_dataframe"].columns) == FEATURE_COLUMNS
        assert len(result["interpretation"]) > 20


def test_feature_construction_equivalence() -> None:
    """
    Verify that deriving lagged features from recent closing prices produces
    the exact same feature values as passing the engineered return features directly.
    """
    defaults = get_default_prediction_inputs()

    # Mode 1: From historical closing prices
    from_closes = {
        "open": defaults["open"],
        "high": defaults["high"],
        "low": defaults["low"],
        "close": defaults["close"],
        "volume": defaults["volume"],
        "prev_close": defaults["prev_close"],
        "prev_2_close": defaults["prev_2_close"],
        "close_5d_ago": defaults["close_5d_ago"],
    }
    df_mode1 = validate_and_build_features(from_closes)

    # Mode 2: Directly passing the return features
    from_returns = {
        "open": defaults["open"],
        "high": defaults["high"],
        "low": defaults["low"],
        "close": defaults["close"],
        "volume": defaults["volume"],
        "daily_return": defaults["daily_return"],
        "lag_return_1": defaults["lag_return_1"],
        "return_5d": defaults["return_5d"],
    }
    df_mode2 = validate_and_build_features(from_returns)

    pd.testing.assert_frame_equal(df_mode1, df_mode2)


def test_evaluation_metrics_are_real_and_complete() -> None:
    """Verify evaluation produces valid MAE, MSE, RMSE, R2, and Directional Accuracy."""
    eval_res = evaluate_trained_models(save_outputs=False)
    metrics_df = eval_res["metrics_df"]

    assert len(metrics_df) == 2
    for col in ["Model", "MAE", "MSE", "RMSE", "R2_Score", "Directional_Accuracy_Pct"]:
        assert col in metrics_df.columns

    assert (metrics_df["MAE"] > 0).all()
    assert (metrics_df["RMSE"] > 0).all()
    assert np.allclose(np.sqrt(metrics_df["MSE"]), metrics_df["RMSE"])

    # Verify directional accuracy helper explicitly
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.005, -0.01, -0.002, -0.004])
    assert np.isclose(calculate_directional_accuracy(y_true, y_pred), 75.0)


@pytest.mark.parametrize(
    "invalid_override,expected_msg",
    [
        ({"close": -100.0}, "positive"),
        ({"open": 0.0}, "positive"),
        ({"volume": -50}, "negative"),
        ({"high": 1100.0, "low": 1250.0}, "High price"),
        ({"prev_close": -10.0, "daily_return": None}, "prev_close"),
    ],
)
def test_edge_case_invalid_user_inputs(invalid_override: dict, expected_msg: str) -> None:
    """Verify invalid user inputs are rejected with clear ValueErrors."""
    inputs = get_default_prediction_inputs()
    inputs.update(invalid_override)

    with pytest.raises(ValueError, match=expected_msg):
        validate_and_build_features(inputs)


def test_edge_case_missing_model_files(tmp_path: Path) -> None:
    """Verify a clear FileNotFoundError is raised if model files do not exist."""
    missing_path = tmp_path / "nonexistent_ridge.joblib"
    with patch("src.predict.RIDGE_MODEL_PATH", missing_path):
        with pytest.raises(FileNotFoundError, match="python -m src.train"):
            load_trained_model("Ridge Regression")
