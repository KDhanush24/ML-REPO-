"""
Automated unit and integration tests for model training, scaler isolation,
evaluation metrics, model persistence, and inference (`src/train.py`,
`src/evaluate.py`, and `src/predict.py`).
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
from src.feature_engineering import FEATURE_COLUMNS
from src.predict import (
    REQUIRED_DISCLAIMER,
    get_default_prediction_inputs,
    load_trained_model,
    predict_next_day_return,
    validate_and_build_features,
)
from src.preprocessing import CLEAN_DATASET_PATH, run_preprocessing
from src.report import generate_summary_report
from src.train import (
    chronological_train_test_split,
    train_and_tune_models,
)


def test_scaler_fitted_strictly_on_training_data() -> None:
    """Verify StandardScaler is fitted only on the chronological training split."""
    if not CLEAN_DATASET_PATH.exists():
        run_preprocessing(save_outputs=True)

    df = pd.read_csv(CLEAN_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    train_df, test_df = chronological_train_test_split(df, train_ratio=0.80)

    for model_name in ["Ridge Regression", "Lasso Regression"]:
        pipeline = load_trained_model(model_name)
        scaler = pipeline.named_steps["scaler"]
        assert scaler.n_samples_seen_ == len(train_df)
        assert np.allclose(
            scaler.mean_, train_df[FEATURE_COLUMNS].mean().to_numpy(), rtol=1e-5
        )


def test_train_and_evaluate_pipeline_execution() -> None:
    """Verify train_and_tune_models and evaluate_trained_models return real valid metrics."""
    df = pd.read_csv(CLEAN_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    bundle = train_and_tune_models(df, train_ratio=0.80, n_splits=3, save_models=False)
    eval_res = evaluate_trained_models(
        ridge_pipeline=bundle["ridge_pipeline"],
        lasso_pipeline=bundle["lasso_pipeline"],
        test_df=bundle["test_df"],
        save_outputs=False,
    )

    metrics_df = eval_res["metrics_df"]
    assert len(metrics_df) == 2
    for metric_col in ["MAE", "MSE", "RMSE", "R2_Score"]:
        assert metric_col in metrics_df.columns
        assert np.isfinite(metrics_df[metric_col]).all()

    # Verify directional accuracy calculation helper
    y_true = np.array([0.02, -0.01, 0.03, -0.02])
    y_pred = np.array([0.01, 0.01, 0.02, -0.01])
    # signs match at indices 0, 2, 3 -> 3/4 = 75%
    assert np.isclose(calculate_directional_accuracy(y_true, y_pred), 75.0)


def test_predict_modes_and_disclaimer() -> None:
    """Verify prediction produces finite returns and contains the required disclaimer."""
    defaults = get_default_prediction_inputs()
    for model_name in ["Ridge Regression", "Lasso Regression"]:
        res = predict_next_day_return(defaults, model_choice=model_name)
        assert res["selected_model"] == model_name
        assert np.isfinite(res["predicted_return"])
        assert np.isfinite(res["predicted_return_pct"])
        assert res["estimated_next_close"] > 0
        assert res["disclaimer"] == REQUIRED_DISCLAIMER
        assert "estimates" in res["disclaimer"].lower()


@pytest.mark.parametrize(
    "bad_input,err_pattern",
    [
        ({"open": -10.0}, "positive"),
        ({"high": 90.0, "low": 110.0}, "High price"),
        ({"volume": -500}, "negative"),
        ({"prev_close": -5.0}, "prev_close"),
    ],
)
def test_prediction_input_validation(bad_input: dict, err_pattern: str) -> None:
    """Verify manual user inputs reject negative prices, negative volumes, and bad bounds."""
    inputs = get_default_prediction_inputs()
    inputs.update(bad_input)
    with pytest.raises(ValueError, match=err_pattern):
        validate_and_build_features(inputs)


def test_summary_report_generation() -> None:
    """Verify that generate_summary_report creates a complete report without errors."""
    eval_res = evaluate_trained_models(save_outputs=False)
    meta = eval_res["metrics_df"].iloc[0].to_dict()
    fake_training_meta = {
        "train_rows": 5854,
        "test_rows": 1464,
        "train_start_date": "1996-01-08",
        "train_end_date": "2019-03-27",
        "test_start_date": "2019-03-28",
        "test_end_date": "2025-02-27",
        "ridge": {"best_alpha": 10.0},
        "lasso": {"best_alpha": 0.0005, "non_zero_features": 6},
    }
    fake_prep_summary = {
        "original_rows": 7324,
        "cleaned_rows": 7324,
        "rows_removed_cleaning": 0,
        "num_columns_raw": 7,
        "missing_values_total": 0,
        "missing_numeric_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "invalid_date_rows_removed": 0,
        "invalid_price_or_volume_rows_removed": 0,
        "earliest_date": "1996-01-01",
        "latest_date": "2025-02-28",
        "latest_close_price": 1200.10,
        "average_close_price": 500.25,
        "min_close_price": 5.48,
        "max_close_price": 1608.80,
    }
    report = generate_summary_report(
        dataset_filename="RELIANCE.NS_1973-05-08_2025-03-01.csv",
        prep_summary=fake_prep_summary,
        training_metadata=fake_training_meta,
        eval_result=eval_res,
        save_to_disk=False,
    )
    assert "Summary Report" in report
    assert "Ridge Regression" in report
    assert "Lasso Regression" in report
    assert REQUIRED_DISCLAIMER in report
