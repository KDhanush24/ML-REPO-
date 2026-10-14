"""
Model evaluation module for Reliance Industries Stock Return Prediction.

Evaluates trained Ridge Regression and Lasso Regression pipelines on the
chronologically separated unseen test dataset (last 20% of observations).

Metrics computed:
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. R² Score (Coefficient of Determination)
5. Directional Accuracy (%):
   Percentage of test trading days where the sign of the predicted return
   matches the sign of the actual return:
   `mean(sign(y_pred) == sign(y_true)) * 100`.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.eda import GRAPHS_DIR, plot_actual_vs_predicted_returns
from src.preprocessing import (
    CLEAN_DATASET_PATH,
    FEATURE_COLUMNS,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    run_preprocessing,
)
from src.train import (
    LASSO_MODEL_PATH,
    METADATA_PATH,
    RIDGE_MODEL_PATH,
    chronological_train_test_split,
)

MODEL_METRICS_CSV_PATH = OUTPUTS_DIR / "model_metrics.csv"
PREDICTIONS_CSV_PATH = OUTPUTS_DIR / "predictions.csv"


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Directional Accuracy as a percentage (0 to 100).

    Definition:
        A prediction is directionally correct if:
        - Both actual and predicted returns are positive (> 0), OR
        - Both actual and predicted returns are negative (< 0), OR
        - Both are zero.
        Formally: `np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0`.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute directional accuracy on empty arrays.")
    return float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr)) * 100.0)


def compute_regression_metrics(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute MAE, MSE, RMSE, R² Score, and Directional Accuracy for a model."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = calculate_directional_accuracy(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2,
        "Directional_Accuracy_Pct": dir_acc,
    }


def plot_model_comparison_bar_chart(
    metrics_df: pd.DataFrame, save_dir: Path = GRAPHS_DIR
) -> Path:
    """Generate and save a side-by-side bar chart comparing Ridge and Lasso errors & R²."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "8_model_comparison_bar_chart.png"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    models = metrics_df["Model"].tolist()
    colors = ["#2980b9", "#e74c3c"]

    # Subplot 1: MAE & RMSE
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width / 2, metrics_df["MAE"], width, label="MAE", color="#3498db")
    axes[0].bar(x + width / 2, metrics_df["RMSE"], width, label="RMSE", color="#1f4e78")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("Prediction Error (Lower is Better)")
    axes[0].set_ylabel("Error (Return Units)")
    axes[0].legend()

    # Subplot 2: R2 Score
    axes[1].bar(models, metrics_df["R2_Score"], color=colors, width=0.45)
    axes[1].axhline(0.0, color="#2c3e50", linestyle="--", linewidth=1.0)
    axes[1].set_title("R² Score on Unseen Test Set")
    axes[1].set_ylabel("R² Score")

    # Subplot 3: Directional Accuracy (%)
    axes[2].bar(models, metrics_df["Directional_Accuracy_Pct"], color=colors, width=0.45)
    axes[2].axhline(50.0, color="#2c3e50", linestyle="--", linewidth=1.0, label="50% Coin-Flip Baseline")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Directional Accuracy (%)")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].legend(loc="lower right")

    fig.suptitle("Ridge vs. Lasso Regression - Unseen Test Set Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_trained_models(
    ridge_pipeline: Optional[Pipeline] = None,
    lasso_pipeline: Optional[Pipeline] = None,
    test_df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate trained Ridge and Lasso pipelines on the chronological test set,
    save metrics and predictions to `outputs/`, and generate evaluation plots.
    """
    if ridge_pipeline is None or lasso_pipeline is None:
        if not RIDGE_MODEL_PATH.exists() or not LASSO_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model files were not found in 'models/'. "
                "Please run `python -m src.train` first to train and save the models."
            )
        ridge_pipeline = joblib.load(RIDGE_MODEL_PATH)
        lasso_pipeline = joblib.load(LASSO_MODEL_PATH)

    if test_df is None:
        if CLEAN_DATASET_PATH.exists():
            df = pd.read_csv(CLEAN_DATASET_PATH)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df, _ = run_preprocessing(save_outputs=True)
        _, test_df = chronological_train_test_split(df, train_ratio=0.80)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    ridge_preds = ridge_pipeline.predict(X_test)
    lasso_preds = lasso_pipeline.predict(X_test)

    ridge_metrics = compute_regression_metrics("Ridge Regression", y_test, ridge_preds)
    lasso_metrics = compute_regression_metrics("Lasso Regression", y_test, lasso_preds)

    metrics_df = pd.DataFrame([ridge_metrics, lasso_metrics])

    # Determine which model achieved lower RMSE (and higher R²) on the unseen test data
    if ridge_metrics["RMSE"] < lasso_metrics["RMSE"]:
        best_model_name = "Ridge Regression"
    elif lasso_metrics["RMSE"] < ridge_metrics["RMSE"]:
        best_model_name = "Lasso Regression"
    else:
        best_model_name = (
            "Ridge Regression"
            if ridge_metrics["MAE"] <= lasso_metrics["MAE"]
            else "Lasso Regression"
        )

    explanation = (
        f"On the unseen chronological test dataset ({len(test_df):,} trading days), "
        f"{best_model_name} achieved a slightly lower Root Mean Squared Error "
        f"(Ridge RMSE: {ridge_metrics['RMSE']:.6f} vs. Lasso RMSE: {lasso_metrics['RMSE']:.6f}) "
        f"and R² Score (Ridge R²: {ridge_metrics['R2_Score']:.6f} vs. Lasso R²: {lasso_metrics['R2_Score']:.6f}). "
        "Notice that R² is close to zero (or slightly negative) for both models. In empirical finance, "
        "daily stock returns are dominated by unpredictable news and market noise (consistent with the "
        "Efficient Market Hypothesis), so regularized models intentionally shrink coefficients toward zero "
        "to avoid overfitting. Historical price patterns alone cannot guarantee reliable future stock predictions."
    )

    predictions_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d"),
            "close": test_df["close"].to_numpy(dtype=float),
            "actual_return": y_test,
            "ridge_predicted_return": ridge_preds,
            "lasso_predicted_return": lasso_preds,
            "ridge_error": y_test - ridge_preds,
            "lasso_error": y_test - lasso_preds,
        }
    )

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(MODEL_METRICS_CSV_PATH, index=False)
        predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

        # Generate and save the actual vs predicted graph and comparison bar chart
        plot_actual_vs_predicted_returns(predictions_df)
        plot_model_comparison_bar_chart(metrics_df)

        # Update model_metadata.json with evaluation results if present
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata["evaluation"] = {
            "ridge": ridge_metrics,
            "lasso": lasso_metrics,
            "best_model": best_model_name,
            "explanation": explanation,
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "ridge_metrics": ridge_metrics,
        "lasso_metrics": lasso_metrics,
        "best_model_name": best_model_name,
        "explanation": explanation,
    }


def main() -> None:
    """CLI entry point for `python -m src.evaluate`."""
    print("=" * 60)
    print("Stock Return Prediction - Model Evaluation on Unseen Test Data")
    print("=" * 60)

    eval_result = evaluate_trained_models(save_outputs=True)
    print("\nModel Comparison Table:")
    print(eval_result["metrics_df"].to_string(index=False))
    print(f"\nBetter Performing Model: {eval_result['best_model_name']}")
    print("\nInterpretation:")
    print(eval_result["explanation"])
    print("-" * 60)
    print(f"Saved Metrics CSV     : {MODEL_METRICS_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved Predictions CSV : {PREDICTIONS_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
