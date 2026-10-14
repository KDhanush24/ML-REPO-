"""
Machine Learning training pipeline for Reliance Stock Return Prediction.

Key design principles:
1. Strictly chronological train/test split (first 80% training, last 20% testing)
   to respect time-series ordering and prevent look-ahead bias.
2. Scikit-learn Pipelines combining `StandardScaler` and `Ridge` / `Lasso`
   estimators so scaling statistics are fitted strictly on training folds.
3. Hyperparameter tuning (`alpha`) using `TimeSeriesSplit` cross-validation
   on the training dataset only.
4. Persists trained pipelines (`ridge_model.joblib`, `lasso_model.joblib`)
   and comprehensive metadata (`model_metadata.json`) inside `models/`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import (
    CLEAN_DATASET_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    run_preprocessing,
)

MODELS_DIR = PROJECT_ROOT / "models"
RIDGE_MODEL_PATH = MODELS_DIR / "ridge_model.joblib"
LASSO_MODEL_PATH = MODELS_DIR / "lasso_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Documented regularization search grids
RIDGE_ALPHA_GRID: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
LASSO_ALPHA_GRID: List[float] = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.1]


def chronological_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a chronologically sorted DataFrame into an older training set
    and a newer unseen testing set without shuffling.
    """
    if not (0.5 <= train_ratio < 0.95):
        raise ValueError("train_ratio must be between 0.50 and 0.95.")

    sorted_df = df.sort_values("date", ascending=True).reset_index(drop=True)
    if len(sorted_df) < 20:
        raise ValueError("Dataset is too small for chronological splitting (need >= 20 rows).")

    split_idx = int(len(sorted_df) * train_ratio)
    train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Create a leakage-free Pipeline with StandardScaler and Ridge Regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def build_lasso_pipeline(alpha: float = 1e-4) -> Pipeline:
    """Create a leakage-free Pipeline with StandardScaler and Lasso Regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Lasso(alpha=alpha, max_iter=10000, random_state=42)),
        ]
    )


def train_and_tune_models(
    df: Optional[pd.DataFrame] = None,
    train_ratio: float = 0.80,
    n_splits: int = 5,
    save_models: bool = True,
) -> Dict[str, Any]:
    """
    Train Ridge and Lasso Regression pipelines using TimeSeriesSplit CV on
    the chronological training split, save the models, and run evaluation.
    """
    if df is None:
        if CLEAN_DATASET_PATH.exists():
            df = pd.read_csv(CLEAN_DATASET_PATH)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df, _ = run_preprocessing(save_outputs=True)

    train_df, test_df = chronological_train_test_split(df, train_ratio=train_ratio)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Configure TimeSeriesSplit to tune alpha strictly within the training window
    actual_splits = min(n_splits, max(2, len(train_df) // 10))
    tscv = TimeSeriesSplit(n_splits=actual_splits)

    # 1. Train & Tune Ridge Regression
    ridge_search = GridSearchCV(
        estimator=build_ridge_pipeline(),
        param_grid={"regressor__alpha": RIDGE_ALPHA_GRID},
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=1,
    )
    ridge_search.fit(X_train, y_train)
    best_ridge: Pipeline = ridge_search.best_estimator_
    best_ridge_alpha = float(ridge_search.best_params_["regressor__alpha"])

    # 2. Train & Tune Lasso Regression
    lasso_search = GridSearchCV(
        estimator=build_lasso_pipeline(),
        param_grid={"regressor__alpha": LASSO_ALPHA_GRID},
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=1,
    )
    lasso_search.fit(X_train, y_train)
    best_lasso: Pipeline = lasso_search.best_estimator_
    best_lasso_alpha = float(lasso_search.best_params_["regressor__alpha"])

    # Extract learned standardized coefficients for interpretability
    ridge_coefs = {
        feat: float(coef)
        for feat, coef in zip(
            FEATURE_COLUMNS, best_ridge.named_steps["regressor"].coef_
        )
    }
    lasso_coefs = {
        feat: float(coef)
        for feat, coef in zip(
            FEATURE_COLUMNS, best_lasso.named_steps["regressor"].coef_
        )
    }
    lasso_non_zero = int(
        np.sum(np.abs(best_lasso.named_steps["regressor"].coef_) > 1e-10)
    )

    metadata: Dict[str, Any] = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_ratio": train_ratio,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start_date": pd.to_datetime(train_df["date"].min()).strftime("%Y-%m-%d"),
        "train_end_date": pd.to_datetime(train_df["date"].max()).strftime("%Y-%m-%d"),
        "test_start_date": pd.to_datetime(test_df["date"].min()).strftime("%Y-%m-%d"),
        "test_end_date": pd.to_datetime(test_df["date"].max()).strftime("%Y-%m-%d"),
        "cv_splits": actual_splits,
        "ridge": {
            "best_alpha": best_ridge_alpha,
            "alpha_search_grid": RIDGE_ALPHA_GRID,
            "cv_best_neg_mse": float(ridge_search.best_score_),
            "intercept": float(best_ridge.named_steps["regressor"].intercept_),
            "coefficients": ridge_coefs,
            "model_path": str(RIDGE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "lasso": {
            "best_alpha": best_lasso_alpha,
            "alpha_search_grid": LASSO_ALPHA_GRID,
            "cv_best_neg_mse": float(lasso_search.best_score_),
            "intercept": float(best_lasso.named_steps["regressor"].intercept_),
            "coefficients": lasso_coefs,
            "non_zero_features": lasso_non_zero,
            "zeroed_features": int(len(FEATURE_COLUMNS) - lasso_non_zero),
            "model_path": str(LASSO_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_ridge, RIDGE_MODEL_PATH)
        joblib.dump(best_lasso, LASSO_MODEL_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "ridge_pipeline": best_ridge,
        "lasso_pipeline": best_lasso,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": metadata,
    }


def main() -> None:
    """CLI entry point for `python -m src.train`."""
    print("=" * 60)
    print("Stock Return Prediction - Model Training (Ridge & Lasso)")
    print("=" * 60)

    training_bundle = train_and_tune_models(save_models=True)
    meta = training_bundle["metadata"]

    print(f"Training Set : {meta['train_rows']:,} rows ({meta['train_start_date']} to {meta['train_end_date']})")
    print(f"Testing Set  : {meta['test_rows']:,} rows ({meta['test_start_date']} to {meta['test_end_date']})")
    print(f"Features ({len(meta['feature_columns'])}): {', '.join(meta['feature_columns'])}")
    print("-" * 60)
    print(f"Ridge Best Alpha (TimeSeriesSplit CV) : {meta['ridge']['best_alpha']}")
    print(f"Lasso Best Alpha (TimeSeriesSplit CV) : {meta['lasso']['best_alpha']}")
    print(f"Lasso Non-Zero Features Kept          : {meta['lasso']['non_zero_features']} / {len(meta['feature_columns'])}")
    print("-" * 60)
    print(f"Saved Ridge Pipeline : {meta['ridge']['model_path']}")
    print(f"Saved Lasso Pipeline : {meta['lasso']['model_path']}")

    # Automatically run evaluation so metrics & prediction files stay in sync
    from src.evaluate import evaluate_trained_models

    eval_result = evaluate_trained_models(
        ridge_pipeline=training_bundle["ridge_pipeline"],
        lasso_pipeline=training_bundle["lasso_pipeline"],
        test_df=training_bundle["test_df"],
        save_outputs=True,
    )
    print("\nTest Set Evaluation Summary:")
    print(eval_result["metrics_df"].to_string(index=False))
    print(f"\nSelected Best Model on Unseen Test Data: {eval_result['best_model_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
