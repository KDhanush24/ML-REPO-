"""
Report generation module for Stock Return Prediction Using Ridge & Lasso Regression.

Generates a downloadable summary report (in Markdown / plain text) built
strictly from the actual uploaded dataset, preprocessing summary, trained
model metadata, test-set evaluation metrics, and latest prediction output.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.feature_engineering import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, TARGET_COLUMN
from src.predict import REQUIRED_DISCLAIMER
from src.preprocessing import OUTPUTS_DIR

REPORTS_DIR = OUTPUTS_DIR / "reports"
DEFAULT_REPORT_PATH = REPORTS_DIR / "summary_report.md"


def generate_summary_report(
    dataset_filename: str,
    prep_summary: Dict[str, Any],
    training_metadata: Dict[str, Any],
    eval_result: Dict[str, Any],
    latest_prediction: Optional[Dict[str, Any]] = None,
    save_to_disk: bool = True,
) -> str:
    """
    Build a detailed summary report from actual dataset and model execution results.
    """
    ridge_m = eval_result["ridge_metrics"]
    lasso_m = eval_result["lasso_metrics"]
    best_model = eval_result["best_model_name"]

    winner_statement = (
        "Ridge Regression performed better on the selected test metrics."
        if "Ridge" in best_model
        else "Lasso Regression performed better on the selected test metrics."
    )

    lines = [
        "# Stock Return Prediction Using Ridge & Lasso Regression — Summary Report",
        f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Dataset Information & Date Range",
        f"- **Uploaded Dataset Filename:** `{dataset_filename}`",
        f"- **Raw Dataset Observations:** {prep_summary['original_rows']:,} rows x {prep_summary['num_columns_raw']} columns",
        f"- **Cleaned Valid Observations:** {prep_summary['cleaned_rows']:,} rows",
        f"- **Historical Date Range (Cleaned):** `{prep_summary['earliest_date']}` to `{prep_summary['latest_date']}`",
        f"- **Latest Historical Closing Price:** {prep_summary['latest_close_price']:,.2f}",
        f"- **Average Closing Price:** {prep_summary['average_close_price']:,.2f}",
        f"- **Minimum / Maximum Closing Price:** {prep_summary['min_close_price']:,.2f} / {prep_summary['max_close_price']:,.2f}",
        "",
        "## 2. Data Preprocessing Summary",
        f"- **Original Row Count:** {prep_summary['original_rows']:,}",
        f"- **Cleaned Row Count:** {prep_summary['cleaned_rows']:,}",
        f"- **Total Rows Removed (Cleaning):** {prep_summary['rows_removed_cleaning']:,}",
        f"- **Missing Values Detected & Handled:** {prep_summary['missing_values_total']:,} (rows with missing numeric values removed: {prep_summary['missing_numeric_rows_removed']:,})",
        f"- **Duplicate Records Removed:** {prep_summary['duplicate_rows_removed']:,}",
        f"- **Invalid Date Rows Removed:** {prep_summary['invalid_date_rows_removed']:,}",
        f"- **Invalid Price / Volume Rows Removed:** {prep_summary['invalid_price_or_volume_rows_removed']:,}",
        f"- **Modeling Rows (After 5-Day Lag Warmup & Final Target Removal):** {training_metadata['train_rows'] + training_metadata['test_rows']:,}",
        "",
        "## 3. Feature Engineering & Target Variable",
        f"- **Target Variable (`{TARGET_COLUMN}`):** `(next_close / current_close) - 1`",
        "- **Input Features Used (10 Leakage-Free Features Available at Close of Day t):**",
    ]

    for idx, feat in enumerate(FEATURE_COLUMNS, start=1):
        lines.append(f"  {idx}. `{feat}`: {FEATURE_DESCRIPTIONS[feat]}")

    lines.extend(
        [
            "",
            "## 4. Chronological Train / Test Split & Hyperparameter Tuning",
            f"- **Training Set (First 80%):** {training_metadata['train_rows']:,} observations (`{training_metadata['train_start_date']}` to `{training_metadata['train_end_date']}`)",
            f"- **Testing Set (Last 20% Unseen):** {training_metadata['test_rows']:,} observations (`{training_metadata['test_start_date']}` to `{training_metadata['test_end_date']}`)",
            f"- **Scaling Strategy:** `StandardScaler` inside a `scikit-learn` `Pipeline` fitted strictly on the training set.",
            f"- **Ridge Regression Best Alpha (TimeSeriesSplit CV):** `{training_metadata['ridge']['best_alpha']}`",
            f"- **Lasso Regression Best Alpha (TimeSeriesSplit CV):** `{training_metadata['lasso']['best_alpha']}` (kept {training_metadata['lasso']['non_zero_features']}/{len(FEATURE_COLUMNS)} features non-zero)",
            "",
            "## 5. Model Evaluation Metrics (Unseen Chronological Test Set)",
            "",
            "| Model | MAE | MSE | RMSE | R² Score | Directional Accuracy (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Ridge Regression** | {ridge_m['MAE']:.6f} | {ridge_m['MSE']:.6f} | {ridge_m['RMSE']:.6f} | {ridge_m['R2_Score']:.6f} | {ridge_m['Directional_Accuracy_Pct']:.2f}% |",
            f"| **Lasso Regression** | {lasso_m['MAE']:.6f} | {lasso_m['MSE']:.6f} | {lasso_m['RMSE']:.6f} | {lasso_m['R2_Score']:.6f} | {lasso_m['Directional_Accuracy_Pct']:.2f}% |",
            "",
            "## 6. Model Comparison Conclusion",
            f"- **Conclusion:** {winner_statement}",
            f"- **Detailed Analysis:** {eval_result['explanation']}",
            "",
        ]
    )

    if latest_prediction is not None:
        lines.extend(
            [
                "## 7. Latest Next-Trading-Day Return Prediction",
                f"- **Selected Model:** {latest_prediction['selected_model']}",
                f"- **Predicted Return (Decimal):** `{latest_prediction['predicted_return']:+.6f}`",
                f"- **Predicted Return (Percentage):** `{latest_prediction['predicted_return_pct']:+.2f}%`",
                f"- **Implied Next Closing Price:** `{latest_prediction['estimated_next_close']:,.2f}`",
                f"- **Interpretation:** {latest_prediction['interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 8. Prediction Methodology & Project Limitations",
            "- **Prediction Methodology:** Features are constructed strictly from historical observations up to day `t` using the exact same column order (`FEATURE_COLUMNS`) and mathematical definitions as model training, then scaled via the training-fitted `StandardScaler` inside the selected model's `Pipeline`.",
            "- **Limitations:**",
            "  1. Historical OHLCV prices do not capture breaking corporate news, earnings surprises, regulatory changes, or macroeconomic shocks.",
            "  2. Ridge and Lasso Regression model only linear relationships between engineered features and next-day returns.",
            "  3. Daily stock returns contain substantial market noise, so $R^2$ values on unseen data are naturally close to zero.",
            f"- **Disclaimer:** {REQUIRED_DISCLAIMER}",
            "",
        ]
    )

    report_text = "\n".join(lines)
    if save_to_disk:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_REPORT_PATH.write_text(report_text, encoding="utf-8")

    return report_text
