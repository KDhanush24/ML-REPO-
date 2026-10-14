"""
Stock Return Prediction Using Ridge & Lasso Regression
Interactive Streamlit Web Application (`app.py`).

Implements the complete 11-step user workflow across 9 guided pages:
1. Upload Dataset (CSV upload + validation + "Analyze Dataset" button)
2. Dataset Analysis (Raw dataset metrics, types, missing values, duplicates, stats)
3. Automatic Data Preprocessing (Column normalization, cleaning summary, CSV download)
4. Exploratory Data Analysis (7 dynamic charts with date filtering, interpretations & PNG download)
5. Feature Engineering (Next-day target_return & 10 leakage-free features)
6. Machine Learning Training (Chronological 80/20 split, StandardScaler + Ridge/Lasso Pipelines)
7. Model Evaluation (MAE, MSE, RMSE, R² comparison table, charts & honest interpretation)
8. Stock Return Prediction (Mode A: Latest Historical Data & Mode B: Manual Input)
9. Download Reports (Cleaned CSV, Metrics CSV, Predictions CSV, PNG Charts, Summary Report)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    DEFAULT_RAW_FILENAME,
    REQUIRED_STANDARD_COLUMNS,
    compute_raw_dataset_analysis,
    read_uploaded_csv,
)
from src.eda import (
    EDA_INTERPRETATIONS,
    GRAPHS_DIR,
    generate_eda_png_bytes,
)
from src.evaluate import (
    MODEL_METRICS_CSV_PATH,
    PREDICTIONS_CSV_PATH,
    evaluate_trained_models,
)
from src.feature_engineering import (
    COLUMN_DESCRIPTIONS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    TARGET_COLUMN,
    engineer_features,
)
from src.predict import (
    REQUIRED_DISCLAIMER,
    get_default_prediction_inputs,
    predict_next_day_return,
)
from src.preprocessing import (
    CLEAN_DATASET_PATH,
    DATASET_DIR,
    clean_dataset,
)
from src.report import generate_summary_report
from src.train import (
    LASSO_ALPHA_GRID,
    LASSO_MODEL_PATH,
    METADATA_PATH,
    MODELS_DIR,
    RIDGE_ALPHA_GRID,
    RIDGE_MODEL_PATH,
    chronological_train_test_split,
    train_and_tune_models,
)

# ---------------------------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Return Prediction Using Ridge & Lasso Regression",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1e3a8a;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    .interpretation-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        color: #0c4a6e;
        font-size: 0.94rem;
        margin-top: 0.4rem;
        margin-bottom: 1.1rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #78350f;
        font-size: 0.92rem;
        margin-top: 1rem;
    }
    .stage-pill-done {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        border-radius: 6px;
        padding: 0.35rem 0.5rem;
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
    }
    .stage-pill-pending {
        background-color: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 0.35rem 0.5rem;
        font-size: 0.78rem;
        font-weight: 500;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PAGES = [
    "1. Upload Dataset",
    "2. Dataset Analysis",
    "3. Automatic Data Preprocessing",
    "4. Exploratory Data Analysis",
    "5. Feature Engineering",
    "6. Machine Learning Training",
    "7. Model Evaluation",
    "8. Stock Return Prediction",
    "9. Download Reports",
]


# ---------------------------------------------------------------------------
# Session State Initialization & Caching Helpers
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all workflow session-state keys if not already present."""
    defaults = {
        "nav_page": PAGES[0],
        "dataset_hash": None,
        "uploaded_filename": None,
        "raw_df": None,
        "validation_info": None,
        "dataset_analyzed": False,
        "raw_analysis": None,
        "cleaned_df": None,
        "prep_summary": None,
        "featured_df": None,
        "eda_explored": False,
        "models_trained": False,
        "training_bundle": None,
        "eval_result": None,
        "prediction_completed": False,
        "latest_prediction": None,
        "reports_visited": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_downstream_state_for_new_upload(
    raw_df: pd.DataFrame, validation_info: Dict[str, Any]
) -> None:
    """
    When a user uploads a different CSV dataset, reset downstream analysis,
    preprocessing, model training, and prediction state so stale results are
    never reused across different datasets.
    """
    st.session_state["dataset_hash"] = validation_info["dataset_hash"]
    st.session_state["uploaded_filename"] = validation_info["filename"]
    st.session_state["raw_df"] = raw_df
    st.session_state["validation_info"] = validation_info
    st.session_state["dataset_analyzed"] = False
    st.session_state["raw_analysis"] = None
    st.session_state["cleaned_df"] = None
    st.session_state["prep_summary"] = None
    st.session_state["featured_df"] = None
    st.session_state["eda_explored"] = False
    st.session_state["models_trained"] = False
    st.session_state["training_bundle"] = None
    st.session_state["eval_result"] = None
    st.session_state["prediction_completed"] = False
    st.session_state["latest_prediction"] = None
    st.session_state["reports_visited"] = False


@st.cache_data(show_spinner=False)
def cached_analyze_and_preprocess(
    dataset_hash: str, raw_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Cached helper keyed by `dataset_hash` to run raw dataset analysis,
    cleaning, and feature engineering without redundant recomputation.
    """
    _ = dataset_hash
    raw_analysis = compute_raw_dataset_analysis(raw_df)
    cleaned_df, prep_summary = clean_dataset(raw_df, min_valid_rows=30)
    featured_df = engineer_features(cleaned_df, include_target=True)
    prep_summary["rows_after_feature_engineering"] = int(len(featured_df))
    prep_summary["rows_removed_for_lags_and_final_target"] = int(
        len(cleaned_df) - len(featured_df)
    )
    return {
        "raw_analysis": raw_analysis,
        "cleaned_df": cleaned_df,
        "prep_summary": prep_summary,
        "featured_df": featured_df,
    }


def render_kpi_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_indicator() -> None:
    """
    Render the 8-stage workflow progress indicator at the top of the app.
    Only marks a stage as complete after its operation has actually succeeded.
    """
    upload_done = (
        st.session_state["validation_info"] is not None
        and st.session_state["validation_info"].get("is_valid", False)
    )
    analyze_done = bool(st.session_state["dataset_analyzed"])
    preprocess_done = st.session_state["cleaned_df"] is not None
    explore_done = bool(st.session_state["eda_explored"]) or preprocess_done
    train_done = bool(st.session_state["models_trained"])
    compare_done = st.session_state["eval_result"] is not None
    predict_done = bool(st.session_state["prediction_completed"])
    download_done = bool(st.session_state["reports_visited"])

    stages = [
        ("1. Upload", upload_done),
        ("2. Analyze", analyze_done),
        ("3. Preprocess", preprocess_done),
        ("4. Explore", explore_done),
        ("5. Train", train_done),
        ("6. Compare", compare_done),
        ("7. Predict", predict_done),
        ("8. Download", download_done),
    ]

    cols = st.columns(len(stages))
    for col, (label, is_done) in zip(cols, stages):
        icon = "✅" if is_done else "⏳"
        css_class = "stage-pill-done" if is_done else "stage-pill-pending"
        with col:
            st.markdown(
                f'<div class="{css_class}">{icon} {label}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("---")


def require_analyzed_dataset() -> bool:
    """
    Guard helper for Pages 2–5: verify that a valid dataset has been uploaded
    and analyzed before showing downstream content.
    """
    if not st.session_state["dataset_analyzed"] or st.session_state["cleaned_df"] is None:
        st.warning(
            "⚠️ **Please upload and validate a stock-market CSV dataset first.**\n\n"
            "Go to **1. Upload Dataset**, upload a CSV file, and click **Analyze Dataset** to unlock this step."
        )
        if st.button("⬅️ Go to Step 1: Upload Dataset"):
            st.session_state["nav_page"] = PAGES[0]
            st.rerun()
        return False
    return True


def require_trained_models() -> bool:
    """
    Guard helper for Pages 7–8: verify that both Ridge and Lasso models have
    been trained on the currently uploaded dataset before showing evaluation or predictions.
    """
    if not require_analyzed_dataset():
        return False

    if not st.session_state["models_trained"] or st.session_state["eval_result"] is None:
        st.warning(
            "⚠️ **Models have not been trained on the current dataset yet.**\n\n"
            "Go to **6. Machine Learning Training** and click **Train Both Models** before viewing "
            "evaluation results or generating next-trading-day predictions."
        )
        if st.button("⚙️ Go to Step 6: Machine Learning Training"):
            st.session_state["nav_page"] = PAGES[5]
            st.rerun()
        return False
    return True


# ---------------------------------------------------------------------------
# PAGE 1: UPLOAD DATASET
# ---------------------------------------------------------------------------
def page_upload_dataset() -> None:
    st.title("📈 Stock Return Prediction Using Ridge & Lasso Regression")
    st.subheader(
        "Analyze historical stock market data and compare two regularized Machine Learning regression models."
    )

    st.info(
        "📤 **Upload your historical stock market dataset to begin automatic analysis and model training.**"
    )

    ucol1, ucol2 = st.columns([2.2, 1.2])
    with ucol1:
        uploaded_file = st.file_uploader(
            "Select a Raw Stock Market CSV File (.csv only)",
            type=["csv"],
            help=(
                "Expected columns: date, open, high, low, close, volume (and optional adj_close). "
                "Common capitalization variations such as Date, Open, High, Low, Close, Volume, "
                "and Adjusted Close are handled automatically."
            ),
        )
    with ucol2:
        st.markdown("##### Quick Project Dataset Option")
        st.caption(
            "Testing with the project's bundled Reliance Industries dataset? "
            "You can upload it using the uploader on the left, or load it directly below:"
        )
        load_sample_clicked = st.button(
            "📂 Load Bundled Reliance Dataset (CSV)",
            use_container_width=True,
        )

    # Process uploaded file or bundled sample button
    if uploaded_file is not None:
        try:
            raw_df, val_info = read_uploaded_csv(uploaded_file, filename=uploaded_file.name)
            if val_info["dataset_hash"] != st.session_state["dataset_hash"]:
                reset_downstream_state_for_new_upload(raw_df, val_info)
        except Exception as exc:
            st.error(f"❌ **Error Reading Uploaded CSV:** {exc}")
            return
    elif load_sample_clicked:
        sample_path = DATASET_DIR / DEFAULT_RAW_FILENAME
        if not sample_path.exists():
            st.error(f"Bundled dataset not found at `{sample_path}`.")
            return
        try:
            raw_df, val_info = read_uploaded_csv(sample_path, filename=sample_path.name)
            if val_info["dataset_hash"] != st.session_state["dataset_hash"]:
                reset_downstream_state_for_new_upload(raw_df, val_info)
        except Exception as exc:
            st.error(f"❌ **Error Loading Bundled Dataset:** {exc}")
            return

    # If a dataset is currently in session_state, display its details & validation
    if st.session_state["raw_df"] is None or st.session_state["validation_info"] is None:
        st.markdown(
            """
            #### Supported Dataset Format
            Your raw CSV file should contain daily historical stock records with columns corresponding to:
            - `date` *(e.g., `Date`, `Trading Date`, `Timestamp`)*
            - `open` *(e.g., `Open`, `Opening Price`)*
            - `high` *(e.g., `High`, `Highest Price`)*
            - `low` *(e.g., `Low`, `Lowest Price`)*
            - `close` *(e.g., `Close`, `Closing Price`)*
            - `volume` *(e.g., `Volume`, `Shares Traded`)*
            - `adj_close` *(Optional — e.g., `Adj Close`, `Adjusted Close`)*
            """
        )
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    val_info: Dict[str, Any] = st.session_state["validation_info"]

    st.markdown("### Uploaded File Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Filename", str(val_info["filename"]), "Raw CSV File")
    with c2:
        render_kpi_card(
            "File Size",
            f"{val_info['file_size_kb']:,.2f} KB",
            f"{val_info['file_size_bytes']:,} bytes",
        )
    with c3:
        render_kpi_card(
            "Dataset Dimensions",
            f"{val_info['row_count']:,} × {val_info['col_count']}",
            f"{val_info['row_count']:,} rows, {val_info['col_count']} columns",
        )
    with c4:
        status_label = "Valid Stock CSV" if val_info["is_valid"] else "Invalid CSV"
        render_kpi_card(
            "Validation Status",
            status_label,
            "All required columns found" if val_info["is_valid"] else "Missing required columns/data",
        )

    st.markdown("#### First 5 Rows of Uploaded CSV")
    st.dataframe(raw_df.head(5), use_container_width=True)

    st.markdown("#### Column Validation & Standardization Check")
    if val_info["column_mapping"]:
        map_rows = [
            {"Original CSV Column": orig, "Mapped Standardized Column": std}
            for orig, std in val_info["column_mapping"].items()
        ]
        st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)

    for warn in val_info.get("warnings", []):
        st.info(f"ℹ️ {warn}")

    if not val_info["is_valid"]:
        for err in val_info["errors"]:
            st.error(f"❌ **Validation Error:** {err}")
        st.warning(
            "**How to fix this:** Please upload a stock-market CSV file containing valid columns for "
            "`date`, `open`, `high`, `low`, `close`, and `volume` with at least 30 historical trading days."
        )
        return

    st.success(
        f"✅ **Dataset '{val_info['filename']}' passed validation!** "
        "Click **Analyze Dataset** below to run automatic data analysis, preprocessing, and feature engineering."
    )

    bcol1, bcol2 = st.columns([1.4, 2.6])
    with bcol1:
        analyze_clicked = st.button(
            "🔍 Analyze Dataset",
            type="primary",
            use_container_width=True,
        )
    if analyze_clicked:
        with st.spinner("Analyzing dataset, cleaning records, and engineering features..."):
            try:
                bundle = cached_analyze_and_preprocess(
                    val_info["dataset_hash"], raw_df
                )
                st.session_state["raw_analysis"] = bundle["raw_analysis"]
                st.session_state["cleaned_df"] = bundle["cleaned_df"]
                st.session_state["prep_summary"] = bundle["prep_summary"]
                st.session_state["featured_df"] = bundle["featured_df"]
                st.session_state["dataset_analyzed"] = True
            except Exception as exc:
                st.error(f"❌ **Preprocessing Failed:** {exc}")
                return

        st.success(
            "🎉 **Dataset successfully analyzed and preprocessed!** "
            "Use the sidebar or click below to inspect **2. Dataset Analysis**."
        )
        if st.button("➡️ Proceed to Page 2: Dataset Analysis"):
            st.session_state["nav_page"] = PAGES[1]
            st.rerun()

    elif st.session_state["dataset_analyzed"]:
        st.info(
            "✅ This dataset has already been analyzed and preprocessed in the current session. "
            "Navigate through the steps in the sidebar or proceed to **2. Dataset Analysis**."
        )


# ---------------------------------------------------------------------------
# PAGE 2: DATASET ANALYSIS
# ---------------------------------------------------------------------------
def page_dataset_analysis() -> None:
    st.title("🗂️ Step 2: Dataset Analysis")
    if not require_analyzed_dataset():
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
    analysis: Dict[str, Any] = st.session_state["raw_analysis"]
    filename = st.session_state["uploaded_filename"]

    st.write(
        f"Complete diagnostic summary of the uploaded raw dataset (**`{filename}`**). "
        "The original uploaded data is preserved untouched in memory."
    )

    st.markdown("### 1. Key Dataset Metrics")
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1:
        render_kpi_card("Total Rows", f"{analysis['total_rows']:,}", "Raw uploaded rows")
    with r1c2:
        render_kpi_card("Total Columns", f"{analysis['total_columns']}", "Raw uploaded columns")
    with r1c3:
        render_kpi_card("Earliest Date", str(analysis["earliest_date"]), "Start of historical period")
    with r1c4:
        render_kpi_card("Latest Date", str(analysis["latest_date"]), "End of historical period")
    with r1c5:
        render_kpi_card(
            "Valid Closing Prices",
            f"{analysis['valid_closing_price']:,}" if "valid_closing_price" in analysis else f"{analysis['valid_closing_prices']:,}",
            "Positive numeric close rows",
        )

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    with r2c1:
        render_kpi_card("Missing Values", f"{analysis['missing_values_total']:,}", "Total null cells in raw CSV")
    with r2c2:
        render_kpi_card("Duplicate Records", f"{analysis['duplicate_rows']:,}", "Duplicate rows / dates")
    with r2c3:
        render_kpi_card("Average Closing Price", f"{analysis['average_closing_price']:,.2f}", "Mean historical close")
    with r2c4:
        render_kpi_card("Minimum Closing Price", f"{analysis['minimum_closing_price']:,.2f}", "Lowest historical close")
    with r2c5:
        render_kpi_card("Maximum Closing Price", f"{analysis['maximum_closing_price']:,.2f}", "Highest historical close")

    st.markdown("### 2. Inspect Original Raw Dataset vs. Cleaned Dataset")
    tab_raw, tab_clean = st.tabs(
        [
            f"📄 Original Uploaded Raw Dataset ({len(raw_df):,} rows)",
            f"✨ Cleaned Standardized Dataset ({len(cleaned_df):,} rows)",
        ]
    )
    with tab_raw:
        st.caption("Original uploaded CSV data (never overwritten):")
        st.dataframe(raw_df.head(50), use_container_width=True, height=320)
    with tab_clean:
        st.caption("Cleaned and chronologically sorted dataset:")
        st.dataframe(cleaned_df.head(50), use_container_width=True, height=320)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 3. Column Names, Standardized Mapping & Missing-Value Table")
        st.dataframe(analysis["column_info_df"], use_container_width=True, hide_index=True)
    with col_right:
        st.markdown("### 4. Duplicate & Date Range Summary")
        dup_df = pd.DataFrame(
            [
                {"Check": "Dataset Date Range", "Result": f"{analysis['earliest_date']} to {analysis['latest_date']}"},
                {"Check": "Exact Duplicate Rows in Raw CSV", "Result": f"{analysis['exact_duplicate_rows']:,}"},
                {"Check": "Duplicate Trading Dates", "Result": f"{analysis['duplicate_dates']:,}"},
                {"Check": "Total Missing Values Across All Columns", "Result": f"{analysis['missing_values_total']:,}"},
                {"Check": "Valid Positive Closing Price Rows", "Result": f"{analysis['valid_closing_prices']:,}"},
            ]
        )
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.markdown("### 5. Statistical Summary of Numeric Columns")
    st.dataframe(
        analysis["stats_summary_df"].style.format("{:,.4f}"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# PAGE 3: AUTOMATIC DATA PREPROCESSING
# ---------------------------------------------------------------------------
def page_preprocessing() -> None:
    st.title("🧹 Step 3: Automatic Data Preprocessing")
    if not require_analyzed_dataset():
        return

    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
    summary: Dict[str, Any] = st.session_state["prep_summary"]

    st.write(
        "The application automatically standardized column names, parsed and normalized dates, "
        "sorted records chronologically, removed duplicates and missing values, and verified "
        "that all OHLC prices are positive and logically consistent."
    )

    st.markdown("### 1. Preprocessing Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Original Row Count", f"{summary['original_rows']:,}", "Raw uploaded records")
    with c2:
        render_kpi_card("Cleaned Row Count", f"{summary['cleaned_rows']:,}", "Valid chronological records")
    with c3:
        render_kpi_card("Rows Removed", f"{summary['rows_removed_cleaning']:,}", "Total filtered out in cleaning")
    with c4:
        render_kpi_card(
            "Missing Values Handled",
            f"{summary['missing_values_total']:,}",
            f"{summary['missing_numeric_rows_removed']:,} incomplete rows dropped",
        )
    with c5:
        render_kpi_card(
            "Duplicates Removed",
            f"{summary['duplicate_rows_removed']:,}",
            "Duplicate rows/dates dropped",
        )

    st.markdown("### 2. Detailed Validation & Cleaning Checks Performed")
    checks_df = pd.DataFrame(
        [
            {
                "Preprocessing Step": "1. Column Name Normalization",
                "Action Taken": f"Mapped headers to standard columns: {', '.join(cleaned_df.columns)}",
                "Status": "✅ Passed",
            },
            {
                "Preprocessing Step": "2. Date Parsing & Timezone Normalization",
                "Action Taken": f"Converted dates to datetime and removed {summary['invalid_date_rows_removed']} invalid date rows",
                "Status": "✅ Passed",
            },
            {
                "Preprocessing Step": "3. Chronological Ordering",
                "Action Taken": f"Sorted ascending from {summary['earliest_date']} to {summary['latest_date']}",
                "Status": "✅ Passed",
            },
            {
                "Preprocessing Step": "4. Deduplication",
                "Action Taken": f"Removed {summary['duplicate_rows_removed']} duplicate records/trading dates",
                "Status": "✅ Passed",
            },
            {
                "Preprocessing Step": "5. Missing Value Policy (No Data Fabrication)",
                "Action Taken": f"Dropped {summary['missing_numeric_rows_removed']} rows with missing numeric values without inventing prices",
                "Status": "✅ Passed",
            },
            {
                "Preprocessing Step": "6. Positive Price & OHLC Consistency Check",
                "Action Taken": f"Removed {summary['invalid_price_or_volume_rows_removed']} rows with non-positive prices, negative volume, or high < low",
                "Status": "✅ Passed",
            },
        ]
    )
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

    st.markdown("### 3. Cleaned Dataset Preview & Download")
    st.dataframe(cleaned_df.head(25), use_container_width=True, height=340)

    clean_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned Dataset (CSV)",
        data=clean_csv_bytes,
        file_name=f"cleaned_{st.session_state['uploaded_filename']}",
        mime="text/csv",
        type="primary",
    )


# ---------------------------------------------------------------------------
# PAGE 4: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
def page_eda() -> None:
    st.title("📊 Step 4: Exploratory Data Analysis (EDA)")
    if not require_analyzed_dataset():
        return

    st.session_state["eda_explored"] = True
    featured_df: pd.DataFrame = st.session_state["featured_df"]
    eval_result: Optional[Dict[str, Any]] = st.session_state["eval_result"]

    st.write(
        "Explore interactive charts and downloadable static graphs generated directly from your "
        "uploaded dataset. Use the date filter below to zoom into any historical sub-period."
    )

    min_d = featured_df["date"].min().date()
    max_d = featured_df["date"].max().date()
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        start_d = st.date_input("Filter Start Date", value=min_d, min_value=min_d, max_value=max_d)
    with fcol2:
        end_d = st.date_input("Filter End Date", value=max_d, min_value=min_d, max_value=max_d)

    if start_d > end_d:
        st.error("Filter Start Date cannot be after Filter End Date.")
        return

    filtered_df = featured_df[
        (featured_df["date"].dt.date >= start_d) & (featured_df["date"].dt.date <= end_d)
    ].copy()

    if len(filtered_df) < 5:
        st.warning("Please select a date range containing at least 5 trading days.")
        return

    st.caption(
        f"Displaying **{len(filtered_df):,}** trading days from **{start_d}** to **{end_d}**."
    )

    # Generate downloadable PNG bytes for all graphs
    preds_df = eval_result["predictions_df"] if eval_result is not None else None
    png_map = generate_eda_png_bytes(filtered_df, predictions_df=preds_df)

    # Graph 1: Historical Closing Price Trend
    st.markdown("### Graph 1: Historical Closing Price Trend")
    fig1 = px.line(
        filtered_df,
        x="date",
        y="close",
        labels={"date": "Trading Date", "close": "Closing Price"},
        title="Historical Closing Price Trend",
        color_discrete_sequence=["#1e3a8a"],
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["closing_price_trend"]}</div>',
        unsafe_allow_html=True,
    )
    if "1_closing_price_trend.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 1 (PNG)",
            data=png_map["1_closing_price_trend.png"],
            file_name="1_closing_price_trend.png",
            mime="image/png",
        )

    # Graph 2: Trading Volume Over Time
    st.markdown("### Graph 2: Trading Volume Over Time")
    vol_df = filtered_df[["date", "volume"]].copy()
    vol_df["volume_m"] = vol_df["volume"] / 1e6
    vol_df["ma_30d"] = vol_df["volume_m"].rolling(30, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=vol_df["date"],
            y=vol_df["volume_m"],
            mode="lines",
            name="Daily Volume (M)",
            line=dict(color="#94a3b8", width=1),
            opacity=0.5,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=vol_df["date"],
            y=vol_df["ma_30d"],
            mode="lines",
            name="30-Day Moving Average (M)",
            line=dict(color="#d35400", width=2),
        )
    )
    fig2.update_layout(
        title="Trading Volume Over Time (with 30-Day Moving Average)",
        xaxis_title="Trading Date",
        yaxis_title="Volume (Million Shares)",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["trading_volume_trend"]}</div>',
        unsafe_allow_html=True,
    )
    if "2_trading_volume_trend.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 2 (PNG)",
            data=png_map["2_trading_volume_trend.png"],
            file_name="2_trading_volume_trend.png",
            mime="image/png",
        )

    # Graph 3: Daily Stock Return Distribution
    st.markdown("### Graph 3: Daily Stock Return Distribution")
    fig3 = px.histogram(
        filtered_df,
        x=filtered_df["daily_return"] * 100.0,
        nbins=75,
        labels={"x": "Daily Return (%)"},
        title="Daily Stock Return Distribution (%)",
        color_discrete_sequence=["#0284c7"],
    )
    fig3.add_vline(x=0.0, line_dash="dash", line_color="#0f172a")
    fig3.update_layout(height=390)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["daily_return_distribution"]}</div>',
        unsafe_allow_html=True,
    )
    if "3_daily_return_distribution.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 3 (PNG)",
            data=png_map["3_daily_return_distribution.png"],
            file_name="3_daily_return_distribution.png",
            mime="image/png",
        )

    # Graph 4: Correlation Heatmap
    st.markdown("### Graph 4: Correlation Heatmap")
    corr_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    corr_matrix = filtered_df[corr_cols].corr().round(3)
    fig4 = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Heatmap of Features and Next-Day Target Return",
    )
    fig4.update_layout(height=520)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["correlation_heatmap"]}</div>',
        unsafe_allow_html=True,
    )
    if "4_correlation_heatmap.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 4 (PNG)",
            data=png_map["4_correlation_heatmap.png"],
            file_name="4_correlation_heatmap.png",
            mime="image/png",
        )

    # Graph 5: Daily Return Boxplot
    st.markdown("### Graph 5: Daily Return Boxplot")
    box_long = pd.DataFrame(
        {
            "Daily Return (%)": filtered_df["daily_return"] * 100.0,
            "Intraday Return (%)": filtered_df["intraday_return"] * 100.0,
            "Next-Day Target Return (%)": filtered_df[TARGET_COLUMN] * 100.0,
        }
    ).melt(var_name="Return Metric", value_name="Return (%)")
    fig5 = px.box(
        box_long,
        x="Return (%)",
        y="Return Metric",
        color="Return Metric",
        title="Daily Return Boxplot (%)",
        color_discrete_sequence=["#2563eb", "#7c3aed", "#059669"],
    )
    fig5.update_layout(height=370, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["daily_return_boxplot"]}</div>',
        unsafe_allow_html=True,
    )
    if "5_daily_return_boxplot.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 5 (PNG)",
            data=png_map["5_daily_return_boxplot.png"],
            file_name="5_daily_return_boxplot.png",
            mime="image/png",
        )

    # Graph 6: Closing Price Distribution
    st.markdown("### Graph 6: Closing Price Distribution")
    fig6 = px.histogram(
        filtered_df,
        x="close",
        nbins=60,
        labels={"close": "Closing Price"},
        title="Closing Price Distribution",
        color_discrete_sequence=["#0d9488"],
    )
    fig6.update_layout(height=390)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(
        f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["closing_price_distribution"]}</div>',
        unsafe_allow_html=True,
    )
    if "6_closing_price_distribution.png" in png_map:
        st.download_button(
            "⬇️ Download Graph 6 (PNG)",
            data=png_map["6_closing_price_distribution.png"],
            file_name="6_closing_price_distribution.png",
            mime="image/png",
        )

    # Graph 7: Actual Versus Predicted Returns
    st.markdown("### Graph 7: Actual Versus Predicted Returns")
    if eval_result is None:
        st.info(
            "ℹ️ **Graph 7 (Actual vs. Predicted Returns)** will appear here automatically once you click "
            "**Train Both Models** on **Page 6: Machine Learning Training**."
        )
    else:
        sub_pred = eval_result["predictions_df"].tail(120).copy()
        fig7 = go.Figure()
        fig7.add_trace(
            go.Scatter(
                x=sub_pred["date"],
                y=sub_pred["actual_return"] * 100.0,
                mode="lines",
                name="Actual Next-Day Return (%)",
                line=dict(color="#64748b", width=1.2),
                opacity=0.65,
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=sub_pred["date"],
                y=sub_pred["ridge_predicted_return"] * 100.0,
                mode="lines",
                name="Ridge Predicted Return (%)",
                line=dict(color="#2563eb", width=2),
            )
        )
        fig7.add_trace(
            go.Scatter(
                x=sub_pred["date"],
                y=sub_pred["lasso_predicted_return"] * 100.0,
                mode="lines",
                name="Lasso Predicted Return (%)",
                line=dict(color="#dc2626", width=2, dash="dash"),
            )
        )
        fig7.add_hline(y=0.0, line_dash="dot", line_color="#0f172a")
        fig7.update_layout(
            title="Actual vs. Predicted Next-Day Returns (Recent 120 Unseen Test Days)",
            xaxis_title="Trading Date",
            yaxis_title="Next-Day Return (%)",
            height=410,
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown(
            f'<div class="interpretation-box"><strong>Interpretation:</strong> {EDA_INTERPRETATIONS["actual_vs_predicted"]}</div>',
            unsafe_allow_html=True,
        )
        if "7_actual_vs_predicted_returns.png" in png_map:
            st.download_button(
                "⬇️ Download Graph 7 (PNG)",
                data=png_map["7_actual_vs_predicted_returns.png"],
                file_name="7_actual_vs_predicted_returns.png",
                mime="image/png",
            )


# ---------------------------------------------------------------------------
# PAGE 5: FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def page_feature_engineering() -> None:
    st.title("🛠️ Step 5: Feature Engineering & Target Formulation")
    if not require_analyzed_dataset():
        return

    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
    featured_df: pd.DataFrame = st.session_state["featured_df"]

    st.write(
        "This project predicts **next-trading-day stock returns** rather than raw stock prices. "
        "All input features are calculated strictly from historical information available at the "
        "end of the current trading day (day $t$) to prevent any future data leakage."
    )

    st.markdown("### 1. Target Variable Formula (`target_return`)")
    st.latex(r"\text{target\_return}_t = \frac{\text{close}_{t+1}}{\text{close}_t} - 1")
    st.code(
        'df["target_return"] = df["close"].shift(-1) / df["close"] - 1',
        language="python",
    )
    st.info(
        f"✅ **Leakage Prevention Check:**\n"
        f"- Cleaned observations before feature engineering: **{len(cleaned_df):,} rows**\n"
        f"- Initial warmup rows removed (for 5-day historical return): **5 rows**\n"
        f"- Final row removed (because its next-day closing price is unavailable): **1 row**\n"
        f"- Final modeling dataset ready for training/testing: **{len(featured_df):,} rows** "
        f"(`{featured_df['date'].min().strftime('%Y-%m-%d')}` to `{featured_df['date'].max().strftime('%Y-%m-%d')}`)\n"
        f"- `target_return` and future closing prices are **never** included in `FEATURE_COLUMNS`."
    )

    st.markdown("### 2. Engineered Input Features Explained in Simple English")
    feat_table = pd.DataFrame(
        [
            {
                "Feature Order": idx + 1,
                "Feature Name": col,
                "Available at End of Day t?": "Yes (No Future Info)",
                "Simple-English Explanation": FEATURE_DESCRIPTIONS[col],
            }
            for idx, col in enumerate(FEATURE_COLUMNS)
        ]
    )
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.markdown("### 3. Preview of Engineered Features (`X`) and Next-Day Target (`y`)")
    st.dataframe(
        featured_df[["date"] + FEATURE_COLUMNS + [TARGET_COLUMN]].head(20),
        use_container_width=True,
        height=350,
    )


# ---------------------------------------------------------------------------
# PAGE 6: MACHINE LEARNING TRAINING
# ---------------------------------------------------------------------------
def page_ml_training() -> None:
    st.title("⚙️ Step 6: Machine Learning Model Training")
    if not require_analyzed_dataset():
        return

    featured_df: pd.DataFrame = st.session_state["featured_df"]
    train_df, test_df = chronological_train_test_split(featured_df, train_ratio=0.80)

    st.write(
        "Train **Ridge Regression ($L_2$ Regularization)** and **Lasso Regression ($L_1$ Regularization)** "
        "on your uploaded dataset using a strict chronological split and leakage-free `scikit-learn` Pipelines."
    )

    st.markdown("### 1. Chronological Split & Pipeline Architecture")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Total Modeling Rows",
            f"{len(featured_df):,}",
            f"{featured_df['date'].min().strftime('%Y-%m-%d')} to {featured_df['date'].max().strftime('%Y-%m-%d')}",
        )
    with c2:
        render_kpi_card(
            "Training Split (First 80%)",
            f"{len(train_df):,} rows",
            f"{train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}",
        )
    with c3:
        render_kpi_card(
            "Testing Split (Last 20%)",
            f"{len(test_df):,} rows",
            f"{test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}",
        )
    with c4:
        render_kpi_card(
            "Pipeline Structure",
            "StandardScaler + Model",
            "Scaler fitted on Training Set ONLY",
        )

    st.markdown(
        f"""
        - **No Random Shuffling:** Earlier observations (`{train_df['date'].min().strftime('%Y-%m-%d')}` to `{train_df['date'].max().strftime('%Y-%m-%d')}`) are used for training, and later observations (`{test_df['date'].min().strftime('%Y-%m-%d')}` to `{test_df['date'].max().strftime('%Y-%m-%d')}`) are held out for testing.
        - **Hyperparameter Tuning (`TimeSeriesSplit` CV on Training Set Only):**
          - **Ridge $\\alpha$ Grid:** `{RIDGE_ALPHA_GRID}`
          - **Lasso $\\alpha$ Grid:** `{LASSO_ALPHA_GRID}`
        """
    )

    st.markdown("### 2. Train Models on Uploaded Dataset")
    col_train_btn, col_save_chk = st.columns([1.4, 2.6])
    with col_save_chk:
        save_to_disk = st.checkbox(
            "Also save trained model pipelines to `models/ridge_model.joblib` and `models/lasso_model.joblib`",
            value=True,
        )
    with col_train_btn:
        train_clicked = st.button(
            "🚀 Train Both Models",
            type="primary",
            use_container_width=True,
        )

    if train_clicked:
        progress = st.progress(0, text="Initializing chronological split and pipelines...")
        try:
            progress.progress(25, text="Training Ridge & Lasso Regression with TimeSeriesSplit CV...")
            training_bundle = train_and_tune_models(
                df=featured_df,
                train_ratio=0.80,
                n_splits=5,
                save_models=save_to_disk,
            )
            progress.progress(75, text="Evaluating trained models on unseen chronological test set...")
            eval_result = evaluate_trained_models(
                ridge_pipeline=training_bundle["ridge_pipeline"],
                lasso_pipeline=training_bundle["lasso_pipeline"],
                test_df=training_bundle["test_df"],
                save_outputs=save_to_disk,
            )
            progress.progress(100, text="Training and evaluation complete!")

            st.session_state["training_bundle"] = training_bundle
            st.session_state["eval_result"] = eval_result
            st.session_state["models_trained"] = True
        except Exception as exc:
            st.error(f"❌ **Model Training Error:** {exc}")
            return

        st.success(
            "✅ **Both Ridge Regression and Lasso Regression have been trained and evaluated successfully!**"
        )

    if st.session_state["models_trained"] and st.session_state["training_bundle"] is not None:
        meta = st.session_state["training_bundle"]["metadata"]
        st.markdown("### 3. Trained Model Summary & Learned Coefficients")
        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi_card(
                "Ridge Regression (L2)",
                f"Best α = {meta['ridge']['best_alpha']}",
                "All 10 features retained with L2 shrinkage",
            )
        with m2:
            render_kpi_card(
                "Lasso Regression (L1)",
                f"Best α = {meta['lasso']['best_alpha']}",
                f"Kept {meta['lasso']['non_zero_features']} / {len(FEATURE_COLUMNS)} features non-zero",
            )
        with m3:
            render_kpi_card(
                "Features Zeroed by Lasso",
                f"{meta['lasso']['zeroed_features']} features",
                "Automatic L1 feature selection",
            )

        coef_df = pd.DataFrame(
            {
                "Feature Name": FEATURE_COLUMNS,
                "Ridge Standardized Coefficient": [
                    meta["ridge"]["coefficients"][f] for f in FEATURE_COLUMNS
                ],
                "Lasso Standardized Coefficient": [
                    meta["lasso"]["coefficients"][f] for f in FEATURE_COLUMNS
                ],
            }
        )
        st.dataframe(
            coef_df.style.format(
                {
                    "Ridge Standardized Coefficient": "{:+.6f}",
                    "Lasso Standardized Coefficient": "{:+.6f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("➡️ Proceed to Step 7: Model Evaluation"):
            st.session_state["nav_page"] = PAGES[6]
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE 7: MODEL EVALUATION
# ---------------------------------------------------------------------------
def page_model_evaluation() -> None:
    st.title("📏 Step 7: Model Evaluation & Comparison")
    if not require_trained_models():
        return

    eval_result: Dict[str, Any] = st.session_state["eval_result"]
    metrics_df: pd.DataFrame = eval_result["metrics_df"]
    predictions_df: pd.DataFrame = eval_result["predictions_df"]
    best_model: str = eval_result["best_model_name"]

    if "Ridge" in best_model:
        st.success("🏆 **Conclusion:** Ridge Regression performed better on the selected test metrics.")
    else:
        st.success("🏆 **Conclusion:** Lasso Regression performed better on the selected test metrics.")

    st.markdown("### 1. Chronological Test-Set Evaluation Table")
    comparison_table = metrics_df[["Model", "MAE", "MSE", "RMSE", "R2_Score"]].rename(
        columns={"R2_Score": "R² Score"}
    )
    st.dataframe(
        comparison_table.style.format(
            {
                "MAE": "{:.6f}",
                "MSE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R² Score": "{:.6f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f'<div class="interpretation-box"><strong>Honest Performance Analysis:</strong> {eval_result["explanation"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 2. Bar Chart Comparing Ridge and Lasso")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        err_long = metrics_df.melt(
            id_vars="Model",
            value_vars=["MAE", "RMSE"],
            var_name="Metric",
            value_name="Error Value",
        )
        fig_bar = px.bar(
            err_long,
            x="Model",
            y="Error Value",
            color="Metric",
            barmode="group",
            title="MAE & RMSE Comparison (Lower is Better)",
            color_discrete_sequence=["#0284c7", "#1e3a8a"],
            text_auto=".5f",
        )
        fig_bar.update_layout(height=370)
        st.plotly_chart(fig_bar, use_container_width=True)

    with bcol2:
        fig_r2 = px.bar(
            metrics_df,
            x="Model",
            y="R2_Score",
            color="Model",
            title="R² Score Comparison (Higher is Better)",
            color_discrete_sequence=["#2563eb", "#dc2626"],
            text_auto=".5f",
        )
        fig_r2.add_hline(y=0.0, line_dash="dash", line_color="#334155")
        fig_r2.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("### 3. Actual Versus Predicted Returns on the Unseen Test Set")
    max_days = int(len(predictions_df))
    default_days = min(150, max_days)
    min_days = min(20, max_days)
    if max_days > min_days:
        n_days = st.slider(
            "Select number of test trading days to display:",
            min_value=min_days,
            max_value=max_days,
            value=default_days,
        )
    else:
        n_days = max_days

    sub_pred = predictions_df.tail(n_days).copy()
    sub_pred["date"] = pd.to_datetime(sub_pred["date"])
    fig_avp = go.Figure()
    fig_avp.add_trace(
        go.Scatter(
            x=sub_pred["date"],
            y=sub_pred["actual_return"] * 100.0,
            mode="lines",
            name="Actual Return (%)",
            line=dict(color="#64748b", width=1.2),
            opacity=0.65,
        )
    )
    fig_avp.add_trace(
        go.Scatter(
            x=sub_pred["date"],
            y=sub_pred["ridge_predicted_return"] * 100.0,
            mode="lines",
            name="Ridge Predicted Return (%)",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig_avp.add_trace(
        go.Scatter(
            x=sub_pred["date"],
            y=sub_pred["lasso_predicted_return"] * 100.0,
            mode="lines",
            name="Lasso Predicted Return (%)",
            line=dict(color="#dc2626", width=2, dash="dash"),
        )
    )
    fig_avp.add_hline(y=0.0, line_dash="dot", line_color="#0f172a")
    fig_avp.update_layout(
        title=f"Actual vs. Predicted Next-Trading-Day Returns (Last {n_days} Test Days)",
        xaxis_title="Trading Date",
        yaxis_title="Return (%)",
        height=420,
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("### 4. Evaluation Metrics Explained in Simple English")
    metric_guide = pd.DataFrame(
        [
            {
                "Metric": "MAE (Mean Absolute Error)",
                "Simple-English Meaning": "Average absolute difference between actual and predicted returns.",
                "How to Interpret": "Lower MAE indicates smaller average prediction errors.",
            },
            {
                "Metric": "MSE (Mean Squared Error)",
                "Simple-English Meaning": "Average squared prediction error.",
                "How to Interpret": "Penalizes large outlier errors heavily; lower is better.",
            },
            {
                "Metric": "RMSE (Root Mean Squared Error)",
                "Simple-English Meaning": "Square root of MSE, expressed in the target's units (daily return).",
                "How to Interpret": "Lower RMSE indicates tighter predictions around actual returns.",
            },
            {
                "Metric": "R² Score",
                "Simple-English Meaning": "Measures how well predictions explain variation in the test targets relative to a mean baseline.",
                "How to Interpret": "Higher R² indicates a better fit, though R² can be near zero or negative on noisy unseen stock returns.",
            },
        ]
    )
    st.dataframe(metric_guide, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 8: STOCK RETURN PREDICTION
# ---------------------------------------------------------------------------
def page_prediction() -> None:
    st.title("🔮 Step 8: Next-Trading-Day Stock Return Prediction")
    if not require_trained_models():
        return

    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
    training_bundle: Dict[str, Any] = st.session_state["training_bundle"]

    defaults = get_default_prediction_inputs(cleaned_df)

    st.markdown("### 1. Choose Regression Model & Prediction Mode")
    col_m1, col_m2 = st.columns([1.2, 1.8])
    with col_m1:
        selected_model = st.selectbox(
            "Select Trained Model:",
            options=["Ridge Regression", "Lasso Regression"],
            index=0,
        )
    with col_m2:
        pred_mode = st.radio(
            "Select Prediction Mode:",
            options=[
                "MODE A: Predict Using Latest Historical Data",
                "MODE B: Manual Input",
            ],
            horizontal=False,
        )

    pipeline = (
        training_bundle["ridge_pipeline"]
        if "Ridge" in selected_model
        else training_bundle["lasso_pipeline"]
    )

    if pred_mode.startswith("MODE A"):
        st.info(
            f"📅 **Latest Observation Date in Uploaded Dataset:** `{defaults['reference_date']}` "
            f"(Historical Closing Price: **{defaults['close']:,.2f}**).\n\n"
            "*Note: This date is the final trading date present in your uploaded CSV dataset, not necessarily today's live calendar date.*"
        )
        active_inputs = defaults
        reference_date_label = f"Next Trading Day After {defaults['reference_date']}"
        run_pred = True
    else:
        st.markdown(
            "#### Enter Current Trading Day OHLCV & Required Recent Closing Prices"
        )
        st.caption(
            "To compute `daily_return`, `lag_return_1`, and `return_5d` without inventing missing lagged values, "
            "provide today's OHLCV along with the recent historical closing prices below (pre-filled from your dataset's latest values)."
        )
        with st.form("manual_prediction_form"):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                open_val = st.number_input(
                    "Open Price (> 0)",
                    min_value=0.0001,
                    value=float(defaults["open"]),
                    step=1.0,
                    format="%.2f",
                )
                high_val = st.number_input(
                    "High Price (>= Open, Low, Close)",
                    min_value=0.0001,
                    value=float(defaults["high"]),
                    step=1.0,
                    format="%.2f",
                )
            with mc2:
                low_val = st.number_input(
                    "Low Price (<= Open, High, Close)",
                    min_value=0.0001,
                    value=float(defaults["low"]),
                    step=1.0,
                    format="%.2f",
                )
                close_val = st.number_input(
                    "Close Price (> 0)",
                    min_value=0.0001,
                    value=float(defaults["close"]),
                    step=1.0,
                    format="%.2f",
                )
            with mc3:
                volume_val = st.number_input(
                    "Trading Volume (>= 0)",
                    min_value=0,
                    value=int(defaults["volume"]),
                    step=10000,
                )

            st.markdown("##### Recent Historical Closing Prices (For Lagged Return Features)")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                prev_close_val = st.number_input(
                    "Previous Trading Day Close (t-1)",
                    min_value=0.0001,
                    value=float(defaults["prev_close"]),
                    step=1.0,
                    format="%.2f",
                )
            with lc2:
                prev_2_close_val = st.number_input(
                    "2 Trading Days Ago Close (t-2)",
                    min_value=0.0001,
                    value=float(defaults["prev_2_close"]),
                    step=1.0,
                    format="%.2f",
                )
            with lc3:
                close_5d_ago_val = st.number_input(
                    "5 Trading Days Ago Close (t-5)",
                    min_value=0.0001,
                    value=float(defaults["close_5d_ago"]),
                    step=1.0,
                    format="%.2f",
                )

            run_pred = st.form_submit_button(
                "🚀 Generate Manual Prediction", type="primary", use_container_width=True
            )

        active_inputs = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
            "prev_close": prev_close_val,
            "prev_2_close": prev_2_close_val,
            "close_5d_ago": close_5d_ago_val,
        }
        reference_date_label = "Next Trading Session (1 Trading Day Ahead)"

    if run_pred or st.session_state["latest_prediction"] is not None:
        try:
            pred_res = predict_next_day_return(
                active_inputs, model_choice=selected_model, pipeline=pipeline
            )
            st.session_state["latest_prediction"] = pred_res
            st.session_state["prediction_completed"] = True
        except ValueError as err:
            st.error(f"❌ **Input Validation Error:** {err}")
            return

        st.markdown("### 2. Prediction Result")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi_card(
                "Selected Model",
                pred_res["selected_model"],
                "StandardScaler + Trained Estimator",
            )
        with p2:
            render_kpi_card(
                "Predicted Next-Trading-Day Return",
                f"{pred_res['predicted_return_pct']:+.2f}%",
                f"Decimal Return: {pred_res['predicted_return']:+.6f}",
            )
        with p3:
            render_kpi_card(
                "Reference Closing Price",
                f"{pred_res['features']['close']:,.2f}",
                f"Implied Next Close: {pred_res['estimated_next_close']:,.2f}",
            )
        with p4:
            render_kpi_card(
                "Prediction Horizon",
                "1 Trading Day",
                reference_date_label,
            )

        st.markdown(
            f'<div class="interpretation-box"><strong>Interpretation:</strong><br>{pred_res["interpretation"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 3. Selected Input Features (Exact Training Order)")
        feat_vals = pred_res["features"]
        feat_df_display = pd.DataFrame(
            [
                {
                    "Feature Order": idx + 1,
                    "Feature Name": col,
                    "Input Value": (
                        f"{feat_vals[col]:,.0f}"
                        if col == "volume"
                        else (
                            f"{feat_vals[col]:+.6f} ({feat_vals[col] * 100:+.2f}%)"
                            if "return" in col
                            else f"{feat_vals[col]:,.4f}"
                        )
                    ),
                    "Description": FEATURE_DESCRIPTIONS[col],
                }
                for idx, col in enumerate(FEATURE_COLUMNS)
            ]
        )
        st.dataframe(feat_df_display, use_container_width=True, hide_index=True)

        st.markdown(
            f'<div class="disclaimer-box">⚠️ <strong>Disclaimer:</strong> {REQUIRED_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PAGE 9: DOWNLOAD REPORTS
# ---------------------------------------------------------------------------
def page_download_reports() -> None:
    st.title("📥 Step 9: Download Results & Reports")
    if not require_analyzed_dataset():
        return

    st.session_state["reports_visited"] = True
    cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
    featured_df: pd.DataFrame = st.session_state["featured_df"]
    prep_summary: Dict[str, Any] = st.session_state["prep_summary"]
    filename = st.session_state["uploaded_filename"]

    st.write(
        "Download the cleaned dataset, model evaluation metrics, unseen test-set predictions, "
        "generated EDA charts, and a complete project summary report built from your actual execution."
    )

    st.markdown("### 1. Download Datasets & Model Outputs (CSV)")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.download_button(
            label="⬇️ 1. Download Cleaned Dataset (CSV)",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{filename}",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state["models_trained"] and st.session_state["eval_result"] is not None:
        eval_result = st.session_state["eval_result"]
        training_meta = st.session_state["training_bundle"]["metadata"]

        with dcol2:
            st.download_button(
                label="⬇️ 2. Download Model Evaluation Metrics (CSV)",
                data=eval_result["metrics_df"].to_csv(index=False).encode("utf-8"),
                file_name="model_evaluation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol3:
            st.download_button(
                label="⬇️ 3. Download Test-Set Predictions (CSV)",
                data=eval_result["predictions_df"].to_csv(index=False).encode("utf-8"),
                file_name="test_set_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### 2. Download Comprehensive Summary Report")
        report_text = generate_summary_report(
            dataset_filename=filename,
            prep_summary=prep_summary,
            training_metadata=training_meta,
            eval_result=eval_result,
            latest_prediction=st.session_state["latest_prediction"],
            save_to_disk=True,
        )
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.download_button(
                label="📄 Download Summary Report (.md)",
                data=report_text.encode("utf-8"),
                file_name="stock_return_prediction_report.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
            )
        with rcol2:
            st.download_button(
                label="📝 Download Summary Report (.txt)",
                data=report_text.encode("utf-8"),
                file_name="stock_return_prediction_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with st.expander("🔍 Preview Generated Summary Report", expanded=True):
            st.markdown(report_text)
    else:
        st.info(
            "ℹ️ Train the models on **Page 6: Machine Learning Training** to unlock the "
            "Model Evaluation CSV, Test Predictions CSV, and full Model Comparison Report."
        )

    st.markdown("### 3. Download Generated Charts (PNG)")
    preds_df = (
        st.session_state["eval_result"]["predictions_df"]
        if st.session_state["eval_result"] is not None
        else None
    )
    png_map = generate_eda_png_bytes(featured_df, predictions_df=preds_df)
    gcols = st.columns(4)
    for idx, (chart_name, png_bytes) in enumerate(png_map.items()):
        with gcols[idx % 4]:
            st.download_button(
                label=f"🖼️ {chart_name}",
                data=png_bytes,
                file_name=chart_name,
                mime="image/png",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    st.sidebar.title("📈 Stock Return Predictor")
    st.sidebar.caption("Ridge ($L_2$) vs. Lasso ($L_1$) Regression")
    st.sidebar.markdown("---")

    selected_page = st.sidebar.radio(
        "Workflow Navigation",
        options=PAGES,
        index=PAGES.index(st.session_state["nav_page"])
        if st.session_state["nav_page"] in PAGES
        else 0,
    )
    st.session_state["nav_page"] = selected_page

    st.sidebar.markdown("---")
    if st.session_state["uploaded_filename"]:
        st.sidebar.success(f"**Active CSV:** `{st.session_state['uploaded_filename']}`")
    else:
        st.sidebar.info("**Active CSV:** None uploaded yet")

    st.sidebar.caption(
        "⚖️ **Educational Disclaimer:** For B.Tech Machine Learning academic demonstration only. "
        "Not financial advice."
    )

    render_progress_indicator()

    if selected_page == PAGES[0]:
        page_upload_dataset()
    elif selected_page == PAGES[1]:
        page_dataset_analysis()
    elif selected_page == PAGES[2]:
        page_preprocessing()
    elif selected_page == PAGES[3]:
        page_eda()
    elif selected_page == PAGES[4]:
        page_feature_engineering()
    elif selected_page == PAGES[5]:
        page_ml_training()
    elif selected_page == PAGES[6]:
        page_model_evaluation()
    elif selected_page == PAGES[7]:
        page_prediction()
    elif selected_page == PAGES[8]:
        page_download_reports()


if __name__ == "__main__":
    main()
