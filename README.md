# Stock Return Prediction Using Ridge & Lasso Regression

**Domain:** Finance and Machine Learning  
**Operating System:** macOS  
**IDE:** Visual Studio Code  
**Programming Language:** Python 3  
**Framework:** Streamlit Interactive Web Application  
**Machine Learning Models:** Ridge Regression ($L_2$ Regularization) vs. Lasso Regression ($L_1$ Regularization)

---

## 1. Project Title
**Stock Return Prediction Using Ridge & Lasso Regression**  
*Analyze historical stock market data and compare two regularized Machine Learning regression models.*

---

## 2. Abstract
Financial stock market data is noisy, collinear across daily price attributes (`open`, `high`, `low`, `close`), and non-stationary in nominal price levels over multi-year horizons. This project builds a complete, beginner-friendly, upload-driven Machine Learning web application in Python and Streamlit that accepts any compatible historical stock-market CSV dataset (including the bundled **Reliance Industries Ltd. (`RELIANCE.NS`)** dataset covering $7,324$ trading days from 1996 to 2025). The system validates the uploaded file, normalizes column names automatically, cleans invalid or duplicate observations without fabricating data, generates 7 exploratory visualizations, engineers 10 leakage-free features available at the close of trading day $t$, and trains **Ridge Regression ($L_2$)** and **Lasso Regression ($L_1$)** inside `scikit-learn` `Pipeline` objects (`StandardScaler` + Estimator) on a strictly chronological $80\% / 20\%$ train-test split. Users can compare both models using real test-set metrics (**MAE, MSE, RMSE, and $R^2$ Score**), predict the next trading day's return via **Mode A (Latest Historical Data)** or **Mode B (Manual Input)**, and download the cleaned dataset, evaluation tables, charts, and a complete summary report.

---

## 3. Introduction
Attempting to predict raw stock prices directly using unregularized linear regression and random train-test splits leads to severe look-ahead leakage and poor generalization when a stock moves into a new price range. In quantitative finance, models instead predict **next-trading-day percentage returns** ($\text{target\_return}_t = \text{close}_{t+1} / \text{close}_t - 1$), which remain stationary around $0\%$ over time. Regularized linear models—**Ridge Regression** and **Lasso Regression**—control variance in the presence of correlated OHLC features and prevent overfitting to historical market noise.

---

## 4. Problem Statement
1. **Arbitrary CSV Variations & Data Quality Issues:** Raw stock CSV files often use varying column headers (`Date`, `Open`, `High`, `Low`, `Close`, `Adjusted Close`, `Volume`) and may contain missing values, duplicate dates, or invalid rows.
2. **Multicollinearity & Non-Stationarity:** Same-day price columns (`open`, `high`, `low`, `close`) are heavily correlated, while nominal share prices drift across decades.
3. **Need for an End-to-End Educational System:** Students and examiners need a guided, transparent workflow that starts from uploading a raw CSV file and progresses step-by-step through validation, preprocessing, EDA, feature engineering, chronological training, evaluation, next-day return prediction, and report generation.

---

## 5. Objectives
1. Provide a prominent CSV file uploader that validates raw stock datasets and automatically maps common column name variations.
2. Perform automatic data preprocessing without overwriting the original uploaded dataset.
3. Generate 7 interactive and downloadable Exploratory Data Analysis (EDA) visualizations with date filtering and plain-English interpretations.
4. Engineer 10 leakage-free features from historical data up to day $t$ to predict the next trading day's return (day $t+1$).
5. Train **Ridge Regression** and **Lasso Regression** using chronological splitting ($80\%$ training, $20\%$ testing), `StandardScaler` pipelines fitted strictly on training data, and `TimeSeriesSplit` cross-validation.
6. Evaluate and compare both models honestly using **MAE, MSE, RMSE, and $R^2$ Score**.
7. Support next-trading-day return prediction using **Mode A (Latest Historical Data)** and **Mode B (Manual Input)**, and provide one-click downloads for all datasets, charts, and reports.

---

## 6. Dataset Format
The application accepts any raw `.csv` file containing daily historical stock records.
- **Required Standardized Columns:** `date`, `open`, `high`, `low`, `close`, `volume`
- **Optional Column:** `adj_close`
- **Recognized Column Variations:** `Date`, `Trading Date`, `Timestamp`, `Open`, `Open Price`, `High`, `Low`, `Close`, `Close Price`, `Adj Close`, `Adjusted Close`, `Volume`, `Traded Volume`, `Shares Traded`.
- **Bundled Project Dataset:** `dataset/RELIANCE.NS_1973-05-08_2025-03-01.csv` ($7,324$ trading days from `1996-01-01` to `2025-02-28`).

---

## 7. Software Requirements
- **Operating System:** macOS (also compatible with Linux and Windows)
- **IDE:** Visual Studio Code (VS Code)
- **Python Version:** Python 3.9 or newer
- **Required Libraries (`requirements.txt`):** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `joblib`, `pytest`

---

## 8. Installation Steps (macOS & VS Code)

Open the `Stock-Return-Prediction` folder in **VS Code**, open the integrated terminal, and run:

```bash
# 1. Activate the existing virtual environment
source .venv/bin/activate
# (Or if using `venv`: source venv/bin/activate)

# 2. Install required dependencies
python -m pip install -r requirements.txt
```

---

## 9. How to Launch the Website & Run Tests

### Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The website will open at `http://localhost:8501`.

### Run Automated Tests
```bash
python -m pytest -v
```

### Run Individual Pipeline Modules from CLI (Optional)
```bash
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python -m src.predict
```

---

## 10. Guided User Workflow (11 Steps Across 9 Pages)

```text
STEP 1: UPLOAD DATASET          (Page 1: Upload Dataset)
        |
        v
STEP 2: VALIDATE DATASET        (Page 1: Automatic Column & Data Validation)
        |
        v
STEP 3: DISPLAY DATASET INFO    (Page 2: Dataset Analysis)
        |
        v
STEP 4: AUTO PREPROCESSING      (Page 3: Automatic Data Preprocessing)
        |
        v
STEP 5: EXPLORATORY ANALYSIS    (Page 4: Exploratory Data Analysis — 7 Graphs)
        |
        v
STEP 6: FEATURE ENGINEERING     (Page 5: Feature Engineering & Target Formulation)
        |
        v
STEP 7: TRAIN RIDGE REGRESSION  (Page 6: Machine Learning Training)
        |
        v
STEP 8: TRAIN LASSO REGRESSION  (Page 6: Machine Learning Training)
        |
        v
STEP 9: EVALUATE & COMPARE      (Page 7: Model Evaluation)
        |
        v
STEP 10: PREDICT NEXT-DAY RET   (Page 8: Stock Return Prediction — Mode A & Mode B)
        |
        v
STEP 11: DOWNLOAD REPORTS       (Page 9: Download CSVs, PNG Charts & Summary Report)
```

A live 8-stage progress bar at the top of the interface (`1. Upload` → `2. Analyze` → `3. Preprocess` → `4. Explore` → `5. Train` → `6. Compare` → `7. Predict` → `8. Download`) marks each stage complete only after its operation succeeds.

---

## 11. Preprocessing Methodology
Implemented in `src/data_loader.py` and `src/preprocessing.py`:
1. **Column Normalization:** Lowercases and standardizes headers, mapping variations to `date`, `open`, `high`, `low`, `close`, `volume`, and optional `adj_close`.
2. **Datetime Conversion:** Converts `date` to timezone-naive `datetime64` and drops unparseable dates.
3. **Chronological Sorting & Deduplication:** Sorts observations from oldest to newest and removes duplicate rows or duplicate trading dates.
4. **Missing & Invalid Value Handling:** Converts price/volume columns to numeric types, drops rows with missing values (never fabricating stock prices), and removes rows with non-positive prices ($\le 0$), negative volumes ($< 0$), or inconsistent OHLC bounds ($\text{high} < \text{low}$).

---

## 12. Exploratory Data Analysis (EDA) Methodology
Implemented in `src/eda.py` and Page 4 of `app.py` with interactive date filtering and PNG downloads:
1. **Graph 1 — Historical Closing Price Trend:** Visualizes long-term price trajectory and non-stationarity.
2. **Graph 2 — Trading Volume Over Time:** Plots daily share volume alongside its 30-day moving average.
3. **Graph 3 — Daily Stock Return Distribution:** Histogram and kernel density curve centered near $0\%$.
4. **Graph 4 — Correlation Heatmap:** Displays linear Pearson correlations among all features and `target_return`.
5. **Graph 5 — Daily Return Boxplot:** Highlights median return, interquartile range (IQR), and outlier sessions.
6. **Graph 6 — Closing Price Distribution:** Shows the frequency of closing prices across historical price regimes.
7. **Graph 7 — Actual Versus Predicted Returns:** Compares unseen test-set returns against Ridge and Lasso predictions.

---

## 13. Feature Engineering
Implemented in `src/feature_engineering.py`:
- **Target Variable (`target_return`):**
  $$\text{target\_return}_t = \frac{\text{close}_{t+1}}{\text{close}_t} - 1$$
  The final observation is removed because its next-day closing price is unavailable.
- **10 Input Features (Available at Market Close of Day $t$):**
  1. `open`: Opening price on day $t$
  2. `high`: Highest price on day $t$
  3. `low`: Lowest price on day $t$
  4. `close`: Closing price on day $t$
  5. `volume`: Trading volume on day $t$
  6. `daily_return`: $(\text{close}_t / \text{close}_{t-1}) - 1$
  7. `intraday_return`: $(\text{close}_t / \text{open}_t) - 1$
  8. `daily_range`: $\text{high}_t - \text{low}_t$
  9. `lag_return_1`: Previous day's return $(\text{close}_{t-1} / \text{close}_{t-2}) - 1$
  10. `return_5d`: 5-day historical return $(\text{close}_t / \text{close}_{t-5}) - 1$

---

## 14. Ridge Regression Explanation
**Ridge Regression ($L_2$ Regularization)** minimizes:
$$J_{\text{Ridge}}(w) = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} w_j^2$$
By penalizing squared coefficients, Ridge shrinks correlated feature weights smoothly toward zero, stabilizing the model when OHLC prices are collinear while keeping all 10 features active.

---

## 15. Lasso Regression Explanation
**Lasso Regression ($L_1$ Regularization)** minimizes:
$$J_{\text{Lasso}}(w) = \frac{1}{2N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |w_j|$$
By penalizing absolute coefficients, Lasso performs automatic feature selection—driving redundant or non-stationary features (such as raw price levels `open`, `high`, `low`, `close`) to **exact zero ($0.0$)**.

---

## 16. Evaluation Metrics & Bundled Reliance Dataset Results
On the bundled Reliance Industries dataset ($5,854$ training rows: `1996-01-08` to `2019-03-27`; $1,464$ unseen test rows: `2019-03-28` to `2025-02-27`):

| Model | MAE | MSE | RMSE | $R^2$ Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression ($\alpha=10.0$)** | `0.012671` | `0.000346` | `0.018595` | `-0.024513` | `49.18%` |
| **Lasso Regression ($\alpha=0.0005$)** | **`0.012516`** | **`0.000340`** | **`0.018439`** | **`-0.007373`** | **`49.66%`** |

- **MAE:** Average absolute difference between actual and predicted returns.
- **MSE:** Average squared prediction error.
- **RMSE:** Square root of MSE, expressed in daily return units.
- **$R^2$ Score:** Variance explained relative to the test-set mean baseline.

---

## 17. Prediction Methodology
Implemented in `src/predict.py` and Page 8 of `app.py`:
- **Mode A (Predict Using Latest Historical Data):** Automatically extracts the most recent trading day ($t$) and necessary lagged closes ($t-1, t-2, t-5$) from the uploaded dataset to predict the return for the next trading session ($t+1$).
- **Mode B (Manual Input):** Allows users to input custom OHLCV and recent closing prices with strict validation (rejecting non-positive prices, negative volume, or $\text{high} < \text{low}$).
- Both modes construct the exact 10-feature DataFrame in training order (`FEATURE_COLUMNS`) and pass it through the selected model's fitted `Pipeline`.

---

## 18. Project Limitations
1. **Historical OHLCV Scope:** Models do not observe breaking news, earnings announcements, macroeconomic indicators, or order-book depth.
2. **Linearity Assumption:** Ridge and Lasso model only linear relationships between features and next-day returns.
3. **Market Noise:** Daily stock returns have a very low signal-to-noise ratio, so predictions should be treated strictly as educational estimates.

---

## 19. Future Improvements
1. Incorporate financial news sentiment scores (NLP) and market index returns (e.g., Nifty 50).
2. Add rolling walk-forward retraining windows.
3. Pair point return predictions with volatility forecasting (GARCH).

---

## 20. Conclusion
This project delivers an end-to-end, upload-driven Machine Learning web application that guides users from raw CSV validation through automatic preprocessing, EDA, leakage-free feature engineering, chronological Ridge and Lasso Regression training, honest test-set evaluation, next-trading-day return prediction, and full report generation.
