# Stock Return Prediction Using Ridge & Lasso Regression — Summary Report
**Generated At:** 2026-10-13 12:37:14

---

## 1. Dataset Information & Date Range
- **Uploaded Dataset Filename:** `RELIANCE.NS_1973-05-08_2025-03-01.csv`
- **Raw Dataset Observations:** 7,324 rows x 7 columns
- **Cleaned Valid Observations:** 7,324 rows
- **Historical Date Range (Cleaned):** `1996-01-01` to `2025-02-28`
- **Latest Historical Closing Price:** 1,200.10
- **Average Closing Price:** 326.52
- **Minimum / Maximum Closing Price:** 5.49 / 1,600.90

## 2. Data Preprocessing Summary
- **Original Row Count:** 7,324
- **Cleaned Row Count:** 7,324
- **Total Rows Removed (Cleaning):** 0
- **Missing Values Detected & Handled:** 0 (rows with missing numeric values removed: 0)
- **Duplicate Records Removed:** 0
- **Invalid Date Rows Removed:** 0
- **Invalid Price / Volume Rows Removed:** 0
- **Modeling Rows (After 5-Day Lag Warmup & Final Target Removal):** 7,318

## 3. Feature Engineering & Target Variable
- **Target Variable (`target_return`):** `(next_close / current_close) - 1`
- **Input Features Used (10 Leakage-Free Features Available at Close of Day t):**
  1. `open`: Opening stock price at the start of the current trading day.
  2. `high`: Highest stock price reached during the current trading day.
  3. `low`: Lowest stock price reached during the current trading day.
  4. `close`: Closing stock price at the end of the current trading day.
  5. `volume`: Total number of shares traded during the current trading day.
  6. `daily_return`: Current day's close-to-close return: (close_t / close_{t-1}) - 1. Measures how much the stock price changed compared to the previous trading day's close.
  7. `intraday_return`: Current day's open-to-close return: (close_t / open_t) - 1. Measures the percentage price movement within the current trading session.
  8. `daily_range`: Intraday price spread: high_t - low_t. Captures same-day price volatility.
  9. `lag_return_1`: Previous trading day's daily return: (close_{t-1} / close_{t-2}) - 1. Provides 1-day short-term momentum context without looking into the future.
  10. `return_5d`: Five-day historical return: (close_t / close_{t-5}) - 1. Measures the stock's price trend over the past 5 trading days (approximately 1 trading week).

## 4. Chronological Train / Test Split & Hyperparameter Tuning
- **Training Set (First 80%):** 5,854 observations (`1996-01-08` to `2019-03-27`)
- **Testing Set (Last 20% Unseen):** 1,464 observations (`2019-03-28` to `2025-02-27`)
- **Scaling Strategy:** `StandardScaler` inside a `scikit-learn` `Pipeline` fitted strictly on the training set.
- **Ridge Regression Best Alpha (TimeSeriesSplit CV):** `10.0`
- **Lasso Regression Best Alpha (TimeSeriesSplit CV):** `0.0005` (kept 6/10 features non-zero)

## 5. Model Evaluation Metrics (Unseen Chronological Test Set)

| Model | MAE | MSE | RMSE | R² Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 0.012671 | 0.000346 | 0.018595 | -0.024513 | 49.18% |
| **Lasso Regression** | 0.012516 | 0.000340 | 0.018439 | -0.007373 | 49.66% |

## 6. Model Comparison Conclusion
- **Conclusion:** Lasso Regression performed better on the selected test metrics.
- **Detailed Analysis:** On the unseen chronological test dataset (1,464 trading days), Lasso Regression achieved a slightly lower Root Mean Squared Error (Ridge RMSE: 0.018595 vs. Lasso RMSE: 0.018439) and R² Score (Ridge R²: -0.024513 vs. Lasso R²: -0.007373). Notice that R² is close to zero (or slightly negative) for both models. In empirical finance, daily stock returns are dominated by unpredictable news and market noise (consistent with the Efficient Market Hypothesis), so regularized models intentionally shrink coefficients toward zero to avoid overfitting. Historical price patterns alone cannot guarantee reliable future stock predictions.

## 8. Prediction Methodology & Project Limitations
- **Prediction Methodology:** Features are constructed strictly from historical observations up to day `t` using the exact same column order (`FEATURE_COLUMNS`) and mathematical definitions as model training, then scaled via the training-fitted `StandardScaler` inside the selected model's `Pipeline`.
- **Limitations:**
  1. Historical OHLCV prices do not capture breaking corporate news, earnings surprises, regulatory changes, or macroeconomic shocks.
  2. Ridge and Lasso Regression model only linear relationships between engineered features and next-day returns.
  3. Daily stock returns contain substantial market noise, so $R^2$ values on unseen data are naturally close to zero.
- **Disclaimer:** Machine Learning predictions are estimates based on historical data. Actual stock returns may differ substantially. This application is for educational purposes and is not investment advice.
