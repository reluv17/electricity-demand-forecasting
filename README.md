# Electricity Demand Forecasting — Time Series Regression

Hourly electricity consumption forecasting using ensemble machine learning on 17 years of building energy data. This project covers time series feature engineering, lag-based modeling, cyclical encoding, and hyperparameter tuning with TimeSeriesSplit cross-validation.

---

## Overview

Accurate electricity demand forecasting is critical for energy management, grid stability, and cost optimization. This project predicts hourly electricity imported from the grid (kW) using weather data, temporal features, and engineered lag/rolling statistics. Two ensemble models — Random Forest and Gradient Boosting — are compared, with Gradient Boosting achieving a test R² of **0.9795**.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Building energy monitoring system |
| Training period | June 2001 – February 2018 |
| Test period | February 2018 – December 2019 |
| Training records | 146,621 hourly observations |
| Test records | 16,292 hourly observations |
| Target | Electricity imported from grid (kW) |

**Original features:** Horizontal solar irradiation (W), outdoor air temperature (°C), outdoor air humidity (%), wind speed (m/s), wind direction

---

## Feature Engineering

Starting from 6 raw features, 21 engineered features were added for a total of 22 model inputs:

**Temporal features**
- Day of month, week of year, weekend indicator

**Cyclical encodings** (sin/cos transformations to preserve circular continuity)
- Hour of day, month, day of week

**Lag features** (historical electricity consumption)
- 1-hour lag, 24-hour lag, 168-hour (1-week) lag

**Rolling statistics** (computed on lagged values to prevent leakage)
- 24-hour rolling mean and standard deviation
- 7-day rolling mean

**Interaction features**
- Temperature squared (captures U-shaped heating/cooling relationship)

---

## Key Findings from EDA

- Strong daily cycle: consumption peaks midday (~800 kW) and drops at night (~400 kW)
- Weekday consumption ~100 kW higher than weekends on average
- Seasonal pattern: highest consumption in summer months (cooling load)
- ACF analysis confirmed strong autocorrelation at 24h and 168h lags
- U-shaped temperature–consumption relationship motivated `temp_squared` feature

---

## Models

### Random Forest Regressor
- Ensemble of 200 decision trees using bagging
- Naturally handles non-linear relationships and mixed feature types
- No scaling required

### Gradient Boosting Regressor
- Sequential ensemble where each tree corrects prior errors
- L1 regularization via `subsample=0.8`
- Generally stronger on structured/tabular data

**Hyperparameter tuning:** GridSearchCV with TimeSeriesSplit (3 folds) — respects temporal ordering to prevent data leakage from future observations into training.

---

## Results

| Model | Train R² | CV R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| Random Forest | 0.9834 | 0.9545 | 0.9774 | 40.36 kW | 27.97 kW |
| **Gradient Boosting** | **0.9776** | **0.9636** | **0.9795** | **38.46 kW** | **26.96 kW** |

**Gradient Boosting is the recommended model** — higher test R², lower error across all metrics, and a negative overfitting gap (-0.19%), meaning it generalizes slightly better to unseen data than it does on training data.

**Top features by importance (Random Forest):**

| Feature | Importance |
|---|---|
| electricity_lag_1h | 35.8% |
| electricity_lag_168h | 21.4% |
| electricity_lag_24h | 12.9% |
| electricity_rolling_mean_24h | 6.9% |
| hour_cos | 5.4% |

Lag features dominate, confirming that recent consumption history is the strongest predictor of future demand.

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy
- `RandomForestRegressor`, `GradientBoostingRegressor`
- `GridSearchCV`, `TimeSeriesSplit`
- Matplotlib, Seaborn, Plotly
- `statsmodels` (ACF/PACF analysis)

---

## Repository Structure

```
electricity-demand-forecasting/
├── notebook.ipynb    # Full pipeline: EDA, feature engineering, modeling, evaluation
├── README.md
```

---

## Business Applications

This forecasting approach applies directly to:
- **Energy management:** Optimize grid import schedules and reduce peak demand charges
- **Facility operations:** Anticipate HVAC and lighting load requirements
- **Utility planning:** Support demand response programs and capacity planning
- **Sustainability reporting:** Track and forecast energy consumption against targets
