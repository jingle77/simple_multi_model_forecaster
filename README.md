# simple_multi_model_forecaster

Multi-model time series forecasting + Monte Carlo error simulation in Streamlit.

This app lets you:

- Upload a single **daily or hourly** KPI time series (CSV).
- Choose between:
  - **Future Forecast** mode (multi-step ahead forecasts), or  
  - **Backtest (within data range)** mode (train/holdout split by calendar dates or last N points).
- Run multiple models side-by-side:
  - Naive
  - Seasonal Naive
  - Rolling Mean (recursive)
  - Exponential Smoothing (Holt–Winters)
  - SARIMA (via `statsmodels.SARIMAX` without exog)
  - Prophet
  - Lag-based Linear Regression
  - Lag-based XGBoost (with small internal hyperparameter search)
- View interactive forecasts with:
  - Actual vs each model’s forecast (actual = solid, forecasts = dotted)
  - Shaded backtest window and dynamic date zoom
- Run **Monte Carlo** simulations via residual bootstrapping for a chosen model + backtest window.

---

## 1. Cloning / Downloading the Repo

### Using Git (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/jingle77/simple_multi_model_forecaster.git

# 2. Enter the project folder
cd simple_multi_model_forecaster

# 3. Create a virtual environment
python -m venv .venv

# 4. Install requirements
## Windows
.venv\Scripts\activate

## Linux
source .venv/bin/activate

# 5. install dependencies
pip install -r requirements.txt

# 6. run application
streamlit run app.py
```