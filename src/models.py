# src/models.py

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .data_utils import (
    MIN_TRAIN_POINTS,
    build_lagged_supervised,
    compute_error_metrics,
    get_default_lag,
    get_default_rolling_window,
    get_seasonal_period,
    recursive_forecast_regressor,
)

# Optional deps
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False


class BaseModel:
    display_name = "Base"

    def __init__(self):
        self.y_train: Optional[pd.Series] = None
        self.freq_alias: Optional[str] = None

    def fit(self, y: pd.Series, freq_alias: str):
        self.y_train = y.astype(float)
        self.freq_alias = freq_alias

    def forecast(self, steps: int) -> pd.Series:
        raise NotImplementedError


class NaiveModel(BaseModel):
    display_name = "Naive (Last Observation)"

    def forecast(self, steps: int) -> pd.Series:
        last_value = self.y_train.iloc[-1]
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)
        values = np.repeat(last_value, steps)
        return pd.Series(values, index=idx)


class SeasonalNaiveModel(BaseModel):
    display_name = "Seasonal Naive"

    def forecast(self, steps: int) -> pd.Series:
        season = get_seasonal_period(self.freq_alias)
        if len(self.y_train) < season:
            raise ValueError("Not enough history for seasonal naive model.")

        last_season = self.y_train.iloc[-season:].values
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)

        values = []
        for i in range(steps):
            values.append(last_season[i % season])
        return pd.Series(values, index=idx)


class RollingMeanModel(BaseModel):
    display_name = "Rolling Mean"

    def __init__(self):
        super().__init__()
        self.window = None

    def fit(self, y: pd.Series, freq_alias: str):
        """
        Fit just stores the training series and picks a sensible window
        (7 for daily, 24 for hourly, etc). We require at least `window`
        observations.
        """
        super().fit(y, freq_alias)
        self.window = get_default_rolling_window(freq_alias)
        if len(self.y_train) < self.window:
            raise ValueError("Not enough data for selected rolling window.")

    def forecast(self, steps: int) -> pd.Series:
        """
        Recursive rolling-mean forecast:

        At each future step:
          - Take the last `self.window` values from the *current* history
            (which includes previous predictions),
          - Predict the next value as their mean,
          - Append that prediction to the history.

        This gives a smooth, evolving baseline instead of a flat line.
        """
        history = list(self.y_train.values)

        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)

        preds = []
        for _ in range(steps):
            window_vals = history[-self.window:]
            pred = float(np.mean(window_vals))
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, index=idx)


class ExpSmoothingModel(BaseModel):
    display_name = "Exponential Smoothing (Holt-Winters)"

    def __init__(self):
        super().__init__()
        self.fitted_model = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        season = get_seasonal_period(freq_alias)
        if len(y) < 2 * season:
            raise ValueError("Not enough data for Holt-Winters model.")

        self.fitted_model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=season,
            initialization_method="estimated",
        ).fit(optimized=True)

    def forecast(self, steps: int) -> pd.Series:
        return self.fitted_model.forecast(steps)


class SarimaModel(BaseModel):
    display_name = "SARIMA"

    def __init__(self):
        super().__init__()
        self.fitted_model = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        season = get_seasonal_period(freq_alias)

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, season)

        if len(y) < 3 * season:
            raise ValueError("Not enough data for SARIMA model.")

        self.fitted_model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def forecast(self, steps: int) -> pd.Series:
        forecast_res = self.fitted_model.get_forecast(steps)
        return forecast_res.predicted_mean


class ProphetModel(BaseModel):
    display_name = "Prophet"

    def __init__(self, include_holidays: bool = False):
        super().__init__()
        self.include_holidays = include_holidays
        self.model = None

    def fit(self, y: pd.Series, freq_alias: str):
        if not _HAS_PROPHET:
            raise ImportError(
                "prophet is not installed. Install with `pip install prophet`."
            )
        super().fit(y, freq_alias)
        df_p = pd.DataFrame({"ds": y.index, "y": y.values})
        m = Prophet()
        if self.include_holidays:
            m.add_country_holidays(country_name="US")
        m.fit(df_p)
        self.model = m

    def forecast(self, steps: int) -> pd.Series:
        future = self.model.make_future_dataframe(
            periods=steps, freq=self.freq_alias, include_history=False
        )
        fc = self.model.predict(future)
        return pd.Series(fc["yhat"].values, index=future["ds"])


class LagLinearModel(BaseModel):
    display_name = "Lag-based Linear Regression"

    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.max_lag = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        self.max_lag = get_default_lag(freq_alias)
        X, y_target, _ = build_lagged_supervised(
            self.y_train, self.max_lag, add_calendar=True
        )
        if len(y_target) < MIN_TRAIN_POINTS:
            raise ValueError("Not enough data for lag-based linear regression.")
        self.model.fit(X, y_target)

    def forecast(self, steps: int) -> pd.Series:
        return recursive_forecast_regressor(
            self.model,
            self.y_train,
            steps,
            self.max_lag,
            self.freq_alias,
            add_calendar=True,
        )


class LagXGBModel(BaseModel):
    display_name = "Lag-based XGBoost"

    def __init__(self):
        super().__init__()
        self.model = None
        self.max_lag = None

    def fit(self, y: pd.Series, freq_alias: str):
        if not _HAS_XGB:
            raise ImportError(
                "xgboost is not installed. Install with `pip install xgboost`."
            )
        super().fit(y, freq_alias)
        self.max_lag = get_default_lag(freq_alias)

        X, y_target, _ = build_lagged_supervised(
            self.y_train, self.max_lag, add_calendar=True
        )
        n_samples = len(y_target)
        if n_samples < MIN_TRAIN_POINTS:
            raise ValueError("Not enough data for lag-based XGBoost.")

        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_target[:split_idx], y_target[split_idx:]

        shared_params = {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        param_candidates = [
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        ]

        best_rmse = np.inf
        best_model = None

        for candidate in param_candidates:
            params = {**shared_params, **candidate}
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

        best_model.fit(X, y_target)
        self.model = best_model

    def forecast(self, steps: int) -> pd.Series:
        return recursive_forecast_regressor(
            self.model,
            self.y_train,
            steps,
            self.max_lag,
            self.freq_alias,
            add_calendar=True,
        )


MODEL_REGISTRY = {
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
    "rolling_mean": RollingMeanModel,
    "exp_smoothing": ExpSmoothingModel,
    "sarima": SarimaModel,
    "prophet": ProphetModel,
    "lag_linear": LagLinearModel,
    "lag_xgb": LagXGBModel,
}

MODEL_DISPLAY_NAMES = {
    "naive": "Naive (Last Observation)",
    "seasonal_naive": "Seasonal Naive",
    "rolling_mean": "Rolling Mean",
    "exp_smoothing": "Exponential Smoothing (Holt-Winters)",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lag_linear": "Lag-based Linear Regression",
    "lag_xgb": "Lag-based XGBoost",
}


def instantiate_model(key: str, include_holidays: bool = False) -> BaseModel:
    ModelCls = MODEL_REGISTRY[key]
    if key == "prophet":
        return ModelCls(include_holidays=include_holidays)
    return ModelCls()


def run_future_forecasts(
    y: pd.Series,
    freq_alias: str,
    horizon: int,
    model_keys: List[str],
    include_prophet_holidays: bool,
    progress_callback=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Train each selected model on ALL data and forecast next `horizon` steps.
    Returns:
        combined_df: DataFrame with actual + forecasts
        errors: list of error messages (for models that failed)
    """
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if not model_keys:
        raise ValueError("No models selected.")

    forecasts: Dict[str, pd.Series] = {}
    errors: List[str] = []

    n_models = len(model_keys)

    for i, key in enumerate(model_keys, start=1):
        display_name = MODEL_DISPLAY_NAMES[key]
        try:
            model = instantiate_model(key, include_holidays=include_prophet_holidays)
            model.fit(y, freq_alias)
            fc = model.forecast(horizon)
            forecasts[key] = fc
        except Exception as e:
            errors.append(f"{display_name} failed: {e}")

        if progress_callback is not None:
            progress_callback(i / n_models)

    combined = pd.DataFrame({"actual": y})
    for key, fc in forecasts.items():
        col_name = f"forecast_{key}"
        hist_part = pd.Series(np.nan, index=y.index)
        full_series = pd.concat([hist_part, fc])
        combined[col_name] = full_series

    combined = combined.sort_index()
    return combined, errors


def run_backtest_forecasts(
    y: pd.Series,
    freq_alias: str,
    y_train: pd.Series,
    y_test: pd.Series,
    model_keys: List[str],
    include_prophet_holidays: bool,
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, List[str]]:
    """
    Run backtests for each model:
      - Train: y_train
      - Test: y_test
    Returns:
      combined_df: actual + forecasts (backtest window only)
      metrics_df: per-model metrics
      backtest_index: index of backtest window
      errors: list of error messages
    """
    if not model_keys:
        raise ValueError("No models selected for backtest.")

    backtest_index = y_test.index
    forecasts: Dict[str, pd.Series] = {}
    metrics_rows = []
    errors: List[str] = []

    n_models = len(model_keys)

    for i, key in enumerate(model_keys, start=1):
        display_name = MODEL_DISPLAY_NAMES[key]
        try:
            model = instantiate_model(key, include_holidays=include_prophet_holidays)
            model.fit(y_train, freq_alias)
            fc = model.forecast(len(y_test))
            fc = pd.Series(fc.values, index=backtest_index)
            forecasts[key] = fc

            metrics = compute_error_metrics(actual=y_test, forecast=fc)
            metrics["Model"] = display_name
            metrics_rows.append(metrics)
        except Exception as e:
            errors.append(f"{display_name} failed: {e}")

        if progress_callback is not None:
            progress_callback(i / n_models)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    else:
        metrics_df = pd.DataFrame(columns=["ME", "MAE", "RMSE", "MAPE"])

    combined = pd.DataFrame({"actual": y})
    for key, fc in forecasts.items():
        col_name = f"forecast_{key}"
        series_all = pd.Series(np.nan, index=y.index)
        series_all.loc[backtest_index] = fc
        combined[col_name] = series_all

    combined = combined.sort_index()
    return combined, metrics_df, backtest_index, errors
