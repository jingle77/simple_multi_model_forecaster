# src/models.py

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
except ImportError:  # fallback if environment uses fbprophet
    from fbprophet import Prophet  # type: ignore

from .data_utils import build_calendar_feature_frame, _us_thanksgiving, _cyber_monday


# ----------------------------------------------------------------------
# Base model and helpers
# ----------------------------------------------------------------------


class BaseModel:
    display_name = "BaseModel"

    def __init__(self):
        self.y_train: Optional[pd.Series] = None
        self.freq_alias: Optional[str] = None

    def fit(self, y: pd.Series, freq_alias: str):
        self.y_train = y
        self.freq_alias = freq_alias

    def forecast(self, steps: int) -> pd.Series:
        raise NotImplementedError


def get_default_seasonal_period(freq_alias: str) -> int:
    return 7 if freq_alias == "D" else 24


def get_default_rolling_window(freq_alias: str) -> int:
    return 7 if freq_alias == "D" else 24


# ----------------------------------------------------------------------
# Simple baseline models
# ----------------------------------------------------------------------


class NaiveModel(BaseModel):
    display_name = "Naive (Last Observation)"

    def forecast(self, steps: int) -> pd.Series:
        if self.y_train is None:
            raise RuntimeError("Model must be fit before forecasting.")
        last_val = self.y_train.iloc[-1]
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)
        preds = np.full(steps, float(last_val))
        return pd.Series(preds, index=idx)


class SeasonalNaiveModel(BaseModel):
    display_name = "Seasonal Naive"

    def __init__(self):
        super().__init__()
        self.seasonal_period: Optional[int] = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        self.seasonal_period = get_default_seasonal_period(freq_alias)
        if len(self.y_train) < self.seasonal_period:
            raise ValueError("Not enough data for seasonal naive model.")

    def forecast(self, steps: int) -> pd.Series:
        if self.y_train is None or self.seasonal_period is None:
            raise RuntimeError("Model must be fit before forecasting.")
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)

        history = self.y_train.values
        s = self.seasonal_period
        preds = []
        for k in range(steps):
            pred = history[-s + (k % s)]
            preds.append(float(pred))
        return pd.Series(preds, index=idx)


class RollingMeanModel(BaseModel):
    display_name = "Rolling Mean"

    def __init__(self):
        super().__init__()
        self.window: Optional[int] = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        self.window = get_default_rolling_window(freq_alias)
        if len(self.y_train) < self.window:
            raise ValueError("Not enough data for selected rolling window.")

    def forecast(self, steps: int) -> pd.Series:
        """
        Recursive rolling-mean forecast.

        At each future step:
          - Take the last `self.window` values from history (including previous
            predictions),
          - Predict as their mean,
          - Append prediction to history.
        """
        if self.y_train is None or self.window is None:
            raise RuntimeError("Model must be fit before forecasting.")

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


# ----------------------------------------------------------------------
# Classical TS models (Holt–Winters, SARIMA)
# ----------------------------------------------------------------------


class ExpSmoothingModel(BaseModel):
    display_name = "Exponential Smoothing (Holt-Winters)"

    def __init__(self):
        super().__init__()
        self.model_fit = None
        self.seasonal_period: Optional[int] = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        self.seasonal_period = get_default_seasonal_period(freq_alias)
        self.model_fit = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_period,
        ).fit(optimized=True)

    def forecast(self, steps: int) -> pd.Series:
        if self.model_fit is None or self.y_train is None:
            raise RuntimeError("Model must be fit before forecasting.")
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)
        preds = self.model_fit.forecast(steps)
        preds.index = idx
        return preds


class SarimaModel(BaseModel):
    display_name = "SARIMA"

    def __init__(self):
        super().__init__()
        self.model_fit = None
        self.seasonal_period: Optional[int] = None

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)
        self.seasonal_period = get_default_seasonal_period(freq_alias)

        # Lightweight fixed orders; can be tuned later.
        order = (1, 1, 1)
        seasonal_order = (0, 1, 1, self.seasonal_period)

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)

    def forecast(self, steps: int) -> pd.Series:
        if self.model_fit is None or self.y_train is None:
            raise RuntimeError("Model must be fit before forecasting.")
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)
        preds = self.model_fit.forecast(steps)
        preds.index = idx
        return preds


# ----------------------------------------------------------------------
# Prophet model (with special Thanksgiving windows)
# ----------------------------------------------------------------------


class ProphetModel(BaseModel):
    display_name = "Prophet"

    def __init__(self):
        super().__init__()
        self.model: Optional[Prophet] = None
        self.include_holidays: bool = False
        self.use_thanksgiving_windows: bool = False

    def set_include_holidays(self, include: bool):
        self.include_holidays = include

    def set_special_holiday_windows(self, use_windows: bool):
        """Enable special windows for Thanksgiving → Cyber Monday and pre-Thanksgiving."""
        self.use_thanksgiving_windows = use_windows

    def _build_holiday_df(self, index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        if not self.use_thanksgiving_windows:
            return None

        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)

        years = sorted({ts.year for ts in index})
        rows = []
        for year in years:
            tg = _us_thanksgiving(year)
            cm = _cyber_monday(tg)

            # Window 1: Thanksgiving → Cyber Monday
            rows.append(
                {
                    "holiday": "thanksgiving_cyber_window",
                    "ds": pd.Timestamp(tg),
                    "lower_window": 0,
                    "upper_window": (cm - tg).days,
                }
            )

            # Window 2: 14 days leading up to Thanksgiving
            rows.append(
                {
                    "holiday": "pre_thanksgiving_window",
                    "ds": pd.Timestamp(tg),
                    "lower_window": -14,
                    "upper_window": -1,
                }
            )

        return pd.DataFrame(rows)

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)

        holidays_df = self._build_holiday_df(y.index)

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=(freq_alias == "H"),
            holidays=holidays_df,
        )

        if self.include_holidays:
            self.model.add_country_holidays(country_name="US")

        df = pd.DataFrame({"ds": y.index.to_pydatetime(), "y": y.values})
        self.model.fit(df)

    def forecast(self, steps: int) -> pd.Series:
        if self.model is None or self.y_train is None or self.freq_alias is None:
            raise RuntimeError("Model must be fit before forecasting.")

        freq = "D" if self.freq_alias == "D" else self.freq_alias
        future = self.model.make_future_dataframe(
            periods=steps,
            freq=freq,
            include_history=False,
        )
        forecast_df = self.model.predict(future)
        yhat = forecast_df["yhat"].values
        idx = pd.DatetimeIndex(forecast_df["ds"])
        return pd.Series(yhat, index=idx)


# ----------------------------------------------------------------------
# Lag-based models (Linear Regression, XGBoost) with calendar features
# ----------------------------------------------------------------------


def build_lagged_supervised(
    y: pd.Series,
    max_lag: int,
    calendar_features: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised dataset:

      y_t ~ [y_{t-1}, ..., y_{t-max_lag}] + (optional calendar features at time t)

    Returns:
      X, y_target
    """
    df = pd.DataFrame({"y": y})

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    if calendar_features is not None:
        df = df.join(calendar_features)

    df = df.dropna()
    y_target = df["y"]
    X = df.drop(columns=["y"])
    return X, y_target


class LagLinearModel(BaseModel):
    display_name = "Lag-based Linear Regression"

    def __init__(self, max_lag_daily: int = 14, max_lag_hourly: int = 48):
        super().__init__()
        self.max_lag_daily = max_lag_daily
        self.max_lag_hourly = max_lag_hourly
        self.max_lag: Optional[int] = None
        self.model = LinearRegression()
        self.use_calendar_features: bool = False
        self.use_thanksgiving_window: bool = False
        self.feature_columns: List[str] = []

    def set_calendar_options(self, use_calendar: bool, use_thanksgiving_window: bool):
        self.use_calendar_features = use_calendar
        self.use_thanksgiving_window = use_thanksgiving_window

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)

        self.max_lag = self.max_lag_daily if freq_alias == "D" else self.max_lag_hourly

        cal_feats = None
        if self.use_calendar_features or self.use_thanksgiving_window:
            cal_feats = build_calendar_feature_frame(
                y.index,
                include_calendar=self.use_calendar_features,
                include_thanksgiving_window=self.use_thanksgiving_window,
            )

        X, y_target = build_lagged_supervised(y, self.max_lag, calendar_features=cal_feats)
        if len(X) == 0:
            raise ValueError("Not enough data to build lagged features.")

        self.feature_columns = list(X.columns)
        self.model.fit(X, y_target)

    def _make_feature_row(self, history: List[float], ts: pd.Timestamp) -> pd.DataFrame:
        if self.max_lag is None:
            raise RuntimeError("Model must be fit before forecasting.")

        data = {}
        for lag in range(1, self.max_lag + 1):
            data[f"lag_{lag}"] = history[-lag]

        if self.use_calendar_features or self.use_thanksgiving_window:
            cal_df = build_calendar_feature_frame(
                pd.DatetimeIndex([ts]),
                include_calendar=self.use_calendar_features,
                include_thanksgiving_window=self.use_thanksgiving_window,
            )
            for col in cal_df.columns:
                data[col] = cal_df.iloc[0][col]

        row = pd.DataFrame([data])
        # align columns to training columns
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = 0
        row = row[self.feature_columns]
        return row

    def forecast(self, steps: int) -> pd.Series:
        if self.y_train is None or self.max_lag is None:
            raise RuntimeError("Model must be fit before forecasting.")

        history = list(self.y_train.values)
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)

        preds = []
        for ts in idx:
            row = self._make_feature_row(history, ts)
            pred = float(self.model.predict(row)[0])
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, index=idx)


class LagXGBModel(BaseModel):
    display_name = "Lag-based XGBoost"

    def __init__(self, max_lag_daily: int = 14, max_lag_hourly: int = 48):
        super().__init__()
        self.max_lag_daily = max_lag_daily
        self.max_lag_hourly = max_lag_hourly
        self.max_lag: Optional[int] = None
        self.model: Optional[XGBRegressor] = None
        self.use_calendar_features: bool = False
        self.use_thanksgiving_window: bool = False
        self.feature_columns: List[str] = []

    def set_calendar_options(self, use_calendar: bool, use_thanksgiving_window: bool):
        self.use_calendar_features = use_calendar
        self.use_thanksgiving_window = use_thanksgiving_window

    def fit(self, y: pd.Series, freq_alias: str):
        super().fit(y, freq_alias)

        self.max_lag = self.max_lag_daily if freq_alias == "D" else self.max_lag_hourly

        cal_feats = None
        if self.use_calendar_features or self.use_thanksgiving_window:
            cal_feats = build_calendar_feature_frame(
                y.index,
                include_calendar=self.use_calendar_features,
                include_thanksgiving_window=self.use_thanksgiving_window,
            )

        X, y_target = build_lagged_supervised(y, self.max_lag, calendar_features=cal_feats)
        if len(X) == 0:
            raise ValueError("Not enough data to build lagged features.")

        # Time-ordered 80/20 split
        n = len(X)
        split = int(0.8 * n)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y_target.iloc[:split], y_target.iloc[split:]

        param_candidates = [
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        ]

        shared_params = {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        best_rmse = float("inf")
        best_params = None

        for params in param_candidates:
            all_params = {**shared_params, **params}
            model = XGBRegressor(**all_params)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            rmse = float(np.sqrt(((preds_val - y_val) ** 2).mean()))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = all_params

        # Refit on all data with best params
        if best_params is None:
            best_params = {**shared_params, **param_candidates[0]}

        self.model = XGBRegressor(**best_params)
        self.model.fit(X, y_target)

        self.feature_columns = list(X.columns)

    def _make_feature_row(self, history: List[float], ts: pd.Timestamp) -> pd.DataFrame:
        if self.max_lag is None:
            raise RuntimeError("Model must be fit before forecasting.")

        data = {}
        for lag in range(1, self.max_lag + 1):
            data[f"lag_{lag}"] = history[-lag]

        if self.use_calendar_features or self.use_thanksgiving_window:
            cal_df = build_calendar_feature_frame(
                pd.DatetimeIndex([ts]),
                include_calendar=self.use_calendar_features,
                include_thanksgiving_window=self.use_thanksgiving_window,
            )
            for col in cal_df.columns:
                data[col] = cal_df.iloc[0][col]

        row = pd.DataFrame([data])
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = 0
        row = row[self.feature_columns]
        return row

    def forecast(self, steps: int) -> pd.Series:
        if self.y_train is None or self.max_lag is None or self.model is None:
            raise RuntimeError("Model must be fit before forecasting.")

        history = list(self.y_train.values)
        freq_offset = pd.tseries.frequencies.to_offset(self.freq_alias)
        start = self.y_train.index[-1] + freq_offset
        idx = pd.date_range(start=start, periods=steps, freq=self.freq_alias)

        preds = []
        for ts in idx:
            row = self._make_feature_row(history, ts)
            pred = float(self.model.predict(row)[0])
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, index=idx)


# ----------------------------------------------------------------------
# Registry and utilities
# ----------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
    "rolling_mean": RollingMeanModel,
    "exp_smoothing": ExpSmoothingModel,
    "sarima": SarimaModel,
    "prophet": ProphetModel,
    "lag_linear": LagLinearModel,
    "lag_xgb": LagXGBModel,
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "naive": NaiveModel.display_name,
    "seasonal_naive": SeasonalNaiveModel.display_name,
    "rolling_mean": RollingMeanModel.display_name,
    "exp_smoothing": ExpSmoothingModel.display_name,
    "sarima": SarimaModel.display_name,
    "prophet": ProphetModel.display_name,
    "lag_linear": LagLinearModel.display_name,
    "lag_xgb": LagXGBModel.display_name,
}


def _compute_error_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    aligned = pd.concat({"true": y_true, "pred": y_pred}, axis=1).dropna()
    if aligned.empty:
        return {"ME": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    err = aligned["pred"] - aligned["true"]
    me = float(err.mean())
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    # robust MAPE
    denom = aligned["true"].replace(0, np.nan)
    mape = float(((err.abs() / denom).dropna()).mean() * 100) if not denom.dropna().empty else np.nan
    return {"ME": me, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ----------------------------------------------------------------------
# Run forecasts: future mode and backtest mode
# ----------------------------------------------------------------------


def run_future_forecasts(
    y: pd.Series,
    freq_alias: str,
    horizon: int,
    model_keys: List[str],
    include_prophet_holidays: bool,
    use_calendar_features: bool = False,
    use_thanksgiving_window: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Train each selected model on ALL data and forecast next `horizon` steps.

    Returns:
        combined_df: DataFrame indexed by ALL timestamps (history + future),
                     with columns:
                       - 'actual' (NaN on future rows)
                       - 'forecast_<model_key>' for each model.
        errors: list of error messages (for models that failed).
    """
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if not model_keys:
        raise ValueError("No models selected.")

    forecasts: Dict[str, pd.Series] = {}
    errors: List[str] = []

    n_models = len(model_keys)

    for i, key in enumerate(model_keys, start=1):
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        try:
            model_cls = MODEL_REGISTRY[key]
            model = model_cls()

            # Prophet options
            if key == "prophet":
                if hasattr(model, "set_include_holidays"):
                    model.set_include_holidays(include_prophet_holidays)
                if hasattr(model, "set_special_holiday_windows"):
                    model.set_special_holiday_windows(use_thanksgiving_window)

            # Lag models: calendar options
            if key in ("lag_linear", "lag_xgb"):
                if hasattr(model, "set_calendar_options"):
                    model.set_calendar_options(
                        use_calendar_features,
                        use_thanksgiving_window,
                    )

            model.fit(y, freq_alias)
            fc = model.forecast(horizon)

            if not isinstance(fc, pd.Series):
                freq_offset = pd.tseries.frequencies.to_offset(freq_alias)
                start = y.index[-1] + freq_offset
                idx = pd.date_range(start=start, periods=horizon, freq=freq_alias)
                fc = pd.Series(fc, index=idx)

            forecasts[key] = fc
        except Exception as e:
            errors.append(f"{display_name} failed: {e}")

        if progress_callback is not None:
            progress_callback(i / n_models)

    # Build full index: history + forecast horizons
    full_index = y.index
    for fc in forecasts.values():
        full_index = full_index.union(fc.index)
    full_index = full_index.sort_values()

    combined = pd.DataFrame(index=full_index)
    combined["actual"] = y.reindex(full_index)

    for key, fc in forecasts.items():
        col_name = f"forecast_{key}"
        combined[col_name] = fc.reindex(full_index)

    return combined, errors


def run_backtest_forecasts(
    y: pd.Series,
    freq_alias: str,
    y_train: pd.Series,
    y_test: pd.Series,
    model_keys: List[str],
    include_prophet_holidays: bool,
    use_calendar_features: bool = False,
    use_thanksgiving_window: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, List[str]]:
    """
    Backtest: fit each model on y_train and forecast the y_test window.

    Returns:
        combined_df: DataFrame over the FULL index of y, with:
            - 'actual'
            - one forecast_<model_key> column, filled only on y_test.index
        metrics_df: DataFrame of ME, MAE, RMSE, MAPE by model
        backtest_index: DatetimeIndex of the test window
        errors: list of error messages.
    """
    if not model_keys:
        raise ValueError("No models selected for backtest.")

    combined = pd.DataFrame(index=y.index)
    combined["actual"] = y

    metrics_records = []
    errors: List[str] = []
    n_models = len(model_keys)

    for i, key in enumerate(model_keys, start=1):
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        try:
            model_cls = MODEL_REGISTRY[key]
            model = model_cls()

            if key == "prophet":
                if hasattr(model, "set_include_holidays"):
                    model.set_include_holidays(include_prophet_holidays)
                if hasattr(model, "set_special_holiday_windows"):
                    model.set_special_holiday_windows(use_thanksgiving_window)

            if key in ("lag_linear", "lag_xgb"):
                if hasattr(model, "set_calendar_options"):
                    model.set_calendar_options(
                        use_calendar_features,
                        use_thanksgiving_window,
                    )

            model.fit(y_train, freq_alias)
            fc = model.forecast(len(y_test))

            if not isinstance(fc, pd.Series):
                fc = pd.Series(fc, index=y_test.index)
            else:
                fc = fc.reindex(y_test.index)

            col_name = f"forecast_{key}"
            full_series = pd.Series(np.nan, index=y.index)
            full_series.loc[y_test.index] = fc
            combined[col_name] = full_series

            metrics = _compute_error_metrics(y_test, fc)
            metrics["Model"] = display_name
            metrics_records.append(metrics)
        except Exception as e:
            errors.append(f"{display_name} failed: {e}")

        if progress_callback is not None:
            progress_callback(i / n_models)

    if metrics_records:
        metrics_df = (
            pd.DataFrame(metrics_records)
            .set_index("Model")[["ME", "MAE", "RMSE", "MAPE"]]
        )
    else:
        metrics_df = pd.DataFrame(columns=["ME", "MAE", "RMSE", "MAPE"])

    backtest_index = y_test.index
    return combined, metrics_df, backtest_index, errors
