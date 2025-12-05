# src/data_utils.py

import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROLLING_DAILY = 7
DEFAULT_ROLLING_HOURLY = 24

DEFAULT_LAG_DAILY = 14
DEFAULT_LAG_HOURLY = 48

MIN_TRAIN_POINTS = 20  # minimal points required for more complex models


def infer_freq_label_from_raw(dt_series: pd.Series) -> Optional[str]:
    """
    Try to infer frequency from raw datetime series.
    Return 'Daily', 'Hourly', or None if unclear.
    """
    dt_series = pd.to_datetime(dt_series.dropna()).sort_values()
    try:
        freq = pd.infer_freq(dt_series)
    except ValueError:
        freq = None

    if freq is None:
        return None

    if "H" in freq:
        return "Hourly"
    if "D" in freq or "B" in freq:
        return "Daily"
    return None


def get_seasonal_period(freq_alias: str) -> int:
    """Return seasonal period depending on frequency."""
    if freq_alias == "D":
        return 7  # weekly
    if freq_alias == "H":
        return 24  # daily
    return 7


def get_default_rolling_window(freq_alias: str) -> int:
    if freq_alias == "D":
        return DEFAULT_ROLLING_DAILY
    if freq_alias == "H":
        return DEFAULT_ROLLING_HOURLY
    return DEFAULT_ROLLING_DAILY


def get_default_lag(freq_alias: str) -> int:
    if freq_alias == "D":
        return DEFAULT_LAG_DAILY
    if freq_alias == "H":
        return DEFAULT_LAG_HOURLY
    return DEFAULT_LAG_DAILY


def preprocess_data(
    df: pd.DataFrame,
    dt_col: str,
    y_col: str,
    freq_label: str,
) -> Tuple[pd.Series, str, int]:
    """
    Parse datetime, coerce KPI to numeric, sort, set index,
    resample to regular daily/hourly frequency, and forward-fill.

    Returns:
        y_series: pd.Series with DatetimeIndex
        freq_alias: 'D' or 'H'
        dropped_count: number of rows dropped due to non-numeric KPI
    """
    df = df.copy()

    # Parse datetime
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    # Coerce KPI to numeric
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    dropped = int(df[y_col].isna().sum())
    df = df.dropna(subset=[y_col])
    if df.empty:
        raise ValueError("No valid rows left after parsing KPI column.")

    # Sort and index
    df = df.sort_values(dt_col)
    df = df.set_index(dt_col)

    if freq_label == "Daily":
        freq_alias = "D"
    else:
        freq_alias = "H"

    # Resample to regular grid and forward fill
    y = df[[y_col]].resample(freq_alias).mean()
    y[y_col] = y[y_col].ffill()

    if y.empty:
        raise ValueError("Resampled time series is empty.")

    y_series = y[y_col]
    return y_series, freq_alias, dropped


def split_backtest_date_range(
    y: pd.Series,
    start_date: dt.date,
    end_date: dt.date,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split y into train and test using calendar date range.
    Train: all data strictly before backtest start.
    Test: start_date to end_date (inclusive).
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    y_test = y.loc[start_ts:end_ts]
    if y_test.empty:
        raise ValueError("Backtest window has no data after alignment.")

    y_train = y[y.index < y_test.index[0]]
    if y_train.empty:
        raise ValueError("Not enough data before backtest start for training.")

    return y_train, y_test


def split_backtest_last_n(
    y: pd.Series,
    n_points: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split y using last N points as backtest window.
    """
    if n_points <= 0:
        raise ValueError("N must be positive.")
    if n_points >= len(y):
        raise ValueError("N must be smaller than total number of observations.")

    y_test = y.iloc[-n_points:]
    y_train = y.iloc[:-n_points]
    return y_train, y_test


def compute_error_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> Dict[str, float]:
    """
    Compute ME, MAE, RMSE, MAPE between actual and forecast.
    """
    errors = forecast - actual
    me = errors.mean()
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    denom = actual.replace(0, np.nan).abs()
    mape = (errors.abs() / denom).mean() * 100
    mape = float(mape) if not np.isnan(mape) else np.nan

    return {
        "ME": float(me),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": mape,
    }


def build_lagged_supervised(
    y: pd.Series,
    max_lag: int,
    add_calendar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build a supervised dataset with lag features (and optional calendar features).

    For each time t:
        features: y(t-1), ..., y(t-max_lag), [dayofweek, hour]
        target: y(t)
    """
    df = pd.DataFrame({"y": y})
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    if add_calendar:
        df["dayofweek"] = df.index.dayofweek
        df["hour"] = df.index.hour

    df = df.dropna()
    if df.empty:
        raise ValueError("Not enough data to build lagged supervised dataset.")

    y_target = df["y"].values
    X = df.drop(columns=["y"]).values
    idx = df.index
    return X, y_target, idx


def recursive_forecast_regressor(
    model,
    y_train: pd.Series,
    steps: int,
    max_lag: int,
    freq_alias: str,
    add_calendar: bool = True,
) -> pd.Series:
    """
    Recursive multi-step forecast for lag-based models.
    At each step, use last max_lag values (including previous predictions)
    and optional calendar features.
    """
    if len(y_train) < max_lag:
        raise ValueError("Training series shorter than required lag length.")

    freq_offset = pd.tseries.frequencies.to_offset(freq_alias)
    last_ts = y_train.index[-1]
    future_index = pd.date_range(
        start=last_ts + freq_offset,
        periods=steps,
        freq=freq_alias,
    )

    history_values = list(y_train.values)
    preds = []

    for ts in future_index:
        if len(history_values) < max_lag:
            raise ValueError("Insufficient history during recursive forecast.")

        lag_features = history_values[-max_lag:]
        features = lag_features.copy()
        if add_calendar:
            features.append(ts.dayofweek)
            features.append(ts.hour)

        X_step = np.array(features).reshape(1, -1)
        y_pred = float(model.predict(X_step)[0])
        preds.append(y_pred)
        history_values.append(y_pred)

    return pd.Series(preds, index=future_index)
