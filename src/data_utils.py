# src/data_utils.py

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd


def infer_freq_label_from_raw(dt_series: pd.Series) -> str:
    """
    Crude frequency inference: 'Hourly' if median delta <= 2 hours, else 'Daily'.
    """
    dt_parsed = pd.to_datetime(dt_series, errors="coerce")
    dt_parsed = dt_parsed.dropna().sort_values()
    if len(dt_parsed) < 3:
        return "Daily"

    deltas = dt_parsed.diff().dropna()
    median_delta = deltas.median()

    if median_delta <= pd.Timedelta(hours=2):
        return "Hourly"
    return "Daily"


def preprocess_data(
    df: pd.DataFrame,
    dt_col: str,
    y_col: str,
    freq_label: str,
) -> Tuple[pd.Series, str, int]:
    """
    Parse datetime column, coerce KPI column to numeric, sort, set index,
    and resample to regular frequency (Daily/Hourly) with simple interpolation.

    Returns:
      y_series: pd.Series indexed by DatetimeIndex
      freq_alias: 'D' or 'H'
      dropped_rows: count of rows dropped due to non-numeric KPI
    """
    if dt_col not in df.columns:
        raise ValueError(f"Datetime column '{dt_col}' not in DataFrame.")
    if y_col not in df.columns:
        raise ValueError(f"KPI column '{y_col}' not in DataFrame.")

    dt_parsed = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.copy()
    df["_dt_parsed"] = dt_parsed
    df = df.dropna(subset=["_dt_parsed"])

    if df.empty:
        raise ValueError("No valid datetime values after parsing.")

    # Coerce KPI to numeric
    y_numeric = pd.to_numeric(df[y_col], errors="coerce")
    dropped_rows = int((y_numeric.isna()).sum())
    df["_y_numeric"] = y_numeric
    df = df.dropna(subset=["_y_numeric"])

    if df.empty:
        raise ValueError("No valid numeric KPI values after coercion.")

    df = df.sort_values("_dt_parsed")
    series = pd.Series(df["_y_numeric"].values, index=pd.DatetimeIndex(df["_dt_parsed"]))

    freq_alias = "D" if freq_label == "Daily" else "H"

    # Resample to a regular grid and interpolate
    series = (
        series.resample(freq_alias)
        .mean()
        .interpolate(method="time")
    )

    series.name = y_col
    return series, freq_alias, dropped_rows


def split_backtest_date_range(
    y: pd.Series,
    start_date: dt.date,
    end_date: dt.date,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split series into train (before start_date) and test (start_date..end_date, inclusive).
    """
    idx = y.index
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts <= idx.min():
        raise ValueError("Backtest start date must be after the first observation.")

    train_mask = idx < start_ts
    test_mask = (idx >= start_ts) & (idx <= end_ts)

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if y_train.empty or y_test.empty:
        raise ValueError("Backtest split produced empty train or test set.")

    return y_train, y_test


def split_backtest_last_n(
    y: pd.Series,
    n: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split series into train (all but last n points) and test (last n points).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(y) <= n:
        raise ValueError("Not enough data to hold out last n points.")

    y_train = y.iloc[:-n]
    y_test = y.iloc[-n:]
    return y_train, y_test


# ----------------------------------------------------------------------
# Calendar / holiday feature helpers
# ----------------------------------------------------------------------


def _us_thanksgiving(year: int) -> dt.date:
    """
    US Thanksgiving = fourth Thursday in November.
    Returns a date object.
    """
    d = dt.date(year, 11, 1)
    # weekday: Monday=0, Sunday=6; Thursday=3
    days_to_thursday = (3 - d.weekday()) % 7
    first_thursday = d + dt.timedelta(days=days_to_thursday)
    thanksgiving = first_thursday + dt.timedelta(weeks=3)
    return thanksgiving


def _cyber_monday(thanksgiving: dt.date) -> dt.date:
    """
    Cyber Monday = Monday after Thanksgiving.
    Thanksgiving is Thursday, so Monday is +4 days.
    """
    return thanksgiving + dt.timedelta(days=4)


def build_calendar_feature_frame(
    index: pd.DatetimeIndex,
    include_calendar: bool = True,
    include_thanksgiving_window: bool = True,
) -> pd.DataFrame:
    """
    Given a DatetimeIndex, build a DataFrame of calendar/holiday features:

      - dow (0=Mon,...,6=Sun)
      - is_weekend (Sat/Sun)
      - month (1-12)
      - quarter (1-4)
      - is_thanksgiving_to_cyber_monday (1 if date ∈ [Thanksgiving, Cyber Monday])
      - is_pre_thanksgiving_window (1 if date ∈ [tg-14, tg-1])

    Works for both daily and hourly data (we use the .date() part).
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    df = pd.DataFrame(index=index)

    # Basic calendar features
    if include_calendar:
        df["dow"] = index.dayofweek
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
        df["month"] = index.month
        df["quarter"] = index.quarter

    # Thanksgiving windows
    if include_thanksgiving_window:
        dates = index.date
        years = sorted({d.year for d in dates})
        tg_by_year = {year: _us_thanksgiving(year) for year in years}
        cm_by_year = {year: _cyber_monday(tg_by_year[year]) for year in years}

        is_thanksgiving_to_cyber = []
        is_pre_window = []

        for d in dates:
            tg = tg_by_year[d.year]
            cm = cm_by_year[d.year]

            in_tg_cm = 1 if (tg <= d <= cm) else 0

            pre_start = tg - dt.timedelta(days=14)
            pre_end = tg - dt.timedelta(days=1)
            in_pre = 1 if (pre_start <= d <= pre_end) else 0

            is_thanksgiving_to_cyber.append(in_tg_cm)
            is_pre_window.append(in_pre)

        df["is_thanksgiving_to_cyber_monday"] = np.array(is_thanksgiving_to_cyber, dtype=int)
        df["is_pre_thanksgiving_window"] = np.array(is_pre_window, dtype=int)

    return df
