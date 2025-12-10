# src/monte_carlo.py

import datetime as dt
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_utils import split_backtest_date_range
from .models import MODEL_REGISTRY, MODEL_DISPLAY_NAMES


def run_monte_carlo_simulation(
    y: pd.Series,
    freq_alias: str,
    model_key: str,
    start_date: dt.date,
    end_date: dt.date,
    include_prophet_holidays: bool,
    n_sims: int,
    metric_type: str,
    use_calendar_features: bool = False,
    use_thanksgiving_window: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[List[float], float]:
    """
    Monte Carlo via residual bootstrapping:

      1. Backtest specified model on [start_date, end_date] window
         to get residuals e_t = actual_t - forecast_t.
      2. For i=1..n_sims:
           - Sample residuals with replacement
           - Construct simulated_y = forecast + sampled_resid
           - Compute scalar summary metric D^(i):
               - if metric_type == "ME": mean(simulated_y - actual)
               - if metric_type == "MAE": mean(|simulated_y - actual|)
      3. Return list of all D^(i) and the observed metric from the real backtest.

    metric_type: "ME" or "MAE"
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")

    if n_sims <= 0:
        raise ValueError("Number of simulations must be positive.")

    # 1) Backtest split
    y_train, y_test = split_backtest_date_range(y, start_date, end_date)

    # 2) Fit model and get backtest forecast
    model_cls = MODEL_REGISTRY[model_key]
    model = model_cls()

    # Prophet options
    if model_key == "prophet":
        if hasattr(model, "set_include_holidays"):
            model.set_include_holidays(include_prophet_holidays)
        if hasattr(model, "set_special_holiday_windows"):
            model.set_special_holiday_windows(use_thanksgiving_window)

    # Lag models: calendar options
    if model_key in ("lag_linear", "lag_xgb"):
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

    # Residuals: actual - forecast
    residuals = (y_test - fc).dropna()
    if residuals.empty:
        raise ValueError("No residuals available for Monte Carlo simulation.")

    # Observed metric from actual vs forecast
    diff = fc - y_test
    if metric_type == "ME":
        observed_metric = float(diff.mean())
    else:  # MAE
        observed_metric = float(diff.abs().mean())

    # 3) Monte Carlo simulations
    metrics: List[float] = []
    res_values = residuals.values
    n = len(y_test)

    for i in range(n_sims):
        sampled_resid = np.random.choice(res_values, size=n, replace=True)
        simulated_y = fc.values + sampled_resid

        if metric_type == "ME":
            D = float((simulated_y - y_test.values).mean())
        else:
            D = float(np.abs(simulated_y - y_test.values).mean())
        metrics.append(D)

        if progress_callback is not None and (i + 1) % max(1, n_sims // 50) == 0:
            progress_callback((i + 1) / n_sims)

    return metrics, observed_metric
