# src/monte_carlo.py

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd

from .data_utils import split_backtest_date_range
from .models import instantiate_model


def run_monte_carlo_simulation(
    y: pd.Series,
    freq_alias: str,
    model_key: str,
    start_date: dt.date,
    end_date: dt.date,
    include_prophet_holidays: bool,
    n_sims: int,
    metric_type: str,
    progress_callback=None,
) -> Tuple[np.ndarray, float]:
    """
    Monte Carlo via residual bootstrapping on a chosen model
    over a chosen backtest window.

    metric_type: 'ME' or 'MAE'
    """
    # Split train/test
    y_train, y_test = split_backtest_date_range(y, start_date, end_date)
    if len(y_test) < 2:
        raise ValueError("Backtest window too short for Monte Carlo.")

    # Train + forecast
    model = instantiate_model(model_key, include_holidays=include_prophet_holidays)
    model.fit(y_train, freq_alias)
    fc = model.forecast(len(y_test))
    fc = pd.Series(fc.values, index=y_test.index)

    # Residuals: e_t = actual - forecast
    residuals = y_test - fc
    residuals = residuals.dropna()
    if residuals.empty:
        raise ValueError("No residuals available for Monte Carlo.")

    # Observed metric
    if metric_type == "ME":
        observed_metric = float((fc - y_test).mean())
    else:  # MAE
        observed_metric = float((fc - y_test).abs().mean())

    metrics = np.zeros(n_sims)
    update_every = max(1, n_sims // 20)

    actual_vals = y_test.values
    forecast_vals = fc.values

    for i in range(n_sims):
        sampled_resid = residuals.sample(
            n=len(residuals), replace=True
        ).values

        simulated_y = forecast_vals + sampled_resid
        diff = simulated_y - actual_vals

        if metric_type == "ME":
            metrics[i] = diff.mean()
        else:
            metrics[i] = np.abs(diff).mean()

        if progress_callback is not None and (i + 1) % update_every == 0:
            progress_callback((i + 1) / n_sims)

    if progress_callback is not None:
        progress_callback(1.0)

    return metrics, observed_metric
