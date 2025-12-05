# app.py

import datetime as dt

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import (
    MODEL_DISPLAY_NAMES,
    MODEL_REGISTRY,
    infer_freq_label_from_raw,
    preprocess_data,
    run_backtest_forecasts,
    run_future_forecasts,
    run_monte_carlo_simulation,
    split_backtest_date_range,
    split_backtest_last_n,
)


def main():
    st.set_page_config(
        page_title="Multi-Model Time Series Forecasting + Monte Carlo",
        layout="wide",
    )

    st.title("Multi-Model Time Series Forecasting + Monte Carlo Simulation")

    tabs = st.tabs(["Data & Settings", "Forecast & Evaluation", "Monte Carlo Simulation"])

    # ----------------------------------------------------------------
    # Tab 1: Data & Settings
    # ----------------------------------------------------------------
    with tabs[0]:
        st.header("1. Data & Settings")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Column selection
            dt_col = st.selectbox(
                "Datetime column",
                options=df.columns,
                index=0,
            )
            y_col = st.selectbox(
                "KPI (numeric) column",
                options=df.columns,
                index=min(1, len(df.columns) - 1),
            )

            # Frequency inference
            inferred_label = infer_freq_label_from_raw(df[dt_col])
            default_index = 0 if inferred_label != "Hourly" else 1

            st.caption(
                f"Inferred frequency (best guess): {inferred_label or 'Unknown'}"
            )
            freq_label = st.radio(
                "Select frequency to treat the data as:",
                options=["Daily", "Hourly"],
                index=default_index,
            )

            # Parse & preprocess
            y_series = None
            freq_alias = None
            try:
                y_series, freq_alias, dropped = preprocess_data(
                    df, dt_col, y_col, freq_label
                )
                if dropped > 0:
                    st.warning(f"Dropped {dropped} rows with non-numeric KPI values.")
            except Exception as e:
                st.error(f"Error while preprocessing data: {e}")
            else:
                st.success(
                    f"Parsed {len(y_series)} points from "
                    f"{y_series.index.min().date()} to {y_series.index.max().date()}."
                )

                # Store in session_state for other tabs
                st.session_state["y_series"] = y_series
                st.session_state["freq_alias"] = freq_alias

                # Mode: Future vs Backtest
                mode = st.radio(
                    "Forecast mode",
                    options=["Future Forecast", "Backtest (Within Data Range)"],
                )
                st.session_state["mode"] = mode

                min_date = y_series.index.min().date()
                max_date = y_series.index.max().date()

                backtest_method = None
                backtest_dates = None
                backtest_last_n = None
                horizon = None

                # ---------------- Future Forecast Mode ----------------
                if mode == "Future Forecast":
                    default_horizon = 7 if freq_alias == "D" else 24
                    horizon = st.number_input(
                        "Forecast horizon (number of periods ahead)",
                        min_value=1,
                        max_value=1000,
                        value=default_horizon,
                        step=1,
                    )
                    st.session_state["horizon"] = int(horizon)

                    st.caption(
                        "Optional: You can also choose a future end date for reference."
                    )
                    _ = st.date_input(
                        "Optional future end date (not used directly)",
                        value=max_date + dt.timedelta(days=default_horizon),
                    )

                # ---------------- Backtest Mode ----------------
                else:
                    backtest_method = st.radio(
                        "How to define the backtest window?",
                        options=["Date range", "Last N points"],
                    )

                    if backtest_method == "Date range":
                        default_range = [
                            max(min_date, max_date - dt.timedelta(days=30)),
                            max_date,
                        ]
                        backtest_dates = st.date_input(
                            "Backtest window (start and end dates)",
                            value=default_range,
                        )
                    else:
                        max_n = max(5, len(y_series) - 5)
                        backtest_last_n = st.number_input(
                            "Use last N points for backtest",
                            min_value=5,
                            max_value=max_n,
                            value=min(30, max_n),
                            step=1,
                        )

                    st.session_state["backtest_method"] = backtest_method
                    st.session_state["backtest_dates"] = backtest_dates
                    st.session_state["backtest_last_n"] = backtest_last_n

                # Model selection
                all_model_keys = list(MODEL_REGISTRY.keys())
                default_models = ["naive", "seasonal_naive", "rolling_mean", "exp_smoothing"]

                selected_model_keys = st.multiselect(
                    "Select models to run",
                    options=all_model_keys,
                    default=default_models,
                    format_func=lambda k: MODEL_DISPLAY_NAMES[k],
                )
                st.session_state["selected_model_keys"] = selected_model_keys

                # Prophet holidays checkbox
                include_prophet_holidays = st.checkbox(
                    "Include US holidays for Prophet?",
                    value=False,
                )
                st.session_state["include_prophet_holidays"] = include_prophet_holidays

                run_button = st.button("Run Forecasts")

                if run_button:
                    if not selected_model_keys:
                        st.error("Please select at least one model.")
                    else:
                        try:
                            progress_bar = st.progress(0.0)

                            def progress_cb(frac: float):
                                progress_bar.progress(frac)

                            if mode == "Future Forecast":
                                combined_df, errors = run_future_forecasts(
                                    y_series,
                                    freq_alias,
                                    int(st.session_state["horizon"]),
                                    selected_model_keys,
                                    include_prophet_holidays,
                                    progress_callback=progress_cb,
                                )
                                metrics_df = None
                                backtest_index = None
                            else:
                                backtest_method = st.session_state["backtest_method"]
                                if backtest_method == "Date range":
                                    dates = st.session_state["backtest_dates"]
                                    if not dates or len(dates) != 2:
                                        raise ValueError(
                                            "Please select start and end date."
                                        )
                                    start_date, end_date = sorted(dates)
                                    y_train, y_test = split_backtest_date_range(
                                        y_series, start_date, end_date
                                    )
                                else:
                                    last_n = st.session_state["backtest_last_n"]
                                    y_train, y_test = split_backtest_last_n(
                                        y_series, int(last_n)
                                    )

                                combined_df, metrics_df, backtest_index, errors = (
                                    run_backtest_forecasts(
                                        y_series,
                                        freq_alias,
                                        y_train,
                                        y_test,
                                        selected_model_keys,
                                        include_prophet_holidays,
                                        progress_callback=progress_cb,
                                    )
                                )

                            st.session_state["combined_df"] = combined_df
                            st.session_state["metrics_df"] = metrics_df
                            st.session_state["backtest_index"] = backtest_index

                            for msg in errors:
                                st.warning(msg)

                            st.success("Forecasting finished. Check the next tab.")
                        except Exception as e:
                            st.error(f"Error while running forecasts: {e}")

    # ----------------------------------------------------------------
    # Tab 2: Forecast & Evaluation
    # ----------------------------------------------------------------
    with tabs[1]:
        st.header("2. Forecast & Evaluation")

        if "combined_df" not in st.session_state:
            st.info("Run forecasts in the 'Data & Settings' tab first.")
        else:
            combined_df = st.session_state["combined_df"]
            mode = st.session_state.get("mode", "Future Forecast")
            metrics_df = st.session_state.get("metrics_df", None)
            backtest_index = st.session_state.get("backtest_index", None)

            # Dynamic display window selection
            full_min_date = combined_df.index.min().date()
            full_max_date = combined_df.index.max().date()

            st.subheader("Display Window")
            display_range = st.date_input(
                "Select date range for display",
                value=[full_min_date, full_max_date],
                min_value=full_min_date,
                max_value=full_max_date,
            )

            if (
                isinstance(display_range, (list, tuple))
                and len(display_range) == 2
            ):
                view_start_date, view_end_date = sorted(display_range)
                view_start_ts = pd.Timestamp(view_start_date)
                view_end_ts = pd.Timestamp(view_end_date)
            else:
                view_start_ts = combined_df.index.min()
                view_end_ts = combined_df.index.max()

            # Filter combined_df to display window
            mask = (combined_df.index >= view_start_ts) & (
                combined_df.index <= view_end_ts
            )
            plot_df = combined_df.loc[mask].copy()

            st.subheader("Time Series Plot (interactive)")

            # Prepare long-form data for Altair
            idx_name = plot_df.index.name or "index"
            plot_df_reset = plot_df.reset_index().rename(
                columns={idx_name: "datetime"}
            )

            value_cols = [
                c for c in plot_df.columns
                if c == "actual" or c.startswith("forecast_")
            ]
            if not value_cols:
                st.warning("No series to plot.")
            else:
                long_df = plot_df_reset.melt(
                    id_vars="datetime",
                    value_vars=value_cols,
                    var_name="series_raw",
                    value_name="value",
                )
                # Drop NaN values (e.g. forecasts outside backtest window)
                long_df = long_df.dropna(subset=["value"])

                # Map raw series names to display labels
                def label_from_raw(raw: str) -> str:
                    if raw == "actual":
                        return "Actual"
                    if raw.startswith("forecast_"):
                        key = raw.replace("forecast_", "")
                        return MODEL_DISPLAY_NAMES.get(key, raw)
                    return raw

                long_df["series"] = long_df["series_raw"].apply(label_from_raw)
                long_df["is_actual"] = long_df["series"] == "Actual"

                legend_ncol = 4 if (
                    mode == "Backtest (Within Data Range)" and backtest_index is not None
                ) else 1

                base = alt.Chart(long_df).mark_line().encode(
                    x=alt.X("datetime:T", title="Time"),
                    y=alt.Y("value:Q", title="Value"),
                    color=alt.Color(
                        "series:N",
                        title=None,
                        legend=alt.Legend(
                            orient="bottom",          # legend BELOW chart
                            direction="horizontal",
                            columns=legend_ncol,
                        ),
                    ),
                    strokeDash=alt.condition(
                        alt.datum.series == "Actual",
                        alt.value([1, 0]),          # solid
                        alt.value([4, 4]),          # dotted
                    ),
                    tooltip=[
                        alt.Tooltip("datetime:T", title="Datetime"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("value:Q", title="Value"),
                    ],
                )

                chart = base

                # Backtest shading (only in intersection with display window)
                if mode == "Backtest (Within Data Range)" and backtest_index is not None:
                    bt_start = backtest_index.min()
                    bt_end = backtest_index.max()

                    shade_start = max(bt_start, view_start_ts)
                    shade_end = min(bt_end, view_end_ts)

                    if shade_start < shade_end:
                        shade_df = pd.DataFrame(
                            {"start": [shade_start], "end": [shade_end]}
                        )
                        shade_chart = (
                            alt.Chart(shade_df)
                            .mark_rect(opacity=0.08)
                            .encode(
                                x="start:T",
                                x2="end:T",
                            )
                        )
                        chart = shade_chart + chart

                    title_text = (
                        "Backtest: Actual vs Forecasts (shaded area = backtest window)"
                    )
                else:
                    title_text = "Future Forecast: Actual vs Model Forecasts"

                chart = chart.properties(
                    width="container",
                    height=400,
                    title=title_text,
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

            # --- Metrics for backtest mode ---
            if mode == "Backtest (Within Data Range)" and metrics_df is not None:
                st.subheader("Backtest Metrics")
                st.caption(
                    "ME = Mean Error (signed), MAE = Mean Absolute Error, "
                    "RMSE = Root Mean Squared Error, MAPE = Mean Absolute Percentage Error."
                )
                st.dataframe(metrics_df)

            # --- Download combined CSV ---
            st.subheader("Download Forecasts")
            csv_data = (
                combined_df.reset_index()
                .rename(columns={combined_df.index.name or "index": "datetime"})
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "Download forecasts as CSV",
                data=csv_data,
                file_name="forecasts.csv",
                mime="text/csv",
            )

    # ----------------------------------------------------------------
    # Tab 3: Monte Carlo Simulation
    # ----------------------------------------------------------------
    with tabs[2]:
        st.header("3. Monte Carlo Simulation")

        if "y_series" not in st.session_state or "freq_alias" not in st.session_state:
            st.info("Upload and configure data in 'Data & Settings' first.")
        else:
            y_series = st.session_state["y_series"]
            freq_alias = st.session_state["freq_alias"]
            include_prophet_holidays = st.session_state.get(
                "include_prophet_holidays", False
            )

            min_date = y_series.index.min().date()
            max_date = y_series.index.max().date()

            model_key_mc = st.selectbox(
                "Model for Monte Carlo",
                options=list(MODEL_REGISTRY.keys()),
                format_func=lambda k: MODEL_DISPLAY_NAMES[k],
            )

            mc_dates = st.date_input(
                "Backtest window for Monte Carlo (start and end dates)",
                value=[
                    max(min_date, max_date - dt.timedelta(days=30)),
                    max_date,
                ],
            )

            if not isinstance(mc_dates, (list, tuple)) or len(mc_dates) != 2:
                st.warning("Please select both start and end dates for Monte Carlo.")
            else:
                start_date_mc, end_date_mc = sorted(mc_dates)

                n_sims = st.number_input(
                    "Number of simulations",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100,
                )

                metric_choice_label = st.radio(
                    "Metric to simulate",
                    options=["Mean Error (signed)", "Mean Absolute Error (MAE)"],
                )
                metric_type = "ME" if "signed" in metric_choice_label else "MAE"

                run_mc_button = st.button("Run Monte Carlo Simulation")

                if run_mc_button:
                    try:
                        prog = st.progress(0.0)

                        def mc_cb(frac: float):
                            prog.progress(frac)

                        metrics, observed_metric = run_monte_carlo_simulation(
                            y_series,
                            freq_alias,
                            model_key_mc,
                            start_date_mc,
                            end_date_mc,
                            include_prophet_holidays,
                            int(n_sims),
                            metric_type,
                            progress_callback=mc_cb,
                        )

                        st.subheader("Simulation Results")

                        # Histogram with ~50 bins
                        fig, ax = plt.subplots()
                        ax.hist(metrics, bins=50)
                        ax.axvline(
                            observed_metric,
                            linestyle="--",
                            linewidth=2,
                            label="Observed metric",
                        )
                        ax.set_xlabel(
                            "Simulated "
                            + ("Mean Error" if metric_type == "ME" else "MAE")
                        )
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)

                        mean_val = float(np.mean(metrics))
                        median_val = float(np.median(metrics))
                        p5, p50, p95 = np.percentile(metrics, [5, 50, 95])

                        st.write(
                            f"Observed metric: {observed_metric:.4f} "
                            f"({metric_type})"
                        )
                        st.write(f"Simulation mean: {mean_val:.4f}")
                        st.write(f"Simulation median: {median_val:.4f}")
                        st.write(
                            f"5th / 50th / 95th percentiles: "
                            f"{p5:.4f}, {p50:.4f}, {p95:.4f}"
                        )

                    except Exception as e:
                        st.error(f"Error during Monte Carlo simulation: {e}")


if __name__ == "__main__":
    main()
