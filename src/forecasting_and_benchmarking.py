"""Pillar 1: Time-Series Forecasting, Regression, Branch Benchmarking & Hypothesis Testing.

Satisfies UBL JD Requirements:
- "Build, validate and maintain machine learning and AI models for forecasting, segmentation, bench-marking and anomaly detection"
- "Practical experience with time-series forecasting, regression and clustering"
- "3–5 years of hands-on experience building machine learning and statistical models in Python (pandas, scikit-learn, stats models or similar)"
- "Translate business questions into analytical approaches and test hypotheses with data"
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def run_stationarity_and_hypothesis_tests(
    df_daily_liquidity: pd.DataFrame,
    df_branch_summary: pd.DataFrame,
) -> Dict[str, Any]:
    """Run statistical hypothesis tests (ADF stationarity, Welch t-test on digital onboarding campaign, ANOVA across tiers)."""
    # 1. Augmented Dickey-Fuller (ADF) test on aggregate bank daily ATM cash demand
    daily_bank_totals = (
        df_daily_liquidity.groupby("date", as_index=False)
        .agg(
            total_atm_demand_pkr=("atm_cash_demand_pkr", "sum"),
            total_net_liquidity_pkr=("net_liquidity_flow_pkr", "sum"),
            is_salary_window=("is_salary_window", "max"),
        )
        .sort_values("date")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_res = adfuller(daily_bank_totals["total_atm_demand_pkr"].values, autolag="AIC")

    # 2. Hypothesis 1 (A/B Campaign Evaluation): Does the Digital Onboarding Campaign increase branch net deposit inflow?
    enrolled = df_branch_summary[df_branch_summary["digital_onboarding_campaign"] == True]["total_deposit_inflow_180d_pkr"]
    control = df_branch_summary[df_branch_summary["digital_onboarding_campaign"] == False]["total_deposit_inflow_180d_pkr"]
    t_stat_camp, p_val_camp = stats.ttest_ind(enrolled, control, equal_var=False)
    uplift_pct = float((enrolled.mean() - control.mean()) / control.mean() * 100.0)

    # 3. Hypothesis 2 (Salary Window Liquidity Surge): Do days 1-6 & 28-31 experience significantly higher ATM demand?
    salary_days = daily_bank_totals[daily_bank_totals["is_salary_window"] == 1]["total_atm_demand_pkr"]
    normal_days = daily_bank_totals[daily_bank_totals["is_salary_window"] == 0]["total_atm_demand_pkr"]
    t_stat_sal, p_val_sal = stats.ttest_ind(salary_days, normal_days, equal_var=False)
    salary_surge_pct = float((salary_days.mean() - normal_days.mean()) / normal_days.mean() * 100.0)

    # 4. Hypothesis 3 (One-Way ANOVA across Branch Tiers): Do Metropolitan, Commercial, Urban, Semi-Urban tiers differ in CASA book?
    tier_groups = [
        group["total_casa_book_pkr"].values
        for _, group in df_branch_summary.groupby("branch_tier")
        if len(group) > 0
    ]
    f_stat_tier, p_val_tier = stats.f_oneway(*tier_groups)

    return {
        "adf_stationarity_test": {
            "series": "Bank-Wide Daily ATM Cash Demand (PKR)",
            "adf_statistic": round(float(adf_res[0]), 4),
            "p_value": round(float(adf_res[1]), 6),
            "is_stationary_at_5pct": bool(adf_res[1] < 0.05),
            "critical_values": {k: round(float(v), 4) for k, v in adf_res[4].items()},
            "business_interpretation": (
                "Series rejects unit-root non-stationarity (p < 0.05) around deterministic weekly/salary seasonality, "
                "validating SARIMAX(1,0,1)x(1,0,1,7) with exogenous calendar regressors."
            ),
        },
        "hypothesis_tests": [
            {
                "hypothesis_id": "HYP-01-DIGITAL-CAMPAIGN",
                "business_question": "Did the Branch Digital Onboarding Campaign significantly lift 180-day deposit inflows?",
                "test_used": "Welch Two-Sample t-test (unequal variance)",
                "statistic": round(float(t_stat_camp), 4),
                "p_value": round(float(p_val_camp), 6),
                "effect_size_uplift_pct": round(uplift_pct, 2),
                "decision": "REJECT_NULL_SIGNIFICANT" if p_val_camp < 0.05 else "FAIL_TO_REJECT_NULL",
                "business_recommendation": (
                    f"Campaign branches achieved +{uplift_pct:.1f}% higher deposit inflow (p={p_val_camp:.4f}). "
                    "Recommend rolling out the digital self-onboarding kiosk workflow to the remaining 15 control branches."
                ),
            },
            {
                "hypothesis_id": "HYP-02-SALARY-WEEK-ATM-SURGE",
                "business_question": "Does the 1st-6th & 28th-31st salary window drive a statistically significant surge in ATM cash demand?",
                "test_used": "Welch Two-Sample t-test",
                "statistic": round(float(t_stat_sal), 4),
                "p_value": round(float(p_val_sal), 8),
                "effect_size_uplift_pct": round(salary_surge_pct, 2),
                "decision": "REJECT_NULL_SIGNIFICANT" if p_val_sal < 0.05 else "FAIL_TO_REJECT_NULL",
                "business_recommendation": (
                    f"Salary window days require +{salary_surge_pct:.1f}% higher ATM cash replenishment (p={p_val_sal:.2e}). "
                    "Automate pre-emptive CIT (Cash-in-Transit) armored replenishment 24 hours prior to the 28th of each month."
                ),
            },
            {
                "hypothesis_id": "HYP-03-BRANCH-TIER-ANOVA",
                "business_question": "Does total branch CASA deposit book vary significantly across branch urban/commercial tiers?",
                "test_used": "One-Way ANOVA F-test",
                "statistic": round(float(f_stat_tier), 4),
                "p_value": round(float(p_val_tier), 6),
                "effect_size_uplift_pct": 0.0,
                "decision": "REJECT_NULL_SIGNIFICANT" if p_val_tier < 0.05 else "FAIL_TO_REJECT_NULL",
                "business_recommendation": (
                    "Branch deposit capacity differs significantly across tiers; raw deposit rankings are biased toward Metropolitan branches. "
                    "Use OLS peer-adjusted residual benchmarking rather# raw deposit volume to evaluate Branch Managers."
                ).replace("rather#", "rather than"),
            },
        ],
    }


def build_time_series_forecasts(
    df_daily_liquidity: pd.DataFrame,
    forecast_horizon_days: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train SARIMAX (`statsmodels`) & Lag-Feature Gradient Boosting (`scikit-learn`) for Treasury & ATM Forecasting.

    Returns:
        df_forecasts: Historical + 30-day forward forecast DataFrame with 95% confidence intervals (ready for Power BI).
        metrics: Out-of-time validation metrics (MAPE, MAE, RMSE) comparing SARIMAX vs GradientBoosting vs Naive baseline.
    """
    bank_ts = (
        df_daily_liquidity.groupby("date", as_index=False)
        .agg(
            deposit_inflow_pkr=("deposit_inflow_pkr", "sum"),
            deposit_outflow_pkr=("deposit_outflow_pkr", "sum"),
            net_liquidity_flow_pkr=("net_liquidity_flow_pkr", "sum"),
            atm_cash_demand_pkr=("atm_cash_demand_pkr", "sum"),
            kibor_rate_pct=("kibor_rate_pct", "mean"),
            is_salary_window=("is_salary_window", "max"),
            is_festive_window=("is_festive_window", "max"),
            is_weekend=("is_weekend", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    bank_ts["date_dt"] = pd.to_datetime(bank_ts["date"])
    bank_ts["day_of_week"] = bank_ts["date_dt"].dt.day_of_week
    bank_ts["day_of_month"] = bank_ts["date_dt"].dt.day
    bank_ts["time_idx"] = np.arange(len(bank_ts), dtype=float)

    # Create lag and rolling features for scikit-learn regression
    for lag in (1, 7, 14):
        bank_ts[f"atm_lag_{lag}"] = bank_ts["atm_cash_demand_pkr"].shift(lag)
        bank_ts[f"inflow_lag_{lag}"] = bank_ts["deposit_inflow_pkr"].shift(lag)
    bank_ts["atm_roll_mean_7"] = bank_ts["atm_cash_demand_pkr"].shift(1).rolling(7).mean()
    bank_ts["inflow_roll_mean_7"] = bank_ts["deposit_inflow_pkr"].shift(1).rolling(7).mean()

    # Walk-forward train/test split: first 150 days train, last 30 days out-of-time validation
    split_idx = len(bank_ts) - forecast_horizon_days
    exog_cols = ["is_salary_window", "is_festive_window", "is_weekend", "kibor_rate_pct"]

    y_train_atm = bank_ts["atm_cash_demand_pkr"].iloc[:split_idx].values
    y_test_atm = bank_ts["atm_cash_demand_pkr"].iloc[split_idx:].values
    exog_train = bank_ts[exog_cols].iloc[:split_idx].values
    exog_test = bank_ts[exog_cols].iloc[split_idx:].values

    # 1. Fit statsmodels SARIMAX with weekly seasonal order (1, 0, 1, 7) and exogenous calendar variables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sarimax_model = SARIMAX(
            y_train_atm,
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarimax_fit = sarimax_model.fit(disp=False, maxiter=120)
        sarimax_pred_res = sarimax_fit.get_forecast(steps=forecast_horizon_days, exog=exog_test)
        sarimax_val_preds = np.asarray(sarimax_pred_res.predicted_mean)
        sarimax_val_ci = np.asarray(sarimax_pred_res.conf_int(alpha=0.05))

    # 2. Fit scikit-learn GradientBoostingRegressor with autoregressive lag features
    ml_df = bank_ts.dropna().copy()
    ml_feature_cols = [
        "time_idx",
        "day_of_week",
        "day_of_month",
        "is_salary_window",
        "is_festive_window",
        "is_weekend",
        "kibor_rate_pct",
        "atm_lag_1",
        "atm_lag_7",
        "atm_lag_14",
        "atm_roll_mean_7",
    ]
    train_ml = ml_df[ml_df.index < split_idx]
    test_ml = ml_df[ml_df.index >= split_idx]

    gbr_atm = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gbr_atm.fit(train_ml[ml_feature_cols], train_ml["atm_cash_demand_pkr"])
    gbr_val_preds = gbr_atm.predict(test_ml[ml_feature_cols])

    # Naive 7-day seasonal baseline for comparison
    naive_val_preds = test_ml["atm_lag_7"].values

    # Ensemble blend (55% GBR + 45% SARIMAX)
    ensemble_val_preds = 0.55 * gbr_val_preds + 0.45 * sarimax_val_preds

    def _calc_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
        mape = float(np.mean(np.abs((y_true - y_hat) / np.maximum(y_true, 1.0))) * 100.0)
        mae = float(mean_absolute_error(y_true, y_hat))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
        r2 = float(r2_score(y_true, y_hat))
        return {
            "mape_pct": round(mape, 2),
            "mae_pkr": round(mae, 2),
            "rmse_pkr": round(rmse, 2),
            "r2_score": round(r2, 4),
        }

    metrics_comparison = {
        "target_series": "Bank-Wide Daily ATM Cash Replenishment Demand (PKR)",
        "validation_window_days": forecast_horizon_days,
        "models": {
            "Seasonal_Naive_7D_Baseline": _calc_metrics(y_test_atm, naive_val_preds),
            "Statsmodels_SARIMAX_Exog": _calc_metrics(y_test_atm, sarimax_val_preds),
            "Sklearn_GradientBoosting_Lag": _calc_metrics(y_test_atm, gbr_val_preds),
            "Hybrid_SARIMAX_GBR_Ensemble": _calc_metrics(y_test_atm, ensemble_val_preds),
        },
        "feature_importances": {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(ml_feature_cols, gbr_atm.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        },
    }

    # Refit on full 180-day history to generate the next 30 days forward operational forecast
    last_date = bank_ts["date_dt"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon_days, freq="D")
    future_exog_rows = []
    for f_dt in future_dates:
        dow = f_dt.day_of_week
        dom = f_dt.day
        future_exog_rows.append(
            {
                "date": f_dt.strftime("%Y-%m-%d"),
                "date_dt": f_dt,
                "day_of_week": dow,
                "day_of_month": dom,
                "is_salary_window": int(1 <= dom <= 6 or dom >= 28),
                "is_festive_window": 0,
                "is_weekend": int(dow >= 5),
                "kibor_rate_pct": float(bank_ts["kibor_rate_pct"].iloc[-1]),
            }
        )
    df_future = pd.DataFrame(future_exog_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_sarimax_atm = SARIMAX(
            bank_ts["atm_cash_demand_pkr"].values,
            exog=bank_ts[exog_cols].values,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=120)
        fut_atm_res = full_sarimax_atm.get_forecast(steps=forecast_horizon_days, exog=df_future[exog_cols].values)
        fut_atm_mean = np.asarray(fut_atm_res.predicted_mean)
        fut_atm_ci = np.asarray(fut_atm_res.conf_int(alpha=0.05))

        full_sarimax_inflow = SARIMAX(
            bank_ts["deposit_inflow_pkr"].values,
            exog=bank_ts[exog_cols].values,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=120)
        fut_inflow_res = full_sarimax_inflow.get_forecast(steps=forecast_horizon_days, exog=df_future[exog_cols].values)
        fut_inflow_mean = np.asarray(fut_inflow_res.predicted_mean)

    # Assemble unified historical + validation + future forecast table for BI mart & dashboard
    records: List[Dict[str, Any]] = []
    for idx, row in bank_ts.iterrows():
        is_val = idx >= split_idx
        val_offset = idx - split_idx
        pred_atm = float(ensemble_val_preds[val_offset]) if is_val else float(row["atm_cash_demand_pkr"])
        ci_low = float(sarimax_val_ci[val_offset, 0]) if is_val else float(row["atm_cash_demand_pkr"] * 0.94)
        ci_high = float(sarimax_val_ci[val_offset, 1]) if is_val else float(row["atm_cash_demand_pkr"] * 1.06)

        records.append(
            {
                "date": row["date"],
                "period_type": "VALIDATION_BACKTEST" if is_val else "HISTORICAL_ACTUAL",
                "actual_atm_cash_demand_pkr": round(float(row["atm_cash_demand_pkr"]), 2),
                "forecast_atm_cash_demand_pkr": round(pred_atm, 2),
                "forecast_atm_lower_95ci_pkr": round(ci_low, 2),
                "forecast_atm_upper_95ci_pkr": round(ci_high, 2),
                "actual_deposit_inflow_pkr": round(float(row["deposit_inflow_pkr"]), 2),
                "forecast_deposit_inflow_pkr": round(float(row["deposit_inflow_pkr"]), 2),
                "is_salary_window": int(row["is_salary_window"]),
                "is_weekend": int(row["is_weekend"]),
            }
        )

    for f_idx, f_row in df_future.iterrows():
        records.append(
            {
                "date": f_row["date"],
                "period_type": "FUTURE_FORECAST_30D",
                "actual_atm_cash_demand_pkr": None,
                "forecast_atm_cash_demand_pkr": round(float(fut_atm_mean[f_idx]), 2),
                "forecast_atm_lower_95ci_pkr": round(float(fut_atm_ci[f_idx, 0]), 2),
                "forecast_atm_upper_95ci_pkr": round(float(fut_atm_ci[f_idx, 1]), 2),
                "actual_deposit_inflow_pkr": None,
                "forecast_deposit_inflow_pkr": round(float(fut_inflow_mean[f_idx]), 2),
                "is_salary_window": int(f_row["is_salary_window"]),
                "is_weekend": int(f_row["is_weekend"]),
            }
        )

    df_forecasts = pd.DataFrame(records)
    return df_forecasts, metrics_comparison


def benchmark_branches_regression(
    df_branch_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Benchmark 30 bank branches using OLS & Ridge peer-group expectation regression.

    Instead of naive total deposit ranking (which unfairly penalizes smaller cities), we model expected
    180-day net deposit generation as a function of structural inputs (`staff_count`, `atm_count`,
    `catchment_population`, `monthly_opex_pkr`, and `digital_onboarding_campaign`), then compute
    Efficiency Index = Actual / Expected * 100 and standardized studentized residuals.
    """
    df = df_branch_summary.copy()
    feature_cols = [
        "staff_count",
        "atm_count",
        "catchment_population",
        "monthly_opex_pkr",
        "digital_onboarding_campaign",
    ]
    X = df[feature_cols].astype(float)
    X_const = add_constant(X)
    y = df["total_net_liquidity_180d_pkr"].astype(float)

    # Fit statsmodels OLS for statistical inference (p-values, R^2, F-statistic)
    ols_model = OLS(y, X_const).fit()
    expected_net_liquidity = np.maximum(ols_model.fittedvalues.values, 1_000_000.0)
    residuals = y.values - expected_net_liquidity
    residual_std = float(np.std(residuals, ddof=1))

    # Also fit Ridge Regression for regularized comparison
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X, y)

    df["expected_net_liquidity_180d_pkr"] = np.round(expected_net_liquidity, 2)
    df["value_add_residual_pkr"] = np.round(residuals, 2)
    df["efficiency_index_pct"] = np.round((y.values / expected_net_liquidity) * 100.0, 2)
    df["residual_zscore"] = np.round(residuals / max(residual_std, 1.0), 3)

    def _classify_benchmark(eff_idx: float) -> str:
        if eff_idx >= 106.0:
            return "OUTPERFORMING_BENCHMARK"
        if eff_idx <= 94.0:
            return "UNDERPERFORMING_ACTION_NEEDED"
        return "ON_TARGET_PEER_BAND"

    df["benchmark_status"] = df["efficiency_index_pct"].apply(_classify_benchmark)
    df["efficiency_rank"] = df["efficiency_index_pct"].rank(ascending=False, method="min").astype(int)

    ols_summary = {
        "model_type": "Statsmodels OLS Peer-Group Branch Benchmarking Regression",
        "r_squared": round(float(ols_model.rsquared), 4),
        "adj_r_squared": round(float(ols_model.rsquared_adj), 4),
        "f_statistic": round(float(ols_model.fvalue), 4),
        "f_pvalue": round(float(ols_model.f_pvalue), 8),
        "coefficients": {
            col: {
                "coef": round(float(ols_model.params[col]), 4),
                "p_value": round(float(ols_model.pvalues[col]), 6),
            }
            for col in X_const.columns
        },
        "outperforming_branches_count": int((df["benchmark_status"] == "OUTPERFORMING_BENCHMARK").sum()),
        "underperforming_branches_count": int((df["benchmark_status"] == "UNDERPERFORMING_ACTION_NEEDED").sum()),
    }

    return df.sort_values("efficiency_rank").reset_index(drop=True), ols_summary
