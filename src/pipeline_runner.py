"""End-to-End Banking Data Science Pipeline Orchestrator & Power BI / Tableau Star-Schema Mart Exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data_generator import generate_banking_dataset, save_raw_datasets
from src.forecasting_and_benchmarking import (
    benchmark_branches_regression,
    build_time_series_forecasts,
    run_stationarity_and_hypothesis_tests,
)
from src.genai_narrator import build_controlled_executive_brief
from src.segmentation_and_anomaly import (
    detect_transaction_anomalies,
    segment_customers_rfm_clustering,
)
from src.sql_pipeline import BankingSQLPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MART_DIR = PROJECT_ROOT / "bi" / "mart_exports"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"


def run_full_banking_ds_pipeline(
    seed: int = 42,
    n_branches: int = 30,
    n_customers: int = 1200,
    n_days: int = 180,
    n_transactions: int = 45000,
    persist_to_disk: bool = True,
) -> Dict[str, Any]:
    """Execute the full Enterprise Banking Data Science pipeline from raw generation -> SQL Feature Store -> ML Models -> BI Mart."""
    # 1. Generate realistic banking datasets
    datasets = generate_banking_dataset(
        seed=seed,
        n_branches=n_branches,
        n_customers=n_customers,
        n_days=n_days,
        n_transactions=n_transactions,
    )
    if persist_to_disk:
        save_raw_datasets(datasets, RAW_DIR)

    # 2. Run DuckDB SQL Feature Engineering & Data Quality Suite
    sql_engine = BankingSQLPipeline()
    sql_engine.ingest_dataframes(datasets)
    dq_results = sql_engine.run_data_quality_suite()
    df_customer_360 = sql_engine.build_customer_360_rfm_features()
    df_txn_velocity = sql_engine.build_transaction_velocity_features()
    df_branch_summary = sql_engine.build_branch_performance_summary()

    # 3. Pillar 1: Stationarity, Hypothesis Testing, Time-Series Forecasting & Branch Benchmarking
    hypothesis_results = run_stationarity_and_hypothesis_tests(
        datasets["fact_branch_daily_liquidity"],
        df_branch_summary,
    )
    df_forecasts, forecast_metrics = build_time_series_forecasts(
        datasets["fact_branch_daily_liquidity"],
        forecast_horizon_days=30,
    )
    df_branch_bench, branch_ols_summary = benchmark_branches_regression(df_branch_summary)

    # 4. Pillar 2: Customer 360 Clustering & Transaction Anomaly Detection
    df_segmented_customers, segmentation_summary = segment_customers_rfm_clustering(
        df_customer_360,
        n_clusters=5,
    )
    df_scored_txns, anomaly_metrics = detect_transaction_anomalies(
        df_txn_velocity,
        contamination=0.018,
    )

    # 5. Pillar 3: Controlled Enterprise GenAI Executive Narrator
    fut_rows = df_forecasts[df_forecasts["period_type"] == "FUTURE_FORECAST_30D"]
    future_summary = {
        "next_30d_total_atm_demand_pkr": float(fut_rows["forecast_atm_cash_demand_pkr"].sum()),
        "next_30d_total_deposit_inflow_pkr": float(fut_rows["forecast_deposit_inflow_pkr"].sum()),
        "peak_salary_day_atm_demand_pkr": float(fut_rows["forecast_atm_cash_demand_pkr"].max()),
    }
    underperforming_branches = df_branch_bench[
        df_branch_bench["benchmark_status"] == "UNDERPERFORMING_ACTION_NEEDED"
    ].sort_values("efficiency_index_pct").to_dict("records")

    executive_brief = build_controlled_executive_brief(
        dq_results=dq_results,
        hypothesis_results=hypothesis_results,
        forecast_metrics=forecast_metrics,
        df_forecasts_future_summary=future_summary,
        branch_ols_summary=branch_ols_summary,
        underperforming_branches=underperforming_branches,
        segmentation_summary=segmentation_summary,
        anomaly_metrics=anomaly_metrics,
    )

    # 6. Export Star-Schema Power BI / Tableau Data Mart CSVs & JSON Snapshot
    df_flagged_alerts = (
        df_scored_txns[df_scored_txns["model_anomaly_flag"] == 1]
        .sort_values("anomaly_risk_score", ascending=False)
        .reset_index(drop=True)
    )

    if persist_to_disk:
        MART_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        df_branch_bench.to_csv(MART_DIR / "DimBranchBenchmark.csv", index=False)
        df_segmented_customers.to_csv(MART_DIR / "DimCustomerSegment.csv", index=False)
        df_forecasts.to_csv(MART_DIR / "FactLiquidityForecast.csv", index=False)
        df_flagged_alerts.to_csv(MART_DIR / "FactTransactionAnomaly.csv", index=False)
        pd.DataFrame(executive_brief["recommendations"]).to_csv(
            MART_DIR / "FactExecutiveRecommendation.csv", index=False
        )

    bundle = {
        "kpis": {
            "total_branches": int(len(df_branch_bench)),
            "total_customers": int(len(df_segmented_customers)),
            "total_transactions_analyzed": int(len(df_scored_txns)),
            "total_casa_book_pkr": round(float(df_segmented_customers["current_casa_balance_pkr"].sum()), 2),
            "next_30d_atm_forecast_pkr": round(future_summary["next_30d_total_atm_demand_pkr"], 2),
            "next_30d_deposit_inflow_forecast_pkr": round(future_summary["next_30d_total_deposit_inflow_pkr"], 2),
            "forecast_ensemble_mape_pct": forecast_metrics["models"]["Hybrid_SARIMAX_GBR_Ensemble"]["mape_pct"],
            "clustering_silhouette_score": segmentation_summary["silhouette_score"],
            "anomaly_detection_f1": anomaly_metrics["benchmark_evaluation_vs_injected_fraud"]["f1_score"],
            "anomaly_detection_roc_auc": anomaly_metrics["benchmark_evaluation_vs_injected_fraud"]["roc_auc"],
            "flagged_anomalies_count": anomaly_metrics["total_anomalies_flagged"],
            "exposed_anomaly_volume_pkr": anomaly_metrics["total_exposed_anomaly_volume_pkr"],
        },
        "data_quality_checks": dq_results,
        "hypothesis_testing": hypothesis_results,
        "forecasting": {
            "metrics": forecast_metrics,
            "series": df_forecasts.tail(75).to_dict("records"),
        },
        "branch_benchmarking": {
            "ols_summary": branch_ols_summary,
            "branches": df_branch_bench.to_dict("records"),
        },
        "customer_segmentation": segmentation_summary,
        "anomaly_detection": {
            "metrics": anomaly_metrics,
            "top_alerts": df_flagged_alerts.head(40).to_dict("records"),
        },
        "executive_brief": executive_brief,
    }

    if persist_to_disk:
        with open(ARTIFACTS_DIR / "pipeline_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

    return bundle


if __name__ == "__main__":
    result = run_full_banking_ds_pipeline(persist_to_disk=True)
    print("Pipeline execution complete!")
    print("KPI Summary:", json.dumps(result["kpis"], indent=2))
