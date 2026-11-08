"""Comprehensive Pytest Verification Suite for UBL Banking Data Science & BI Platform."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.data_generator import generate_banking_dataset
from src.forecasting_and_benchmarking import (
    benchmark_branches_regression,
    build_time_series_forecasts,
    run_stationarity_and_hypothesis_tests,
)
from src.genai_narrator import sanitize_pii_text
from src.pipeline_runner import MART_DIR, run_full_banking_ds_pipeline
from src.segmentation_and_anomaly import (
    detect_transaction_anomalies,
    segment_customers_rfm_clustering,
)
from src.sql_pipeline import BankingSQLPipeline


def test_synthetic_banking_data_generator_invariants() -> None:
    datasets = generate_banking_dataset(
        seed=42,
        n_branches=15,
        n_customers=300,
        n_days=90,
        n_transactions=6000,
    )
    assert set(datasets.keys()) == {
        "dim_branch",
        "dim_customer",
        "fact_branch_daily_liquidity",
        "fact_transactions",
    }
    assert len(datasets["dim_branch"]) == 15
    assert len(datasets["dim_customer"]) == 300
    assert len(datasets["fact_branch_daily_liquidity"]) == 15 * 90
    assert len(datasets["fact_transactions"]) == 6000
    assert (datasets["fact_transactions"]["amount_pkr"] > 0).all()


def test_duckdb_sql_pipeline_and_data_quality_gates() -> None:
    datasets = generate_banking_dataset(
        seed=42,
        n_branches=15,
        n_customers=300,
        n_days=90,
        n_transactions=6000,
    )
    sql = BankingSQLPipeline()
    sql.ingest_dataframes(datasets)

    dq_checks = sql.run_data_quality_suite()
    assert len(dq_checks) == 4
    for check in dq_checks:
        assert check["status"] == "PASS"
        assert check["failed_records"] == 0

    df_c360 = sql.build_customer_360_rfm_features()
    assert len(df_c360) == 300
    assert "recency_days" in df_c360.columns
    assert "txn_frequency_180d" in df_c360.columns
    assert "monetary_total_volume_pkr" in df_c360.columns

    df_vel = sql.build_transaction_velocity_features()
    assert len(df_vel) == 6000
    assert "customer_amount_zscore" in df_vel.columns
    assert "is_near_aml_reporting_threshold" in df_vel.columns


def test_forecasting_hypothesis_testing_and_branch_benchmarking() -> None:
    datasets = generate_banking_dataset(
        seed=42,
        n_branches=20,
        n_customers=400,
        n_days=120,
        n_transactions=8000,
    )
    sql = BankingSQLPipeline()
    sql.ingest_dataframes(datasets)
    df_branch_summary = sql.build_branch_performance_summary()

    hyp = run_stationarity_and_hypothesis_tests(
        datasets["fact_branch_daily_liquidity"],
        df_branch_summary,
    )
    assert "adf_stationarity_test" in hyp
    assert len(hyp["hypothesis_tests"]) == 3
    assert hyp["hypothesis_tests"][1]["decision"] == "REJECT_NULL_SIGNIFICANT"

    df_fc, fc_metrics = build_time_series_forecasts(
        datasets["fact_branch_daily_liquidity"],
        forecast_horizon_days=20,
    )
    assert len(df_fc) == 120 + 20
    ensemble_mape = fc_metrics["models"]["Hybrid_SARIMAX_GBR_Ensemble"]["mape_pct"]
    naive_mape = fc_metrics["models"]["Seasonal_Naive_7D_Baseline"]["mape_pct"]
    assert ensemble_mape < naive_mape
    assert ensemble_mape < 8.0

    df_bench, ols_summary = benchmark_branches_regression(df_branch_summary)
    assert len(df_bench) == 20
    assert ols_summary["r_squared"] > 0.65
    assert "efficiency_index_pct" in df_bench.columns


def test_customer_segmentation_and_anomaly_detection() -> None:
    datasets = generate_banking_dataset(
        seed=42,
        n_branches=15,
        n_customers=400,
        n_days=120,
        n_transactions=10000,
    )
    sql = BankingSQLPipeline()
    sql.ingest_dataframes(datasets)
    df_c360 = sql.build_customer_360_rfm_features()
    df_vel = sql.build_transaction_velocity_features()

    df_seg, seg_summary = segment_customers_rfm_clustering(df_c360, n_clusters=5)
    assert len(df_seg) == 400
    assert seg_summary["silhouette_score"] > 0.20
    assert len(seg_summary["segment_profiles"]) == 5

    df_scored, anom_metrics = detect_transaction_anomalies(df_vel, contamination=0.018)
    assert len(df_scored) == 10000
    assert anom_metrics["benchmark_evaluation_vs_injected_fraud"]["f1_score"] > 0.80
    assert anom_metrics["benchmark_evaluation_vs_injected_fraud"]["roc_auc"] > 0.95


def test_pii_redaction_and_fastapi_endpoints() -> None:
    dirty = "Customer CUST-200145 with CNIC 42101-1234567-1 and IBAN PK36UNIL0000001123456789 wired funds."
    cleaned = sanitize_pii_text(dirty)
    assert "42101-1234567-1" not in cleaned
    assert "PK36UNIL0000001123456789" not in cleaned
    assert "CUST-200***" in cleaned

    client = TestClient(app)
    r_health = client.get("/healthz")
    assert r_health.status_code == 200

    r_bundle = client.get("/api/v1/dashboard-bundle")
    assert r_bundle.status_code == 200
    payload = r_bundle.json()
    assert payload["kpis"]["total_branches"] == 30
    assert payload["executive_brief"]["pii_sanitization_passed"] is True

    r_sim = client.post(
        "/api/v1/simulate/alco-stress-test",
        json={
            "kibor_delta_bps": 150,
            "salary_surge_multiplier": 1.25,
            "digital_campaign_rollout_pct": 90.0,
        },
    )
    assert r_sim.status_code == 200
    sim_data = r_sim.json()
    assert sim_data["stressed_30d_deposit_inflow_pkr"] > sim_data["baseline_30d_deposit_inflow_pkr"]
    assert sim_data["stressed_30d_atm_demand_pkr"] > sim_data["baseline_30d_atm_demand_pkr"]

    r_csv = client.get("/api/v1/bi-mart/download/DimBranchBenchmark")
    assert r_csv.status_code == 200
    assert "branch_id" in r_csv.text


def test_powerbi_pbip_and_star_schema_mart_files_exist() -> None:
    run_full_banking_ds_pipeline(persist_to_disk=True)
    expected_csvs = [
        "DimBranchBenchmark.csv",
        "DimCustomerSegment.csv",
        "FactLiquidityForecast.csv",
        "FactTransactionAnomaly.csv",
        "FactExecutiveRecommendation.csv",
    ]
    for fname in expected_csvs:
        assert (MART_DIR / fname).exists(), f"Missing BI mart export {fname}"

    pbip_file = Path(__file__).resolve().parents[1] / "bi" / "UBLBankingExecutiveIntelligence.pbip"
    assert pbip_file.exists()
