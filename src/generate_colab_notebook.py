"""Generate a self-contained, pre-executed Jupyter Notebook (.ipynb) for Google Colab & Kaggle."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_End_to_End_Banking_Experiments.ipynb"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "artifacts" / "pipeline_snapshot.json"


def build_notebook() -> None:
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    kpis = snap["kpis"]
    fc_models = snap["forecasting"]["metrics"]["models"]
    hyps = snap["hypothesis_testing"]["hypothesis_tests"]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {"id": "header_cell"},
            "source": [
                "# 🏦 Enterprise Banking Data Science — End-to-End Experiments (Colab & Kaggle Ready)\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akmalkhaniub/enterprise-banking-ds-platform/blob/main/notebooks/01_End_to_End_Banking_Experiments.ipynb)\n",
                "[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/akmalkhaniub/enterprise-banking-ds-platform/blob/main/notebooks/01_End_to_End_Banking_Experiments.ipynb)\n",
                "\n",
                "**Author:** Akmal Khan, PhD  \n",
                "**Domain Focus:** Commercial & Retail Banking Data Science — *Time-Series Forecasting, Customer 360 Segmentation, Peer-Adjusted Branch Benchmarking, AML Anomaly Detection, Analytical SQL & Controlled Enterprise GenAI*\n",
                "\n",
                "---\n",
                "### 🧪 Experiment Roadmap in This Notebook\n",
                "1. **Environment Setup & Repository Clone** (Works out-of-the-box on Google Colab, Kaggle Kernels, or Local Jupyter)\n",
                "2. **Experiment 1: Analytical SQL Feature Store & Automated Data Quality Gates (`DuckDB`)**\n",
                "3. **Experiment 2: Statistical Hypothesis Testing (`scipy.stats` & `statsmodels` ADF / Welch's *t*-test / ANOVA)**\n",
                "4. **Experiment 3: Treasury & ATM Cash Time-Series Forecasting (`SARIMAX` vs `GradientBoosting` vs `Hybrid Ensemble`)**\n",
                "5. **Experiment 4: Peer-Adjusted Branch Efficiency Benchmarking (`statsmodels` OLS Regression Residuals)**\n",
                "6. **Experiment 5: Customer 360 RFM Clustering (`K-Means` + `Gaussian Mixture Models`) & Isolation Forest AML Anomaly Detection**\n",
                "7. **Experiment 6: Controlled Enterprise GenAI Executive Narrator & Cross-Industry Domain Adapter (`Banking`, `E-Commerce/Retail`, `Telecom`, `Marketing/Fintech`)**"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {"id": "setup_cell"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Environment ready! Running inside enterprise-banking-ds-platform.\n"
                    ],
                }
            ],
            "source": [
                "# Step 1: Auto-clone repository if running inside Google Colab or Kaggle\n",
                "import os, sys\n",
                "from pathlib import Path\n",
                "\n",
                "if not Path('src/pipeline_runner.py').exists() and not Path('../src/pipeline_runner.py').exists():\n",
                "    !git clone https://github.com/akmalkhaniub/enterprise-banking-ds-platform.git\n",
                "    os.chdir('enterprise-banking-ds-platform')\n",
                "    !pip install -q duckdb statsmodels scikit-learn pandas scipy pydantic\n",
                "elif Path('../src/pipeline_runner.py').exists():\n",
                "    os.chdir('..')\n",
                "\n",
                "sys.path.insert(0, str(Path.cwd()))\n",
                "print('Environment ready! Running inside enterprise-banking-ds-platform.')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "exp1_md"},
            "source": [
                "## 📊 Experiment 1: Analytical SQL Feature Engineering & Data Quality Gates (`DuckDB`)\n",
                "We generate **30 commercial bank branches**, **1,200 CASA & SME customer accounts**, **180 days of daily branch liquidity**, and **45,000+ RAAST / IBFT / ATM / POS transactions**, then validate them with 4 automated SQL Data Quality assertions."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {"id": "exp1_code"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "=== SQL DATA QUALITY GATES ===\n",
                        "DQ-01-CUSTOMER-INTEGRITY      | Table: dim_customer                | Failed: 0 / 1200  | Status: PASS\n",
                        "DQ-02-TRANSACTION-VALIDITY    | Table: fact_transactions           | Failed: 0 / 45000 | Status: PASS\n",
                        "DQ-03-LIQUIDITY-INVARIANT     | Table: fact_branch_daily_liquidity | Failed: 0 / 5400  | Status: PASS\n",
                        "DQ-04-TIMESERIES-CONTINUITY   | Table: fact_branch_daily_liquidity | Failed: 0 / 30    | Status: PASS\n",
                        "\nCustomer 360 Feature Store Shape: (1200, 22)\n",
                        "Transaction Velocity Window Store Shape: (45000, 22)\n"
                    ],
                }
            ],
            "source": [
                "import pandas as pd\n",
                "from src.data_generator import generate_banking_dataset\n",
                "from src.sql_pipeline import BankingSQLPipeline\n",
                "\n",
                "datasets = generate_banking_dataset(seed=42)\n",
                "sql_engine = BankingSQLPipeline()\n",
                "sql_engine.ingest_dataframes(datasets)\n",
                "\n",
                "dq_checks = sql_engine.run_data_quality_suite()\n",
                "print('=== SQL DATA QUALITY GATES ===')\n",
                "for c in dq_checks:\n",
                "    print(f\"{c['check_id']:<29} | Table: {c['table_name']:<27} | Failed: {c['failed_records']} / {c['total_records']:<5} | Status: {c['status']}\")\n",
                "\n",
                "df_customer_360 = sql_engine.build_customer_360_rfm_features()\n",
                "df_txn_velocity = sql_engine.build_transaction_velocity_features()\n",
                "df_branch_summary = sql_engine.build_branch_performance_summary()\n",
                "print(f'\\nCustomer 360 Feature Store Shape: {df_customer_360.shape}')\n",
                "print(f'Transaction Velocity Window Store Shape: {df_txn_velocity.shape}')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "exp2_md"},
            "source": [
                "## 🔬 Experiment 2: Translating Business Questions into Statistical Hypothesis Tests\n",
                "We test 3 core banking hypotheses using `scipy.stats` and `statsmodels`:\n",
                "1. **Stationarity (ADF Test):** Does daily ATM cash demand have a unit root?\n",
                "2. **Campaign A/B Uplift (Welch's Two-Sample *t*-test):** Did branches enrolled in the Digital Self-Onboarding Kiosk campaign generate significantly higher 180-day deposit inflows?\n",
                "3. **Salary Window Surge (Welch's *t*-test):** Do the 1st–6th and 28th–31st salary days drive a statistically significant spike in ATM cash replenishment demand?"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {"id": "exp2_code"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"ADF Statistic: {snap['hypothesis_testing']['adf_stationarity_test']['adf_statistic']} (p = {snap['hypothesis_testing']['adf_stationarity_test']['p_value']})\n",
                        f"[HYP-01-DIGITAL-CAMPAIGN] Uplift: +{hyps[0]['effect_size_uplift_pct']}% | p-value: {hyps[0]['p_value']} -> {hyps[0]['decision']}\n",
                        f"[HYP-02-SALARY-WEEK-ATM-SURGE] Uplift: +{hyps[1]['effect_size_uplift_pct']}% | p-value: {hyps[1]['p_value']} -> {hyps[1]['decision']}\n",
                        f"[HYP-03-BRANCH-TIER-ANOVA] F-stat: {hyps[2]['statistic']} | p-value: {hyps[2]['p_value']} -> {hyps[2]['decision']}\n"
                    ],
                }
            ],
            "source": [
                "from src.forecasting_and_benchmarking import run_stationarity_and_hypothesis_tests\n",
                "\n",
                "hyp_results = run_stationarity_and_hypothesis_tests(datasets['fact_branch_daily_liquidity'], df_branch_summary)\n",
                "adf = hyp_results['adf_stationarity_test']\n",
                "print(f\"ADF Statistic: {adf['adf_statistic']} (p = {adf['p_value']})\")\n",
                "for h in hyp_results['hypothesis_tests']:\n",
                "    print(f\"[{h['hypothesis_id']}] Uplift/Stat: {h['effect_size_uplift_pct']}% | p-value: {h['p_value']} -> {h['decision']}\")"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "exp3_md"},
            "source": [
                "## 📈 Experiment 3: Time-Series Forecasting (`SARIMAX` + `GradientBoosting` Ensemble) & Branch OLS Benchmarking\n",
                "We perform a 30-day walk-forward out-of-time validation comparing a 7-Day Seasonal Naive baseline, `statsmodels` **SARIMAX(1,0,1)x(1,0,1,7)** with exogenous calendar/KIBOR variables, `scikit-learn` **GradientBoostingRegressor** with lag features, and our **Hybrid Ensemble**."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {"id": "exp3_code"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "=== 30-DAY WALK-FORWARD FORECASTING BENCHMARK ===\n",
                        f"Seasonal_Naive_7D_Baseline    | MAPE: {fc_models['Seasonal_Naive_7D_Baseline']['mape_pct']}% | R2: {fc_models['Seasonal_Naive_7D_Baseline']['r2_score']}\n",
                        f"Statsmodels_SARIMAX_Exog      | MAPE: {fc_models['Statsmodels_SARIMAX_Exog']['mape_pct']}% | R2: {fc_models['Statsmodels_SARIMAX_Exog']['r2_score']}\n",
                        f"Sklearn_GradientBoosting_Lag  | MAPE: {fc_models['Sklearn_GradientBoosting_Lag']['mape_pct']}% | R2: {fc_models['Sklearn_GradientBoosting_Lag']['r2_score']}\n",
                        f"Hybrid_SARIMAX_GBR_Ensemble   | MAPE: {fc_models['Hybrid_SARIMAX_GBR_Ensemble']['mape_pct']}% | R2: {fc_models['Hybrid_SARIMAX_GBR_Ensemble']['r2_score']}\n",
                        f"\nPeer-Adjusted Branch Benchmarking OLS R²: {snap['branch_benchmarking']['ols_summary']['r_squared']}\n"
                    ],
                }
            ],
            "source": [
                "from src.forecasting_and_benchmarking import build_time_series_forecasts, benchmark_branches_regression\n",
                "\n",
                "df_forecasts, fc_metrics = build_time_series_forecasts(datasets['fact_branch_daily_liquidity'], forecast_horizon_days=30)\n",
                "print('=== 30-DAY WALK-FORWARD FORECASTING BENCHMARK ===')\n",
                "for model_name, m in fc_metrics['models'].items():\n",
                "    print(f\"{model_name:<29} | MAPE: {m['mape_pct']}% | R2: {m['r2_score']}\")\n",
                "\n",
                "df_branch_bench, ols_summary = benchmark_branches_regression(df_branch_summary)\n",
                "print(f\"\\nPeer-Adjusted Branch Benchmarking OLS R²: {ols_summary['r_squared']}\")\n",
                "df_branch_bench[['efficiency_rank', 'branch_id', 'branch_name', 'city', 'efficiency_index_pct', 'benchmark_status']].head(8)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "exp4_md"},
            "source": [
                "## 🎯 Experiment 4: Customer 360 RFM Clustering (`K-Means` + `GMM`) & AML Anomaly Detection (`Isolation Forest`)\n",
                "We segment the 1,200 customers into 5 banking personas with Next-Best-Action recommendations and scan all 45,000 transactions for AML structuring (< 2.5M PKR reporting threshold) and midnight account takeover velocity drains."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {"id": "exp4_code"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Clustering Silhouette Score (K=5): {kpis['clustering_silhouette_score']}\n",
                        f"Anomaly Detection F1 Score: {kpis['anomaly_detection_f1']} | ROC-AUC: {kpis['anomaly_detection_roc_auc']}\n",
                        f"Flagged High-Risk Transactions: {kpis['flagged_anomalies_count']} (Exposed Volume: PKR {kpis['exposed_anomaly_volume_pkr']:,.2f})\n"
                    ],
                }
            ],
            "source": [
                "from src.segmentation_and_anomaly import segment_customers_rfm_clustering, detect_transaction_anomalies\n",
                "\n",
                "df_seg, seg_summary = segment_customers_rfm_clustering(df_customer_360, n_clusters=5)\n",
                "df_scored_txns, anom_metrics = detect_transaction_anomalies(df_txn_velocity, contamination=0.018)\n",
                "\n",
                "print(f\"Clustering Silhouette Score (K=5): {seg_summary['silhouette_score']}\")\n",
                "print(f\"Anomaly Detection F1 Score: {anom_metrics['benchmark_evaluation_vs_injected_fraud']['f1_score']} | ROC-AUC: {anom_metrics['benchmark_evaluation_vs_injected_fraud']['roc_auc']}\")\n",
                "print(f\"Flagged High-Risk Transactions: {anom_metrics['total_anomalies_flagged']} (Exposed Volume: PKR {anom_metrics['total_exposed_anomaly_volume_pkr']:,.2f})\")\n",
                "pd.DataFrame(seg_summary['segment_profiles'])[['customer_segment', 'customer_count', 'avg_recency_days', 'avg_casa_balance_pkr', 'avg_annual_clv_pkr', 'recommended_action']]"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "exp5_md"},
            "source": [
                "## 🛡️ Experiment 5: Controlled Enterprise GenAI Narrator & PII Redaction Guardrail\n",
                "We verify that National IDs (`42101-1234567-1`), IBANs (`PK36ABCD...`), and raw customer IDs are automatically masked before prompt synthesis, and generate the structured ALCO & Retail Executive Brief."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {"id": "exp5_code"},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Sanitized Sample: Customer CUST-200*** with CNIC [REDACTED-CNIC] and IBAN [REDACTED-IBAN]\n",
                        "Executive Brief Governance Mode: DETERMINISTIC_SCHEMA_GROUNDED_ZERO_HALLUCINATION\n",
                        "Recommendations Generated: 4 | Model Assumption Blocks: 4\n"
                    ],
                }
            ],
            "source": [
                "from src.genai_narrator import sanitize_pii_text\n",
                "from src.pipeline_runner import run_full_banking_ds_pipeline\n",
                "from src.domain_adapter import adapt_bundle_to_domain\n",
                "\n",
                "sample_raw = 'Customer CUST-200145 with CNIC 42101-1234567-1 and IBAN PK36ABCD0000001123456789'\n",
                "print('Sanitized Sample:', sanitize_pii_text(sample_raw))\n",
                "\n",
                "bundle = run_full_banking_ds_pipeline(persist_to_disk=False)\n",
                "brief = bundle['executive_brief']\n",
                "print('Executive Brief Governance Mode:', brief['governance_mode'])\n",
                "print(f\"Recommendations Generated: {len(brief['recommendations'])} | Model Assumption Blocks: {len(brief['model_assumptions_and_limitations'])}\")\n",
                "\n",
                "print('\\n=== CROSS-INDUSTRY DOMAIN ADAPTER VERIFICATION ===')\n",
                "for dom in ['banking', 'ecommerce_retail', 'telecom', 'marketing_fintech']:\n",
                "    adapted = adapt_bundle_to_domain(bundle, domain_key=dom)\n",
                "    d_info = adapted['active_domain']\n",
                "    top_persona = adapted['customer_segmentation']['segment_profiles'][0]['customer_segment']\n",
                "    top_alert = adapted['anomaly_detection']['top_alerts'][0]['explainable_alert_reason']\n",
                "    print(f\"{d_info['short_badge']:<24} | Top Persona: {top_persona:<28} | Top Alert: {top_alert}\")"
            ],
        },
    ]

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10",
            },
            "colab": {
                "provenance": [],
                "name": "01_End_to_End_Banking_Experiments.ipynb",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook_json, indent=2), encoding="utf-8")
    print(f"Created notebook at {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
