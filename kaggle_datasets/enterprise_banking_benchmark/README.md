# Enterprise Multi-Domain Banking, Fraud & AML Benchmark

## Overview & Context
This synthetic star-schema benchmark supports **time-series treasury forecasting**, **econometric peer branch benchmarking**, **Customer 360 RFM segmentation**, **AML/account-takeover anomaly triage**, and **Power BI / SQL analytics**. Generated from the deterministic analytical pipeline in [`enterprise-banking-ds-platform`](https://github.com/akmalkhaniub/enterprise-banking-ds-platform), the corpus models **30 network branches** across 10 cities, **1,200 customer accounts**, **210 days of ATM cash demand and deposit liquidity** (150 historical, 30 validation backtest, and 30 forward forecast days), **679 prioritized anomaly queue alerts** isolated from 45,000+ underlying simulated multi-channel transactions, and **4 executive ALCO/compliance action briefs**.

---

## Files Included (5 Star-Schema CSV Tables)

### 1. `DimBranchBenchmark.csv` (30 rows × 30 columns)
Branch-level operational inputs, 180-day deposit/ATM volumes, OLS expected net liquidity, residual efficiency index, and benchmark status (`OUTPERFORMING_BENCHMARK`: 7, `ON_TARGET_PEER_BAND`: 13, `UNDERPERFORMING_ACTION_NEEDED`: 10) across 10 Pakistani cities.
- **Keys & Dimensions:** `branch_id`, `branch_name`, `city`, `province`, `branch_tier`, `digital_onboarding_campaign`
- **Capacity & Book:** `staff_count`, `atm_count`, `catchment_population`, `monthly_opex_pkr`, `active_customer_accounts`, `total_casa_book_pkr`, `avg_casa_per_customer_pkr`, `avg_products_per_customer`, `digital_enrollment_rate`
- **180-Day Flows & OLS Benchmarks:** `total_deposit_inflow_180d_pkr`, `total_deposit_outflow_180d_pkr`, `total_net_liquidity_180d_pkr`, `avg_daily_net_liquidity_pkr`, `total_atm_demand_180d_pkr`, `avg_daily_atm_demand_pkr`, `total_digital_txns_180d`, `cost_to_net_deposit_ratio`, `net_deposit_per_staff_pkr`, `expected_net_liquidity_180d_pkr`, `value_add_residual_pkr`, `efficiency_index_pct`, `residual_zscore`, `benchmark_status`, `efficiency_rank`

### 2. `DimCustomerSegment.csv` (1,200 rows × 27 columns)
Customer 360 account profile, 180-day RFM behavioral features, unsupervised K-Means/GMM cluster assignments, estimated annual CLV, and prescriptive CRM actions.
- **Profile:** `customer_id`, `branch_id`, `account_type`, `latent_archetype`, `tenure_months`, `declared_monthly_income_pkr`, `current_casa_balance_pkr`, `products_count`, `internal_credit_score`, `mobile_app_enrolled`, `kyc_risk_tier`
- **180-Day RFM & Behavior:** `recency_days`, `txn_frequency_180d`, `monetary_total_volume_pkr`, `avg_ticket_pkr`, `std_ticket_pkr`, `max_ticket_pkr`, `digital_txn_share`, `odd_hour_txn_share`, `total_credit_inflow_pkr`, `total_debit_outflow_pkr`, `volume_to_6m_income_ratio`
- **Segmentation & CRM:** `cluster_id`, `cluster_assignment_confidence`, `customer_segment`, `recommended_next_best_action`, `estimated_annual_clv_pkr`

### 3. `FactTransactionAnomaly.csv` (679 rows × 25 columns)
High-risk flagged alert queue (`model_anomaly_flag == 1`) isolated from the full 45,000+ transaction ledger (`642` true injected anomalies across `INCOME_DISPROPORTIONATE_SPIKE` (242), `MIDNIGHT_HIGH_VALUE_DRAIN` (209), and `AML_THRESHOLD_STRUCTURING` (191), plus `37` `NORMAL` false positives for precision/triage analysis).
- **Event & Context:** `transaction_id`, `customer_id`, `branch_id`, `txn_timestamp`, `txn_date`, `txn_hour`, `channel`, `direction`, `amount_pkr`
- **Customer Baselines & Risk Rules:** `declared_monthly_income_pkr`, `current_casa_balance_pkr`, `kyc_risk_tier`, `cust_mean_amt`, `cust_std_amt`, `customer_amount_zscore`, `amount_to_monthly_income_ratio`, `is_odd_hour`, `is_near_aml_reporting_threshold`, `customer_same_day_txn_count`, `customer_same_day_volume_pkr`
- **Labels & Model Outputs:** `ground_truth_anomaly`, `anomaly_pattern_code`, `anomaly_risk_score`, `model_anomaly_flag`, `explainable_alert_reason`

### 4. `FactLiquidityForecast.csv` (210 rows × 10 columns)
Daily network-wide ATM cash demand and deposit inflow series spanning `2026-05-01` through `2026-11-26` (`150` `HISTORICAL_ACTUAL`, `30` `VALIDATION_BACKTEST`, `30` `FUTURE_FORECAST_30D`), complete with 95% confidence bounds and calendar flags (`is_salary_window`, `is_weekend`).
- **Columns:** `date`, `period_type`, `actual_atm_cash_demand_pkr`, `forecast_atm_cash_demand_pkr`, `forecast_atm_lower_95ci_pkr`, `forecast_atm_upper_95ci_pkr`, `actual_deposit_inflow_pkr`, `forecast_deposit_inflow_pkr`, `is_salary_window`, `is_weekend`

### 5. `FactExecutiveRecommendation.csv` (4 rows × 7 columns)
Schema-validated ALCO, branch network, CRM, and AML compliance action recommendations grounded in quantitative evidence.
- **Columns:** `recommendation_id`, `domain`, `priority`, `headline`, `quantified_evidence`, `recommended_action`, `expected_business_impact`

---

## Use Cases & Source
- **AML / False-Positive Alert Triage:** Benchmark precision/ranking rules differentiating the 642 true anomalies from the 37 benign `NORMAL` alerts in `FactTransactionAnomaly.csv`.
- **Treasury Cash Replenishment:** Evaluate salary-window surge accuracy across the 30-day `VALIDATION_BACKTEST` horizon in `FactLiquidityForecast.csv`.
- **Econometric Branch Efficiency:** Audit `value_add_residual_pkr` and `efficiency_index_pct` across capacity covariates in `DimBranchBenchmark.csv`.
- **Source Repository:** https://github.com/akmalkhaniub/enterprise-banking-ds-platform
- **License:** CC0-1.0 (Public Domain — synthetic benchmark data).
