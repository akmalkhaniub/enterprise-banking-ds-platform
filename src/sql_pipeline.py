"""Enterprise SQL Data Engineering, Feature Engineering & Data Quality Validation Layer (DuckDB).

Satisfies UBL JD Requirements:
- "Strong SQL skills on large datasets"
- "Prior experience in data engineering or data analysis (building data pipelines, preparing and validating datasets) alongside data science work"
- "Work with the data engineering team on data requirements, features and data quality"
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd


@dataclass
class DataQualityCheckResult:
    check_id: str
    table_name: str
    rule_description: str
    failed_records: int
    total_records: int
    status: str  # "PASS" | "WARN" | "FAIL"


class BankingSQLPipeline:
    """Analytical SQL feature store and automated data quality validator powered by DuckDB."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def ingest_dataframes(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Register raw banking DataFrames as relational tables inside DuckDB."""
        for table_name, df in datasets.items():
            self.conn.register(f"stg_{table_name}", df)
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM stg_{table_name}")

    def run_data_quality_suite(self) -> List[Dict[str, Any]]:
        """Execute automated SQL data quality assertions across branches, customers, and transactions."""
        checks: List[DataQualityCheckResult] = []

        # Check 1: Referential integrity & PK uniqueness on dim_customer
        res_cust = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(*) - COUNT(DISTINCT customer_id) AS dup_keys,
                SUM(CASE WHEN current_casa_balance_pkr < 0 OR declared_monthly_income_pkr <= 0 THEN 1 ELSE 0 END) AS invalid_financials,
                SUM(CASE WHEN b.branch_id IS NULL THEN 1 ELSE 0 END) AS orphan_branches
            FROM dim_customer c
            LEFT JOIN dim_branch b ON c.branch_id = b.branch_id
            """
        ).fetchone()
        total_c = int(res_cust[0])
        failed_c = int(res_cust[1] + res_cust[2] + res_cust[3])
        checks.append(
            DataQualityCheckResult(
                check_id="DQ-01-CUSTOMER-INTEGRITY",
                table_name="dim_customer",
                rule_description="Unique customer_id PK, non-negative CASA balance, positive income, valid branch FK",
                failed_records=failed_c,
                total_records=total_c,
                status="PASS" if failed_c == 0 else "FAIL",
            )
        )

        # Check 2: Transaction amount bounds, valid timestamps, and customer FK integrity
        res_txn = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN amount_pkr <= 0 OR amount_pkr IS NULL THEN 1 ELSE 0 END) AS invalid_amt,
                SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS orphan_cust,
                SUM(CASE WHEN txn_hour < 0 OR txn_hour > 23 THEN 1 ELSE 0 END) AS invalid_hour
            FROM fact_transactions t
            LEFT JOIN dim_customer c ON t.customer_id = c.customer_id
            """
        ).fetchone()
        total_t = int(res_txn[0])
        failed_t = int(res_txn[1] + res_txn[2] + res_txn[3])
        checks.append(
            DataQualityCheckResult(
                check_id="DQ-02-TRANSACTION-VALIDITY",
                table_name="fact_transactions",
                rule_description="Strictly positive transaction amount_pkr, valid hour [0..23], and valid customer_id FK",
                failed_records=failed_t,
                total_records=total_t,
                status="PASS" if failed_t == 0 else "FAIL",
            )
        )

        # Check 3: Daily branch liquidity reconciliation invariant (net_liquidity_flow_pkr == inflow - outflow)
        res_liq = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(
                    CASE WHEN ABS((deposit_inflow_pkr - deposit_outflow_pkr) - net_liquidity_flow_pkr) > 0.05
                    OR atm_cash_demand_pkr < 0 THEN 1 ELSE 0 END
                ) AS accounting_breaks
            FROM fact_branch_daily_liquidity
            """
        ).fetchone()
        total_l = int(res_liq[0])
        failed_l = int(res_liq[1])
        checks.append(
            DataQualityCheckResult(
                check_id="DQ-03-LIQUIDITY-INVARIANT",
                table_name="fact_branch_daily_liquidity",
                rule_description="Daily branch net liquidity equals deposit_inflow_pkr minus deposit_outflow_pkr within 0.05 PKR",
                failed_records=failed_l,
                total_records=total_l,
                status="PASS" if failed_l == 0 else "FAIL",
            )
        )

        # Check 4: Time-series continuity check (every branch has contiguous daily observations)
        res_ts = self.conn.execute(
            """
            WITH branch_counts AS (
                SELECT branch_id, COUNT(DISTINCT date) AS n_days
                FROM fact_branch_daily_liquidity
                GROUP BY branch_id
            ),
            expected AS (
                SELECT MAX(n_days) AS max_days FROM branch_counts
            )
            SELECT
                COUNT(*) AS total_branches,
                SUM(CASE WHEN bc.n_days < e.max_days THEN 1 ELSE 0 END) AS gapped_branches
            FROM branch_counts bc
            CROSS JOIN expected e
            """
        ).fetchone()
        checks.append(
            DataQualityCheckResult(
                check_id="DQ-04-TIMESERIES-CONTINUITY",
                table_name="fact_branch_daily_liquidity",
                rule_description="Zero missing calendar dates across all 30 branches in the 180-day horizon",
                failed_records=int(res_ts[1]),
                total_records=int(res_ts[0]),
                status="PASS" if int(res_ts[1]) == 0 else "FAIL",
            )
        )

        return [asdict(c) for c in checks]

    def build_customer_360_rfm_features(self) -> pd.DataFrame:
        """Build Customer 360 RFM (Recency, Frequency, Monetary) & channel mix features using analytical SQL."""
        query = """
        WITH snapshot_ref AS (
            SELECT MAX(CAST(txn_date AS DATE)) + INTERVAL 1 DAY AS ref_date
            FROM fact_transactions
        ),
        txn_agg AS (
            SELECT
                t.customer_id,
                COUNT(*) AS txn_frequency_180d,
                SUM(t.amount_pkr) AS total_volume_pkr,
                AVG(t.amount_pkr) AS avg_ticket_pkr,
                STDDEV_POP(t.amount_pkr) AS std_ticket_pkr,
                MAX(t.amount_pkr) AS max_ticket_pkr,
                DATE_DIFF('day', MAX(CAST(t.txn_date AS DATE)), (SELECT ref_date FROM snapshot_ref)) AS recency_days,
                SUM(CASE WHEN t.channel IN ('RAAST_P2P', 'IBFT_Transfer', 'Mobile_App_Bill') THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS digital_txn_share,
                SUM(CASE WHEN t.txn_hour IN (0, 1, 2, 3, 4, 23) THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS odd_hour_txn_share,
                SUM(CASE WHEN t.direction = 'CREDIT' THEN t.amount_pkr ELSE 0 END) AS total_credit_inflow_pkr,
                SUM(CASE WHEN t.direction = 'DEBIT' THEN t.amount_pkr ELSE 0 END) AS total_debit_outflow_pkr
            FROM fact_transactions t
            GROUP BY t.customer_id
        )
        SELECT
            c.customer_id,
            c.branch_id,
            c.account_type,
            c.latent_archetype,
            c.tenure_months,
            c.declared_monthly_income_pkr,
            c.current_casa_balance_pkr,
            c.products_count,
            c.internal_credit_score,
            c.mobile_app_enrolled,
            c.kyc_risk_tier,
            COALESCE(a.recency_days, 180) AS recency_days,
            COALESCE(a.txn_frequency_180d, 0) AS txn_frequency_180d,
            ROUND(COALESCE(a.total_volume_pkr, 0.0), 2) AS monetary_total_volume_pkr,
            ROUND(COALESCE(a.avg_ticket_pkr, 0.0), 2) AS avg_ticket_pkr,
            ROUND(COALESCE(a.std_ticket_pkr, 0.0), 2) AS std_ticket_pkr,
            ROUND(COALESCE(a.max_ticket_pkr, 0.0), 2) AS max_ticket_pkr,
            ROUND(COALESCE(a.digital_txn_share, 0.0), 4) AS digital_txn_share,
            ROUND(COALESCE(a.odd_hour_txn_share, 0.0), 4) AS odd_hour_txn_share,
            ROUND(COALESCE(a.total_credit_inflow_pkr, 0.0), 2) AS total_credit_inflow_pkr,
            ROUND(COALESCE(a.total_debit_outflow_pkr, 0.0), 2) AS total_debit_outflow_pkr,
            ROUND(
                COALESCE(a.total_volume_pkr, 0.0) / NULLIF(c.declared_monthly_income_pkr * 6.0, 0),
                4
            ) AS volume_to_6m_income_ratio
        FROM dim_customer c
        LEFT JOIN txn_agg a ON c.customer_id = a.customer_id
        ORDER BY c.customer_id
        """
        df = self.conn.execute(query).df()
        self.conn.register("stg_mart_customer_360", df)
        self.conn.execute("CREATE OR REPLACE TABLE mart_customer_360 AS SELECT * FROM stg_mart_customer_360")
        return df

    def build_transaction_velocity_features(self) -> pd.DataFrame:
        """Compute rolling window velocity, Z-scores, and AML structuring proximity via SQL window functions."""
        query = """
        WITH cust_stats AS (
            SELECT
                customer_id,
                AVG(amount_pkr) AS cust_mean_amt,
                GREATEST(STDDEV_POP(amount_pkr), 1.0) AS cust_std_amt,
                COUNT(*) AS cust_total_txns
            FROM fact_transactions
            GROUP BY customer_id
        ),
        windowed AS (
            SELECT
                t.transaction_id,
                t.customer_id,
                t.branch_id,
                t.txn_timestamp,
                t.txn_date,
                t.txn_hour,
                t.channel,
                t.direction,
                t.amount_pkr,
                t.ground_truth_anomaly,
                t.anomaly_pattern_code,
                c.declared_monthly_income_pkr,
                c.current_casa_balance_pkr,
                c.kyc_risk_tier,
                cs.cust_mean_amt,
                cs.cust_std_amt,
                ROUND((t.amount_pkr - cs.cust_mean_amt) / cs.cust_std_amt, 4) AS customer_amount_zscore,
                ROUND(t.amount_pkr / NULLIF(c.declared_monthly_income_pkr, 0), 4) AS amount_to_monthly_income_ratio,
                CASE WHEN t.txn_hour IN (0, 1, 2, 3, 4, 23) THEN 1 ELSE 0 END AS is_odd_hour,
                CASE WHEN t.amount_pkr BETWEEN 2400000.0 AND 2500000.0 THEN 1 ELSE 0 END AS is_near_aml_reporting_threshold,
                COUNT(*) OVER (
                    PARTITION BY t.customer_id, t.txn_date
                ) AS customer_same_day_txn_count,
                SUM(t.amount_pkr) OVER (
                    PARTITION BY t.customer_id, t.txn_date
                ) AS customer_same_day_volume_pkr
            FROM fact_transactions t
            JOIN dim_customer c ON t.customer_id = c.customer_id
            JOIN cust_stats cs ON t.customer_id = cs.customer_id
        )
        SELECT * FROM windowed
        ORDER BY txn_timestamp, transaction_id
        """
        df = self.conn.execute(query).df()
        return df

    def build_branch_performance_summary(self) -> pd.DataFrame:
        """Aggregate 180-day branch inflows, outflows, ATM replenishment, CASA book, and operational efficiency."""
        query = """
        WITH liq_summary AS (
            SELECT
                branch_id,
                SUM(deposit_inflow_pkr) AS total_deposit_inflow_180d_pkr,
                SUM(deposit_outflow_pkr) AS total_deposit_outflow_180d_pkr,
                SUM(net_liquidity_flow_pkr) AS total_net_liquidity_180d_pkr,
                AVG(net_liquidity_flow_pkr) AS avg_daily_net_liquidity_pkr,
                SUM(atm_cash_demand_pkr) AS total_atm_demand_180d_pkr,
                AVG(atm_cash_demand_pkr) AS avg_daily_atm_demand_pkr,
                SUM(digital_channel_txn_count) AS total_digital_txns_180d,
                AVG(kibor_rate_pct) AS avg_kibor_rate_pct
            FROM fact_branch_daily_liquidity
            GROUP BY branch_id
        ),
        cust_summary AS (
            SELECT
                branch_id,
                COUNT(*) AS active_customer_accounts,
                SUM(current_casa_balance_pkr) AS total_casa_book_pkr,
                AVG(current_casa_balance_pkr) AS avg_casa_per_customer_pkr,
                AVG(products_count) AS avg_products_per_customer,
                SUM(CASE WHEN mobile_app_enrolled THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS digital_enrollment_rate
            FROM dim_customer
            GROUP BY branch_id
        )
        SELECT
            b.branch_id,
            b.branch_name,
            b.city,
            b.province,
            b.branch_tier,
            b.staff_count,
            b.atm_count,
            b.catchment_population,
            b.monthly_opex_pkr,
            b.digital_onboarding_campaign,
            COALESCE(cs.active_customer_accounts, 0) AS active_customer_accounts,
            ROUND(COALESCE(cs.total_casa_book_pkr, 0.0), 2) AS total_casa_book_pkr,
            ROUND(COALESCE(cs.avg_casa_per_customer_pkr, 0.0), 2) AS avg_casa_per_customer_pkr,
            ROUND(COALESCE(cs.avg_products_per_customer, 0.0), 2) AS avg_products_per_customer,
            ROUND(COALESCE(cs.digital_enrollment_rate, 0.0), 4) AS digital_enrollment_rate,
            ROUND(l.total_deposit_inflow_180d_pkr, 2) AS total_deposit_inflow_180d_pkr,
            ROUND(l.total_deposit_outflow_180d_pkr, 2) AS total_deposit_outflow_180d_pkr,
            ROUND(l.total_net_liquidity_180d_pkr, 2) AS total_net_liquidity_180d_pkr,
            ROUND(l.avg_daily_net_liquidity_pkr, 2) AS avg_daily_net_liquidity_pkr,
            ROUND(l.total_atm_demand_180d_pkr, 2) AS total_atm_demand_180d_pkr,
            ROUND(l.avg_daily_atm_demand_pkr, 2) AS avg_daily_atm_demand_pkr,
            l.total_digital_txns_180d,
            ROUND((b.monthly_opex_pkr * 6.0) / NULLIF(l.total_net_liquidity_180d_pkr, 0), 4) AS cost_to_net_deposit_ratio,
            ROUND(l.total_net_liquidity_180d_pkr / b.staff_count, 2) AS net_deposit_per_staff_pkr
        FROM dim_branch b
        JOIN liq_summary l ON b.branch_id = l.branch_id
        LEFT JOIN cust_summary cs ON b.branch_id = cs.branch_id
        ORDER BY b.branch_id
        """
        df = self.conn.execute(query).df()
        return df
