"""Synthetic Pakistani Banking Data Generator (Branches, CASA Accounts, Transactions, Daily Liquidity).

Simulates realistic United Bank Limited (UBL) style retail, SME, and corporate banking data:
- 30 Bank Branches across Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta
- 1,200 Customer CASA & Term Deposit Accounts with realistic demographic & digital onboarding attributes
- 180 Days of Branch-Level Daily Deposit Inflows, Outflows, ATM Cash Replenishment, and Macro Indicators (KIBOR, Eid/Salary spikes)
- 45,000+ Transactional records across RAAST, IBFT, ATM, POS, Branch Counter, and Mobile App channels,
  including injected fraud/structuring anomalies for benchmarking Isolation Forest & velocity rules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REGIONS = [
    ("Karachi South", "Sindh", "Metropolitan", 1.45),
    ("Karachi Clifton", "Sindh", "Metropolitan", 1.60),
    ("Karachi Gulshan", "Sindh", "Urban", 1.20),
    ("Lahore Gulberg", "Punjab", "Metropolitan", 1.50),
    ("Lahore DHA", "Punjab", "Metropolitan", 1.55),
    ("Lahore Johar Town", "Punjab", "Urban", 1.15),
    ("Islamabad Blue Area", "ICT", "Metropolitan", 1.45),
    ("Islamabad F-10", "ICT", "Urban", 1.25),
    ("Rawalpindi Saddar", "Punjab", "Urban", 1.10),
    ("Faisalabad Clock Tower", "Punjab", "Commercial", 1.30),
    ("Multan Cantt", "Punjab", "Urban", 1.05),
    ("Peshawar University Rd", "KPK", "Urban", 0.95),
    ("Sialkot Export Zone", "Punjab", "Commercial", 1.25),
    ("Gujranwala GT Road", "Punjab", "Commercial", 1.10),
    ("Quetta Jinnah Road", "Balochistan", "Semi-Urban", 0.80),
]


def generate_banking_dataset(
    seed: int = 42,
    n_branches: int = 30,
    n_customers: int = 1200,
    n_days: int = 180,
    n_transactions: int = 45000,
) -> Dict[str, pd.DataFrame]:
    """Generate deterministic, realistic multi-table banking datasets."""
    rng = np.random.default_rng(seed)

    # 1. Branches Table (dim_branch)
    branches = []
    for i in range(n_branches):
        region_info = REGIONS[i % len(REGIONS)]
        branch_code = f"UBL-BR-{1001 + i}"
        tier = region_info[2]
        multiplier = region_info[3]
        staff_count = int(rng.integers(12, 36) * (1.15 if tier == "Metropolitan" else 0.9))
        atm_count = int(rng.integers(2, 6) if tier in ("Metropolitan", "Commercial") else rng.integers(1, 3))
        catchment_pop = int(rng.normal(85000, 22000) * multiplier)
        digital_campaign_enrolled = bool(i % 2 == 0)  # 50/50 split for A/B hypothesis testing

        branches.append(
            {
                "branch_id": branch_code,
                "branch_name": f"{region_info[0]} Branch #{i // len(REGIONS) + 1}",
                "city": region_info[0].split()[0],
                "province": region_info[1],
                "branch_tier": tier,
                "staff_count": max(8, staff_count),
                "atm_count": max(1, atm_count),
                "catchment_population": max(20000, catchment_pop),
                "monthly_opex_pkr": round(float(staff_count * 185000 + atm_count * 95000 + rng.normal(450000, 60000)), 2),
                "digital_onboarding_campaign": digital_campaign_enrolled,
                "regional_multiplier": multiplier,
            }
        )
    df_branches = pd.DataFrame(branches)

    # 2. Customers Table (dim_customer)
    segments_archetype = ["Salaried_Retail", "Affluent_HNW", "SME_Trader", "Freelancer_Remittance", "Dormant_At_Risk"]
    archetype_probs = [0.44, 0.12, 0.18, 0.14, 0.12]
    chosen_archetypes = rng.choice(segments_archetype, size=n_customers, p=archetype_probs)
    assigned_branches = rng.choice(df_branches["branch_id"].values, size=n_customers)

    customers = []
    for idx in range(n_customers):
        cid = f"CUST-{200001 + idx}"
        arch = chosen_archetypes[idx]
        br_id = assigned_branches[idx]

        if arch == "Affluent_HNW":
            avg_balance = float(rng.lognormal(mean=15.1, sigma=0.55))  # ~3.5M - 12M PKR
            monthly_income = float(rng.normal(950000, 220000))
            products_held = int(rng.integers(3, 6))
            credit_score = int(np.clip(rng.normal(765, 35), 650, 850))
            digital_active = True
        elif arch == "SME_Trader":
            avg_balance = float(rng.lognormal(mean=14.4, sigma=0.65))  # ~1.8M - 6M PKR
            monthly_income = float(rng.normal(680000, 190000))
            products_held = int(rng.integers(2, 5))
            credit_score = int(np.clip(rng.normal(715, 45), 580, 840))
            digital_active = bool(rng.random() < 0.85)
        elif arch == "Freelancer_Remittance":
            avg_balance = float(rng.lognormal(mean=13.2, sigma=0.50))  # ~500k - 1.5M PKR
            monthly_income = float(rng.normal(340000, 85000))
            products_held = int(rng.integers(2, 4))
            credit_score = int(np.clip(rng.normal(710, 40), 600, 820))
            digital_active = True
        elif arch == "Salaried_Retail":
            avg_balance = float(rng.lognormal(mean=11.9, sigma=0.55))  # ~150k - 450k PKR
            monthly_income = float(rng.normal(145000, 45000))
            products_held = int(rng.integers(1, 3))
            credit_score = int(np.clip(rng.normal(685, 50), 520, 810))
            digital_active = bool(rng.random() < 0.78)
        else:  # Dormant_At_Risk
            avg_balance = float(rng.lognormal(mean=9.8, sigma=0.65))  # ~18k - 60k PKR
            monthly_income = float(rng.normal(85000, 25000))
            products_held = 1
            credit_score = int(np.clip(rng.normal(610, 55), 450, 740))
            digital_active = bool(rng.random() < 0.20)

        tenure_months = int(rng.integers(4, 144))
        account_type = rng.choice(["Current_CASA", "Savings_PLS", "Asaan_Digital", "SME_Business"], p=[0.42, 0.35, 0.13, 0.10])

        customers.append(
            {
                "customer_id": cid,
                "branch_id": br_id,
                "account_type": account_type,
                "latent_archetype": arch,
                "tenure_months": tenure_months,
                "declared_monthly_income_pkr": round(max(30000.0, monthly_income), 2),
                "current_casa_balance_pkr": round(max(1500.0, avg_balance), 2),
                "products_count": products_held,
                "internal_credit_score": credit_score,
                "mobile_app_enrolled": digital_active,
                "kyc_risk_tier": "HIGH" if (arch == "SME_Trader" and rng.random() < 0.15) else ("MEDIUM" if arch == "Freelancer_Remittance" else "LOW"),
            }
        )
    df_customers = pd.DataFrame(customers)

    # 3. Daily Branch Liquidity & ATM Cash Demand Time-Series (fact_branch_daily_liquidity)
    start_date = pd.Timestamp("2026-05-01")
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    daily_rows = []
    for _, br in df_branches.iterrows():
        br_id = br["branch_id"]
        mult = br["regional_multiplier"]
        campaign_boost = 1.14 if br["digital_onboarding_campaign"] else 1.00
        base_inflow = 14_500_000.0 * mult * campaign_boost
        base_outflow = 11_800_000.0 * mult
        base_atm = 3_600_000.0 * br["atm_count"] * (0.85 + 0.15 * mult)

        ar_error = 0.0
        for d_idx, dt in enumerate(dates):
            dow = dt.day_of_week  # 0=Mon .. 6=Sun
            dom = dt.day

            # Weekly seasonality (lower branch counter traffic on weekends, higher ATM usage on Fri/Sat)
            weekend_branch_factor = 0.25 if dow == 6 else (0.55 if dow == 5 else 1.08)
            weekend_atm_factor = 1.32 if dow in (4, 5) else (1.15 if dow == 6 else 0.92)

            # Salary cycle spike (1st to 7th of each month & 28th to 31st)
            salary_spike = 1.42 if (1 <= dom <= 6 or dom >= 28) else 0.94

            # Festive / Eid / Tax quarter spike around days 40-48 and 110-118
            festive_spike = 1.55 if (40 <= d_idx <= 47 or 110 <= d_idx <= 116) else 1.0

            # Gradual upward growth trend + autoregressive state
            trend = 1.0 + 0.00085 * d_idx
            ar_error = 0.55 * ar_error + float(rng.normal(0, 420_000 * mult))

            deposit_inflow = max(
                1_200_000.0,
                base_inflow * trend * weekend_branch_factor * salary_spike * (1.0 + float(rng.normal(0, 0.06))) + ar_error,
            )
            deposit_outflow = max(
                950_000.0,
                base_outflow * trend * weekend_branch_factor * (1.18 if salary_spike > 1.0 else 0.96) * festive_spike * (1.0 + float(rng.normal(0, 0.065))),
            )
            atm_withdrawal = max(
                450_000.0,
                base_atm * trend * weekend_atm_factor * salary_spike * festive_spike * (1.0 + float(rng.normal(0, 0.055))),
            )

            kibor_rate = round(17.50 - 0.012 * d_idx + float(rng.normal(0, 0.03)), 2)
            active_digital_txns = int(
                max(50, (420 * mult * campaign_boost * trend * (1.20 if salary_spike > 1.0 else 1.0)) + rng.normal(0, 35))
            )

            daily_rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "branch_id": br_id,
                    "deposit_inflow_pkr": round(deposit_inflow, 2),
                    "deposit_outflow_pkr": round(deposit_outflow, 2),
                    "net_liquidity_flow_pkr": round(deposit_inflow - deposit_outflow, 2),
                    "atm_cash_demand_pkr": round(atm_withdrawal, 2),
                    "digital_channel_txn_count": active_digital_txns,
                    "kibor_rate_pct": kibor_rate,
                    "is_salary_window": int(salary_spike > 1.0),
                    "is_festive_window": int(festive_spike > 1.0),
                    "is_weekend": int(dow >= 5),
                }
            )
    df_daily_liquidity = pd.DataFrame(daily_rows)

    # 4. Customer Transactions Table (fact_transactions) with Injected Fraud/Structuring Anomalies
    channels = ["RAAST_P2P", "IBFT_Transfer", "Mobile_App_Bill", "ATM_Withdrawal", "POS_Card", "Branch_OTC"]
    channel_probs = [0.32, 0.22, 0.18, 0.14, 0.09, 0.05]

    cust_records = df_customers.to_dict("records")
    # Weight sampling probability by archetype activity
    arch_weights = {
        "Affluent_HNW": 2.2,
        "SME_Trader": 2.8,
        "Freelancer_Remittance": 1.6,
        "Salaried_Retail": 1.1,
        "Dormant_At_Risk": 0.12,
    }
    weights = np.array([arch_weights[c["latent_archetype"]] for c in cust_records], dtype=float)
    weights /= weights.sum()

    sampled_cust_indices = rng.choice(len(cust_records), size=n_transactions, p=weights)
    sampled_channels = rng.choice(channels, size=n_transactions, p=channel_probs)
    day_offsets = rng.integers(0, n_days, size=n_transactions)
    hours = rng.choice(
        np.arange(24),
        size=n_transactions,
        p=[
            0.01, 0.005, 0.005, 0.005, 0.005, 0.01,
            0.02, 0.04, 0.06, 0.08, 0.09, 0.09,
            0.08, 0.08, 0.08, 0.07, 0.06, 0.05,
            0.05, 0.04, 0.03, 0.02, 0.01, 0.01,
        ],
    )
    minutes = rng.integers(0, 60, size=n_transactions)

    txns = []
    for t_idx in range(n_transactions):
        cust = cust_records[sampled_cust_indices[t_idx]]
        arch = cust["latent_archetype"]
        ch = sampled_channels[t_idx]
        d_off = int(day_offsets[t_idx])
        hr = int(hours[t_idx])

        # Dormant accounts mostly transacted early in the 180-day window (high Recency gap)
        if arch == "Dormant_At_Risk" and d_off > 65:
            d_off = int(rng.integers(0, 60))

        base_amt = {
            "Affluent_HNW": 185_000.0,
            "SME_Trader": 240_000.0,
            "Freelancer_Remittance": 68_000.0,
            "Salaried_Retail": 18_500.0,
            "Dormant_At_Risk": 6_500.0,
        }[arch]

        amt = float(rng.lognormal(mean=np.log(base_amt), sigma=0.62))
        is_anomaly = 0
        anomaly_reason = "NORMAL"

        # Inject ~1.6% realistic banking anomalies (AML structuring, midnight velocity burst, dormant sudden drain)
        if t_idx % 62 == 0:
            is_anomaly = 1
            anomaly_type = t_idx % 3
            if anomaly_type == 0:
                # High-value midnight IBFT / RAAST drain
                amt = float(rng.uniform(2_400_000.0, 6_800_000.0))
                hr = int(rng.choice([1, 2, 3, 4]))
                ch = "IBFT_Transfer"
                anomaly_reason = "MIDNIGHT_HIGH_VALUE_DRAIN"
            elif anomaly_type == 1:
                # Structuring just below 2.5M PKR regulatory reporting threshold
                amt = float(rng.uniform(2_450_000.0, 2_499_500.0))
                ch = "RAAST_P2P"
                anomaly_reason = "AML_THRESHOLD_STRUCTURING"
            else:
                # Sudden extreme spike relative to declared monthly income
                amt = float(cust["declared_monthly_income_pkr"] * rng.uniform(12.0, 22.0))
                hr = int(rng.choice([0, 2, 3, 23]))
                ch = "IBFT_Transfer"
                anomaly_reason = "INCOME_DISPROPORTIONATE_SPIKE"

        ts = start_date + pd.Timedelta(days=d_off, hours=hr, minutes=int(minutes[t_idx]))
        direction = "CREDIT" if (ch in ("RAAST_P2P", "IBFT_Transfer", "Branch_OTC") and rng.random() < 0.42 and not is_anomaly) else "DEBIT"

        txns.append(
            {
                "transaction_id": f"TXN-{5000001 + t_idx}",
                "customer_id": cust["customer_id"],
                "branch_id": cust["branch_id"],
                "txn_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "txn_date": ts.strftime("%Y-%m-%d"),
                "txn_hour": hr,
                "channel": ch,
                "direction": direction,
                "amount_pkr": round(max(100.0, amt), 2),
                "is_cross_border_or_Remittance": int(arch == "Freelancer_Remittance" and direction == "CREDIT"),
                "ground_truth_anomaly": is_anomaly,
                "anomaly_pattern_code": anomaly_reason,
            }
        )

    df_transactions = pd.DataFrame(txns).sort_values("txn_timestamp").reset_index(drop=True)

    return {
        "dim_branch": df_branches,
        "dim_customer": df_customers,
        "fact_branch_daily_liquidity": df_daily_liquidity,
        "fact_transactions": df_transactions,
    }


def save_raw_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str | Path) -> Dict[str, Path]:
    """Persist generated banking tables to CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}
    for name, df in datasets.items():
        file_path = out_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        saved_paths[name] = file_path
    return saved_paths


if __name__ == "__main__":
    data = generate_banking_dataset()
    paths = save_raw_datasets(data, Path(__file__).resolve().parents[1] / "data" / "raw")
    for table_name, p in paths.items():
        print(f"Saved {table_name}: {len(data[table_name]):,} rows -> {p}")
