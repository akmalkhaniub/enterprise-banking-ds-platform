# Enterprise Multi-Domain Data Science, Forecasting, Customer 360 & Controlled GenAI BI Platform 🏦🛒📡📈

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python)](https://python.org)
[![DuckDB](https://img.shields.io/badge/SQL-DuckDB-FFF000?style=flat&logo=duckdb)](https://duckdb.org)
[![Statsmodels](https://img.shields.io/badge/Econometrics-statsmodels-2C5BB4?style=flat)](https://www.statsmodels.org)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Power BI](https://img.shields.io/badge/BI-Power%20BI%20(.pbip%20%2F%20TMDL)-F2C811?style=flat)](https://powerbi.microsoft.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akmalkhaniub/enterprise-banking-ds-platform/blob/main/notebooks/01_End_to_End_Banking_Experiments.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/akmalkhaniub/enterprise-banking-ds-platform/blob/main/notebooks/01_End_to_End_Banking_Experiments.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Production-grade statistical econometrics, machine learning, analytical SQL feature store, and Power BI / Tableau Star-Schema intelligence platform engineered with a Canonical Multi-Domain Adapter across Banking, E-Commerce & Retail, Telecom, and Marketing/Fintech.**

---

## 🧠 Why a Multi-Domain Architecture? (Cross-Industry Mathematical Isomorphism)

A fundamental principle of production data science is that **customer and transactional decision problems are mathematically isomorphic across industries**. Whether a data scientist is analyzing **Commercial Banking**, **E-Commerce & Retail**, **Telecommunications**, or **Performance Marketing & Fintech**, the underlying data structures (`Entity/Node`, `Customer Account`, `Time-Stamped Transaction`, and `Daily Capacity Time-Series`) and statistical engines are identical—only the domain semantics change:

| Core Data Science Engine | 🏦 Banking & Treasury (`banking`) | 🛒 E-Commerce & Retail (`ecommerce_retail`) | 📡 Telecom & CVM (`telecom`) | 📈 Marketing & Fintech (`marketing_fintech`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Time-Series Forecasting** (`SARIMAX` + `GradientBoosting`) | Daily Branch Deposit Inflows & ATM Cash Replenishment Demand | Daily Store/Hub GMV & Warehouse SKU Stock Replenishment | Regional Recharge Inflow & Cell-Site Backhaul Bandwidth Load | Daily Attributed Revenue Inflow & Programmatic Ad Spend Pacing |
| **2. Peer Benchmarking & Hypothesis Testing** (`OLS Residuals`, `t-test`, `ANOVA`) | Branch Efficiency Index (controlling for Staff, ATMs, Catchment, OPEX) | Fulfillment Store Index (controlling for Headcount, SqFt, Catchment, OPEX) | Regional Tower Cluster Efficiency & 5G Bundle ARPU Lift | Growth Pod ROAS Incrementality & Channel Tier ANOVA |
| **3. Customer 360 Segmentation** (`RFM + K-Means + GMM`) | CASA Depositors (`Affluent_HNW`, `SME_Commercial`, `Salaried`, `Dormant`) | Shoppers (`VIP_Champions`, `B2B_Wholesale`, `App_Shoppers`, `Cart_Abandoners`) | Subscribers (`Platinum_Roamers`, `Enterprise_IoT`, `5G_Streamers`, `Multi_SIM_Churners`) | Cohorts (`Power_Advocates`, `Merchant_Partners`, `Promo_Converters`, `Bonus_Hunters`) |
| **4. Anomaly & Risk Detection** (`Isolation Forest` + `Velocity Z-Scores`) | AML Structuring (<2.5M threshold) & Midnight IBFT Account Takeover | Serial Refund/Coupon Stacking Abuse, Bot Scalping & Stolen-Card Voucher Drains | SIM-Box International Grey-Traffic Bypass & Wangiri Call Pumping | Click-Farm Invalid Traffic (IVT), Referral Rings & Promo Brute-Force Exploits |
| **5. Controlled GenAI & BI Mart** (`DuckDB SQL` + `Pydantic` + `Power BI`) | ALCO, Branch Ops & AML Compliance Executive Briefs | Merchandising, Supply Chain & Trust/Safety Briefs | Network Ops, CVM Retention & Revenue Assurance Briefs | CMO Growth Committee, Attribution & Ad-Fraud Briefs |

Rather than duplicating code across four separate repositories, this platform implements a **Canonical SQL & Mathematical Core** paired with **Switchable Industry Domain Packs** ([`src/domain_packs.py`](src/domain_packs.py) & [`src/domain_adapter.py`](src/domain_adapter.py)) selectable live in the Web Console header, REST API (`?domain=telecom`), and Jupyter Notebook.

---

## 🎯 Core Data Science & Engineering Capabilities

| Capability Domain | Engineering & Data Science Implementation | Verified Metric / Artifact |
| :--- | :--- | :--- |
| **Time-Series Forecasting & Regression** | `statsmodels` **SARIMAX(1,0,1)x(1,0,1,7)** with exogenous payday/festive/macro regressors + `scikit-learn` **Gradient Boosting** lag-feature ensemble (`src/forecasting_and_benchmarking.py`). | **3.02% Out-of-Time MAPE** (beats 7D seasonal naive baseline) across 30-day walk-forward backtest. |
| **Peer Benchmarking & Hypothesis Testing** | Peer-adjusted **Statsmodels OLS & Ridge Regression** residual efficiency index (`Actual / Expected * 100`) controlling for capacity, footprint, catchment population & OPEX + **ADF Stationarity**, **Welch's Two-Sample t-test**, and **One-Way ANOVA**. | **OLS $R^2 = 0.89$**; Strategic Campaign uplift **+13.9%** ($p < 0.001$); Peak Cycle surge **+37.1%** ($p < 10^{-8}$). |
| **Customer Segmentation & Clustering** | Customer 360 RFM (Recency, Frequency, Monetary) + behavioral feature store clustered via **K-Means ($K=5$)** and **Gaussian Mixture Models (GMM)** soft-probability assignment (`src/segmentation_and_anomaly.py`). | **5 Actionable Personas per Industry** with **Silhouette = 0.303**, 12-month CLV estimation, and automated Next-Best-Action playbooks. |
| **Transaction Anomaly Detection** | Hybrid **Unsupervised Isolation Forest (150 trees)** + statistical customer velocity Z-score & threshold structuring rules (`src/segmentation_and_anomaly.py`). | **F1 = 0.914**, **ROC-AUC = 0.9996** across **45,000** transactions (**679** high-risk alerts isolated). |
| **Strong SQL & Data Engineering Pipelines** | Analytical **DuckDB SQL** feature store (`src/sql_pipeline.py`) using multi-stage CTEs, `OVER (PARTITION BY ...)` window functions, and 4 automated Data Quality gates (PK uniqueness, FK integrity, accounting balance invariants, time-series continuity). | **4/4 Automated SQL Data Quality Checks Passing** with zero orphan keys or accounting breaks. |
| **Controlled Enterprise GenAI / LLMs** | Schema-constrained (`Pydantic v2`), **PII-redacting** (masks National IDs, IBANs, and customer IDs) executive synthesis engine (`src/genai_narrator.py`) with SHA-256 prompt auditability and explicit **Model Assumptions & Limitations** blocks. | Zero-hallucination Executive Briefs & Recommendations grounded strictly in SQL/ML metrics. |
| **Power BI / Tableau & Automated Dashboards** | Native **Power BI Project (`bi/BankingExecutiveIntelligence.pbip`)** with **TMDL Semantic Model**, pre-authored **DAX measures**, role-based security, 5 Star-Schema CSV exports (`bi/mart_exports/*.csv`), plus a live **FastAPI Executive Console** with Multi-Domain Switcher & What-If Stress Simulator. | Instant Power BI Desktop open + live interactive web console at `http://localhost:8000`. |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph RawLayer ["1. Canonical Transactional & Entity Sources"]
        B["dim_branch (30 Network Nodes: Branches / Stores / Towers / Pods)"]
        C["dim_customer (1,200 Customer / Subscriber / Buyer Accounts)"]
        L["fact_branch_daily_liquidity (180 Days x 30 Nodes Time-Series)"]
        T["fact_transactions (45,000+ Multi-Channel Transactions)"]
    end

    subgraph SQLLayer ["2. DuckDB Analytical SQL & Data Quality Engine (src/sql_pipeline.py)"]
        DQ["4 Automated SQL Data Quality Gates<br/>• PK/FK Integrity • Balance Invariants • Date Continuity"]
        W["SQL Window Feature Store<br/>• Customer 360 RFM CTEs<br/>• Rolling Z-Score & Same-Day Velocity"]
    end

    subgraph MLLayer ["3. Statistical & Machine Learning Pillars"]
        P1["Pillar 1: Forecasting & Benchmarking<br/>• SARIMAX(1,0,1)(1,0,1,7) + GradientBoosting<br/>• Statsmodels OLS Peer-Adjusted Residuals<br/>• ADF, Welch t-test, One-Way ANOVA"]
        P2["Pillar 2: Segmentation & Anomaly Shield<br/>• K-Means (k=5) + GMM Soft Probabilities<br/>• Isolation Forest + Velocity Z-Score Rules"]
        P3["Pillar 3: Controlled Enterprise GenAI<br/>• National ID / IBAN PII Redaction Guardrail<br/>• Pydantic Schema-Grounded Executive Briefs<br/>• Explicit Assumptions & Limitations"]
    end

    subgraph DomainLayer ["4. Multi-Domain Isomorphism Adapter (src/domain_packs.py)"]
        D1["🏦 Banking & Treasury"]
        D2["🛒 E-Commerce & Retail"]
        D3["📡 Telecom & CVM"]
        D4["📈 Marketing & Fintech"]
    end

    subgraph DeliveryLayer ["5. BI Star-Schema Mart & Executive Console"]
        PBI["Power BI Project (.pbip + TMDL + DAX)<br/>bi/BankingExecutiveIntelligence.pbip"]
        CSV["5 Star-Schema Mart Tables<br/>bi/mart_exports/*.csv"]
        UI["FastAPI Interactive Executive Console<br/>+ Live Industry Switcher & Stress Simulator"]
    end

    B & C & L & T --> DQ --> W
    W --> P1 & P2
    P1 & P2 --> P3
    P1 & P2 & P3 --> D1 & D2 & D3 & D4
    D1 & D2 & D3 & D4 --> UI
    P1 & P2 & P3 --> CSV --> PBI
```

---

## 📂 Repository Structure

```text
enterprise-banking-ds-platform/
├── bi/
│   ├── BankingExecutiveIntelligence.pbip                 # Native Power BI Project file
│   ├── BankingExecutiveIntelligence.Report/              # Power BI Report layout & 3 pages
│   ├── BankingExecutiveIntelligence.SemanticModel/       # TMDL Star Schema, Relationships, Roles & DAX Measures
│   └── mart_exports/                                     # Materialized Star-Schema CSVs for Power BI / Tableau
│       ├── DimBranchBenchmark.csv
│       ├── DimCustomerSegment.csv
│       ├── FactLiquidityForecast.csv
│       ├── FactTransactionAnomaly.csv
│       └── FactExecutiveRecommendation.csv
├── notebooks/
│   └── 01_End_to_End_Banking_Experiments.ipynb           # 1-Click Google Colab & Kaggle Notebook
├── src/
│   ├── domain_packs.py                 # 4 Switchable Industry Domain Packs (Banking, Retail, Telecom, Marketing)
│   ├── domain_adapter.py               # Projects canonical SQL/ML outputs into target industry semantics
│   ├── data_generator.py               # Realistic multi-entity transactional & seasonal time-series generator
│   ├── sql_pipeline.py                 # DuckDB SQL CTEs, Window functions & 4 Data Quality assertion gates
│   ├── forecasting_and_benchmarking.py # SARIMAX, GradientBoosting, OLS Benchmarking, ADF/t-test/ANOVA
│   ├── segmentation_and_anomaly.py     # RFM K-Means + GMM Clustering & Isolation Forest Anomaly Detection
│   ├── genai_narrator.py               # Controlled Enterprise GenAI Narrator with PII Redaction
│   ├── pipeline_runner.py              # End-to-end orchestrator exporting BI Mart & JSON artifacts
│   ├── api.py                          # FastAPI backend, Domain Switcher API & What-If Stress Simulator
│   └── static/index.html               # Human-centered Light/Dark Multi-Domain Executive Console
├── tests/
│   └── test_banking_ds_platform.py     # End-to-end Pytest suite (7 tests covering SQL, ML, PII, API, 4 Domains)
└── ci/ci.yml                           # Automated CI pipeline
```

---

## 🚀 Quickstart

### 1. Run the Full Data Science Pipeline & Generate Power BI Mart
```powershell
.\.venv\Scripts\python.exe -m src.pipeline_runner
```

### 2. Launch the Interactive Multi-Domain Executive Console
```powershell
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
```
Then open **`http://localhost:8000`** in your browser. Use the **Industry Pack** selector in the top header (`🏦 Banking & Treasury`, `🛒 E-Commerce & Retail`, `📡 Telecom & CVM`, `📈 Marketing & Fintech`) to switch domains in real time across:
1. **Executive Brief & Recommendations** (with collapsible *Model Assumptions, Confidence Intervals & Known Limitations*)
2. **Time-Series Forecasting & What-If Stress Simulator**
3. **Peer Benchmarking & Statistical Hypothesis Tests** (`statsmodels` OLS peer-adjusted efficiency & Welch's *t*-test / ANOVA)
4. **Customer 360 Segmentation & High-Risk Anomaly Queue** (K-Means/GMM personas + Isolation Forest alerts)
5. **SQL Data Quality & Power BI (`.pbip`) Star-Schema Mart Downloads**

### 3. Run the Automated Test Suite
```powershell
.\.venv\Scripts\pytest.exe -v
```
