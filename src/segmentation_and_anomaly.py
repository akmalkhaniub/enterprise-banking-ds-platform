"""Pillar 2: Customer 360 Clustering/Segmentation & Unsupervised/Statistical Transaction Anomaly Detection.

Satisfies UBL JD Requirements:
- "Build, validate and maintain machine learning and AI models for forecasting, segmentation, bench-marking and anomaly detection"
- "Practical experience with time-series forecasting, regression and clustering"
- "Experience with large customer or transactional datasets in any industry (e.g. telecom, fintech, e-commerce, retail, banking)"
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler


SEGMENT_ACTION_PLAYBOOK: Dict[str, str] = {
    "High_Value_SME_Commercial": "Assign dedicated SME Relationship Manager; pitch Working Capital Finance & POS merchant acquiring fee waiver.",
    "Affluent_HNW_Wealth": "Priority Banking upgrade; pitch UBL Ameen / Mutual Fund wealth products and Premier multi-currency Visa Infinite.",
    "Digital_Active_Growth": "Pre-approve instant digital personal loan / BNPL credit line via UBL Digital App push notification.",
    "Salaried_Core_Retail": "Cross-sell Bancassurance & Auto-Invest SIP; incentivize utility bill auto-debit for sticky CASA float.",
    "Dormant_Churn_Risk": "Trigger automated 90-day dormancy win-back campaign (zero-fee RAAST cashback + RM outbound call).",
}


def segment_customers_rfm_clustering(
    df_customer_360: pd.DataFrame,
    n_clusters: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run K-Means and Gaussian Mixture Model (GMM) clustering on Customer 360 RFM + behavioral features.

    Evaluates cluster quality across k=3..6 using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz,
    then assigns human-readable banking personas and NBA (Next-Best-Action) recommendations.
    """
    df = df_customer_360.copy()
    feature_cols = [
        "recency_days",
        "txn_frequency_180d",
        "monetary_total_volume_pkr",
        "current_casa_balance_pkr",
        "declared_monthly_income_pkr",
        "products_count",
        "digital_txn_share",
        "internal_credit_score",
    ]

    # Log-transform skewed monetary columns before standardization
    X_raw = df[feature_cols].copy()
    for col in ("monetary_total_volume_pkr", "current_casa_balance_pkr", "declared_monthly_income_pkr"):
        X_raw[col] = np.log1p(np.maximum(X_raw[col].values, 0.0))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Evaluate candidate k values (k = 3, 4, 5, 6) for model selection justification
    k_sweep: List[Dict[str, Any]] = []
    for k in (3, 4, 5, 6):
        km_candidate = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km_candidate.fit_predict(X_scaled)
        k_sweep.append(
            {
                "k": k,
                "silhouette_score": round(float(silhouette_score(X_scaled, labels_k)), 4),
                "davies_bouldin_index": round(float(davies_bouldin_score(X_scaled, labels_k)), 4),
                "calinski_harabasz_score": round(float(calinski_harabasz_score(X_scaled, labels_k)), 2),
                "inertia": round(float(km_candidate.inertia_), 2),
            }
        )

    # Fit primary K-Means (k=5) and companion Gaussian Mixture Model (GMM) for soft assignment confidence
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    cluster_ids = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled).max(axis=1)

    df["cluster_id"] = cluster_ids
    df["cluster_assignment_confidence"] = np.round(gmm_probs, 4)

    # Deterministically map numeric cluster IDs to banking business personas based on centroid characteristics
    cluster_stats = (
        df.groupby("cluster_id")
        .agg(
            mean_recency=("recency_days", "mean"),
            mean_freq=("txn_frequency_180d", "mean"),
            mean_volume=("monetary_total_volume_pkr", "mean"),
            mean_balance=("current_casa_balance_pkr", "mean"),
            mean_income=("declared_monthly_income_pkr", "mean"),
        )
        .reset_index()
    )

    assigned_names: Dict[int, str] = {}
    remaining = set(cluster_stats["cluster_id"].tolist())

    # 1. Dormant / Churn Risk = highest recency_days (least recent activity) & lowest frequency
    dormant_cid = int(cluster_stats.loc[cluster_stats["cluster_id"].isin(remaining)].sort_values("mean_recency", ascending=False).iloc[0]["cluster_id"])
    assigned_names[dormant_cid] = "Dormant_Churn_Risk"
    remaining.remove(dormant_cid)

    # 2. Affluent HNW Wealth = highest CASA balance among remaining
    affluent_cid = int(cluster_stats.loc[cluster_stats["cluster_id"].isin(remaining)].sort_values("mean_balance", ascending=False).iloc[0]["cluster_id"])
    assigned_names[affluent_cid] = "Affluent_HNW_Wealth"
    remaining.remove(affluent_cid)

    # 3. High-Value SME Commercial = highest transaction volume among remaining
    sme_cid = int(cluster_stats.loc[cluster_stats["cluster_id"].isin(remaining)].sort_values("mean_volume", ascending=False).iloc[0]["cluster_id"])
    assigned_names[sme_cid] = "High_Value_SME_Commercial"
    remaining.remove(sme_cid)

    # 4. Digital Active Growth vs Salaried Core Retail
    rem_sorted = cluster_stats.loc[cluster_stats["cluster_id"].isin(remaining)].sort_values("mean_volume", ascending=False)
    if len(rem_sorted) >= 1:
        growth_cid = int(rem_sorted.iloc[0]["cluster_id"])
        assigned_names[growth_cid] = "Digital_Active_Growth"
        remaining.remove(growth_cid)
    for cid in remaining:
        assigned_names[int(cid)] = "Salaried_Core_Retail"

    df["customer_segment"] = df["cluster_id"].map(assigned_names)
    df["recommended_next_best_action"] = df["customer_segment"].map(SEGMENT_ACTION_PLAYBOOK)

    # Estimate 12-month Customer Lifetime Value (CLV) in PKR based on CASA float Net Interest Margin (NIM ~ 4.8%) + fee velocity
    df["estimated_annual_clv_pkr"] = np.round(
        df["current_casa_balance_pkr"] * 0.048 + df["monetary_total_volume_pkr"] * 0.0035,
        2,
    )

    segment_profiles = (
        df.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_txn_frequency_180d=("txn_frequency_180d", "mean"),
            avg_monetary_volume_pkr=("monetary_total_volume_pkr", "mean"),
            avg_casa_balance_pkr=("current_casa_balance_pkr", "mean"),
            total_segment_casa_pkr=("current_casa_balance_pkr", "sum"),
            avg_annual_clv_pkr=("estimated_annual_clv_pkr", "mean"),
            digital_txn_share=("digital_txn_share", "mean"),
        )
        .sort_values("total_segment_casa_pkr", ascending=False)
    )
    for col in ("avg_recency_days", "avg_txn_frequency_180d", "avg_monetary_volume_pkr", "avg_casa_balance_pkr", "total_segment_casa_pkr", "avg_annual_clv_pkr"):
        segment_profiles[col] = segment_profiles[col].round(2)
    segment_profiles["digital_txn_share"] = segment_profiles["digital_txn_share"].round(4)
    segment_profiles["recommended_action"] = segment_profiles["customer_segment"].map(SEGMENT_ACTION_PLAYBOOK)

    summary = {
        "selected_k": n_clusters,
        "silhouette_score": round(float(silhouette_score(X_scaled, cluster_ids)), 4),
        "davies_bouldin_index": round(float(davies_bouldin_score(X_scaled, cluster_ids)), 4),
        "calinski_harabasz_score": round(float(calinski_harabasz_score(X_scaled, cluster_ids)), 2),
        "k_selection_sweep": k_sweep,
        "segment_profiles": segment_profiles.to_dict("records"),
    }

    return df, summary


def detect_transaction_anomalies(
    df_txn_features: pd.DataFrame,
    contamination: float = 0.018,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run hybrid Unsupervised Isolation Forest + Statistical Velocity/AML Rule Engine on 45,000+ transactions.

    Combines:
    1. `scikit-learn` IsolationForest on multi-dimensional behavioral ratios (`customer_amount_zscore`,
       `amount_to_monthly_income_ratio`, `is_odd_hour`, `is_near_aml_reporting_threshold`, `customer_same_day_volume_pkr`).
    2. Deterministic compliance & statistical Z-score rules for explainable reason codes.
    """
    df = df_txn_features.copy()

    feature_cols = [
        "amount_pkr",
        "customer_amount_zscore",
        "amount_to_monthly_income_ratio",
        "is_odd_hour",
        "is_near_aml_reporting_threshold",
        "customer_same_day_txn_count",
        "customer_same_day_volume_pkr",
    ]

    X = df[feature_cols].copy()
    X["amount_pkr"] = np.log1p(np.maximum(X["amount_pkr"].values, 0.0))
    X["customer_same_day_volume_pkr"] = np.log1p(np.maximum(X["customer_same_day_volume_pkr"].values, 0.0))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        max_samples=2048,
        random_state=42,
        n_jobs=1,
    )
    iso_preds = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
    raw_scores = -iso_forest.score_samples(X_scaled)  # higher = more anomalous

    # Normalize risk score to 0..100 scale
    min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
    norm_scores = ((raw_scores - min_s) / max(max_s - min_s, 1e-6)) * 100.0

    # Combine Isolation Forest with high-precision statistical Z-score & AML structuring triggers
    rule_high_z_odd_hour = (df["customer_amount_zscore"] >= 3.8) & (df["is_odd_hour"] == 1) & (df["amount_pkr"] >= 1_500_000)
    rule_aml_structuring = (df["is_near_aml_reporting_threshold"] == 1) & (df["channel"].isin(["RAAST_P2P", "IBFT_Transfer"])) & (df["customer_amount_zscore"] >= 2.5)
    rule_income_spike = (df["amount_to_monthly_income_ratio"] >= 10.0) & (df["is_odd_hour"] == 1)

    hybrid_flag = ((iso_preds == -1) & (df["customer_amount_zscore"] >= 2.8)) | rule_high_z_odd_hour | rule_aml_structuring | rule_income_spike

    df["anomaly_risk_score"] = np.round(
        np.where(hybrid_flag, np.maximum(norm_scores, 78.0), norm_scores * 0.65),
        2,
    )
    df["model_anomaly_flag"] = hybrid_flag.astype(int)

    # Assign human-explainable reason code for each flagged alert
    conditions = [
        rule_aml_structuring,
        rule_income_spike,
        rule_high_z_odd_hour,
        df["model_anomaly_flag"] == 1,
    ]
    choices = [
        "AML_STRUCTURING_NEAR_2.5M_THRESHOLD",
        "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR",
        "MIDNIGHT_HIGH_VALUE_VELOCITY_DRAIN",
        "MULTIVARIATE_ISOLATION_FOREST_OUTLIER",
    ]
    df["explainable_alert_reason"] = np.select(conditions, choices, default="NORMAL_BEHAVIOR")

    # Evaluate against ground-truth injected anomalies
    y_true = df["ground_truth_anomaly"].values
    y_pred = df["model_anomaly_flag"].values
    y_score = df["anomaly_risk_score"].values

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_score))

    flagged_df = df[df["model_anomaly_flag"] == 1].sort_values("anomaly_risk_score", ascending=False)
    by_reason = (
        flagged_df.groupby("explainable_alert_reason", as_index=False)
        .agg(
            alert_count=("transaction_id", "count"),
            exposed_volume_pkr=("amount_pkr", "sum"),
            avg_risk_score=("anomaly_risk_score", "mean"),
        )
        .sort_values("exposed_volume_pkr", ascending=False)
    )
    by_reason["exposed_volume_pkr"] = by_reason["exposed_volume_pkr"].round(2)
    by_reason["avg_risk_score"] = by_reason["avg_risk_score"].round(2)

    metrics = {
        "model": "Hybrid Isolation Forest (150 trees) + Statistical Z-Score & AML Structuring Rules",
        "total_transactions_scanned": int(len(df)),
        "total_anomalies_flagged": int(y_pred.sum()),
        "anomaly_rate_pct": round(float(y_pred.mean() * 100.0), 3),
        "total_exposed_anomaly_volume_pkr": round(float(flagged_df["amount_pkr"].sum()), 2),
        "benchmark_evaluation_vs_injected_fraud": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "alerts_by_reason_breakdown": by_reason.to_dict("records"),
    }

    return df, metrics
