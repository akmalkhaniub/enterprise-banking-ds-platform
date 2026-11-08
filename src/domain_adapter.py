"""Domain Adapter that projects the Canonical Mathematical & SQL Pipeline Bundle into any Industry Domain Pack."""

from __future__ import annotations

import copy
from typing import Any, Dict

from src.domain_packs import get_domain_pack, list_domain_packs


def adapt_bundle_to_domain(base_bundle: Dict[str, Any], domain_key: str = "banking") -> Dict[str, Any]:
    """Transform a pre-computed canonical bundle into the target industry domain pack (`banking`, `ecommerce_retail`, `telecom`, `marketing_fintech`)."""
    pack = get_domain_pack(domain_key)
    bundle = copy.deepcopy(base_bundle)
    scale = pack.value_scale

    bundle["active_domain"] = {
        "domain_key": pack.domain_key,
        "display_name": pack.display_name,
        "short_badge": pack.short_badge,
        "description": pack.description,
        "currency_unit": pack.currency_unit,
        "entity_label_singular": pack.entity_label_singular,
        "entity_label_plural": pack.entity_label_plural,
        "primary_inflow_label": pack.primary_inflow_label,
        "secondary_demand_label": pack.secondary_demand_label,
        "seasonality_spike_label": pack.seasonality_spike_label,
        "macro_rate_label": pack.macro_rate_label,
        "portfolio_balance_label": pack.portfolio_balance_label,
        "exec_committee_name": pack.exec_committee_name,
    }
    bundle["available_domains"] = list_domain_packs()

    if domain_key == "banking":
        return bundle

    # 1. Scale monetary KPIs while preserving dimensionless statistical metrics (MAPE, R², Silhouette, F1, ROC-AUC)
    kpis = bundle["kpis"]
    kpis["total_casa_book_pkr"] = round(kpis["total_casa_book_pkr"] * scale, 2)
    kpis["next_30d_atm_forecast_pkr"] = round(kpis["next_30d_atm_forecast_pkr"] * scale, 2)
    kpis["next_30d_deposit_inflow_forecast_pkr"] = round(kpis["next_30d_deposit_inflow_forecast_pkr"] * scale, 2)
    kpis["exposed_anomaly_volume_pkr"] = round(kpis["exposed_anomaly_volume_pkr"] * scale, 2)

    # 2. Adapt Hypothesis Tests
    hyps = bundle["hypothesis_testing"]["hypothesis_tests"]
    if len(hyps) >= 3:
        hyps[0]["hypothesis_id"] = pack.hyp1_title
        hyps[0]["business_question"] = pack.hyp1_question
        hyps[0]["business_recommendation"] = pack.hyp1_recommendation_template.format(
            uplift=hyps[0]["effect_size_uplift_pct"],
            pval=hyps[0]["p_value"],
        )
        hyps[1]["hypothesis_id"] = pack.hyp2_title
        hyps[1]["business_question"] = pack.hyp2_question
        hyps[1]["business_recommendation"] = pack.hyp2_recommendation_template.format(
            uplift=hyps[1]["effect_size_uplift_pct"],
            pval=hyps[1]["p_value"],
        )
        hyps[2]["hypothesis_id"] = pack.hyp3_title
        hyps[2]["business_question"] = pack.hyp3_question

    # 3. Scale Forecast Series & Error Metrics
    for m in bundle["forecasting"]["metrics"]["models"].values():
        m["mae_pkr"] = round(m["mae_pkr"] * scale, 2)
        m["rmse_pkr"] = round(m["rmse_pkr"] * scale, 2)

    for row in bundle["forecasting"]["series"]:
        for col in (
            "actual_atm_cash_demand_pkr",
            "forecast_atm_cash_demand_pkr",
            "forecast_atm_lower_95ci_pkr",
            "forecast_atm_upper_95ci_pkr",
            "actual_deposit_inflow_pkr",
            "forecast_deposit_inflow_pkr",
        ):
            if row.get(col) is not None:
                row[col] = round(float(row[col]) * scale, 2)

    # 4. Adapt Entity / Branch Benchmarking IDs & Names
    for b in bundle["branch_benchmarking"]["branches"]:
        b["branch_id"] = b["branch_id"].replace("BNK-BR", pack.entity_prefix)
        b["branch_name"] = b["branch_name"].replace("Branch", pack.entity_label_singular)
        for col in (
            "total_casa_book_pkr",
            "total_deposit_inflow_180d_pkr",
            "total_deposit_outflow_180d_pkr",
            "total_net_liquidity_180d_pkr",
            "expected_net_liquidity_180d_pkr",
            "value_add_residual_pkr",
        ):
            if col in b and b[col] is not None:
                b[col] = round(float(b[col]) * scale, 2)

    # 5. Adapt Customer 360 Personas & Next-Best-Actions
    for seg in bundle["customer_segmentation"]["segment_profiles"]:
        orig_name = seg["customer_segment"]
        mapped_name = pack.persona_names.get(orig_name, orig_name)
        seg["customer_segment"] = mapped_name
        seg["recommended_action"] = pack.persona_actions.get(mapped_name, seg["recommended_action"])
        for col in ("avg_monetary_volume_pkr", "avg_casa_balance_pkr", "total_segment_casa_pkr", "avg_annual_clv_pkr"):
            if col in seg and seg[col] is not None:
                seg[col] = round(float(seg[col]) * scale, 2)

    # 6. Adapt Transaction Anomalies & Channels
    channel_map = {
        "RAAST_P2P": pack.channels[0],
        "IBFT_Transfer": pack.channels[1],
        "Mobile_App_Bill": pack.channels[2],
        "ATM_Withdrawal": pack.channels[3],
        "POS_Card": pack.channels[4],
        "Branch_OTC": pack.channels[5],
    }
    for idx, alert in enumerate(bundle["anomaly_detection"]["top_alerts"]):
        alert["branch_id"] = str(alert.get("branch_id", "")).replace("BNK-BR", pack.entity_prefix)
        alert["channel"] = channel_map.get(alert["channel"], pack.channels[idx % len(pack.channels)])
        alert["amount_pkr"] = round(float(alert["amount_pkr"]) * scale, 2)
        orig_reason = alert["explainable_alert_reason"]
        alert["explainable_alert_reason"] = pack.anomaly_reasons.get(orig_reason, orig_reason)

    # 7. Adapt Controlled GenAI Executive Recommendations
    recs = bundle["executive_brief"]["recommendations"]
    if len(recs) >= 4:
        recs[0]["domain"] = pack.rec1_domain
        recs[0]["headline"] = pack.rec1_headline
        recs[0]["recommended_action"] = pack.rec1_action
        recs[0]["expected_business_impact"] = pack.rec1_impact

        recs[1]["domain"] = pack.rec2_domain
        recs[1]["headline"] = pack.rec2_headline
        recs[1]["recommended_action"] = pack.rec2_action
        recs[1]["expected_business_impact"] = pack.rec2_impact

        recs[2]["domain"] = pack.rec3_domain
        recs[2]["headline"] = pack.rec3_headline
        recs[2]["recommended_action"] = pack.rec3_action
        recs[2]["expected_business_impact"] = pack.rec3_impact

        recs[3]["domain"] = pack.rec4_domain
        recs[3]["headline"] = pack.rec4_headline
        recs[3]["recommended_action"] = pack.rec4_action
        recs[3]["expected_business_impact"] = pack.rec4_impact

    return bundle
