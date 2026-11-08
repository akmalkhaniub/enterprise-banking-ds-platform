"""Multi-Domain Industry Packs & Canonical Data Science isomorphism Registry.

Demonstrates how identical statistical, econometric, and machine learning foundations
(SARIMAX + GBR Forecasting, OLS Peer-Adjusted Residual Benchmarking, RFM K-Means/GMM Clustering,
Isolation Forest + Velocity Z-Score Anomaly Detection, and Controlled GenAI Narratives)
solve high-value business problems across:
1. Banking & Treasury (`banking`)
2. E-Commerce & Omnichannel Retail (`ecommerce_retail`)
3. Telecom & Network Operations (`telecom`)
4. Marketing, Growth & Fintech (`marketing_fintech`)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class DomainPackConfig:
    domain_key: str
    display_name: str
    short_badge: str
    description: str
    currency_unit: str
    # Canonical Entity / Node (e.g. Branch, Fulfillment Store, Cell Tower Region, Acquisition Channel)
    entity_label_singular: str
    entity_label_plural: str
    entity_prefix: str
    # Time-Series Forecasting Targets
    primary_inflow_label: str
    secondary_demand_label: str
    seasonality_spike_label: str
    macro_rate_label: str
    # Hypothesis Testing Business Questions
    hyp1_title: str
    hyp1_question: str
    hyp1_recommendation_template: str
    hyp2_title: str
    hyp2_question: str
    hyp2_recommendation_template: str
    hyp3_title: str
    hyp3_question: str
    # Customer 360 Personas (5 clusters mapped from Dormant -> Salaried/Core -> Digital/Growth -> SME/Power -> Affluent/VIP)
    persona_names: Dict[str, str]
    persona_actions: Dict[str, str]
    portfolio_balance_label: str
    # Anomaly Detection Reason Codes & Channels
    channels: List[str]
    anomaly_reasons: Dict[str, str]
    # Executive Recommendation Templates
    exec_committee_name: str
    rec1_domain: str
    rec1_headline: str
    rec1_action: str
    rec1_impact: str
    rec2_domain: str
    rec2_headline: str
    rec2_action: str
    rec2_impact: str
    rec3_domain: str
    rec3_headline: str
    rec3_action: str
    rec3_impact: str
    rec4_domain: str
    rec4_headline: str
    rec4_action: str
    rec4_impact: str
    # Scale multiplier for realistic domain magnitudes
    value_scale: float


DOMAIN_PACKS: Dict[str, DomainPackConfig] = {
    "banking": DomainPackConfig(
        domain_key="banking",
        display_name="Commercial & Retail Banking",
        short_badge="🏦 Banking & Treasury",
        description="Branch network liquidity forecasting, peer-adjusted branch OLS benchmarking, CASA Customer 360 RFM segmentation, and AML/IBFT anomaly detection.",
        currency_unit="PKR",
        entity_label_singular="Bank Branch",
        entity_label_plural="Bank Branches",
        entity_prefix="BNK-BR",
        primary_inflow_label="30-Day Deposit Inflow Forecast",
        secondary_demand_label="30-Day ATM Cash Replenishment",
        seasonality_spike_label="Salary Window (1st–6th & 28th–31st)",
        macro_rate_label="Policy Rate / KIBOR (bps)",
        hyp1_title="HYP-01-DIGITAL-ONBOARDING",
        hyp1_question="Did the Branch Digital Self-Onboarding Kiosk campaign significantly lift 180-day deposit inflows?",
        hyp1_recommendation_template="Digital kiosk branches achieved +{uplift:.1f}% higher deposit inflow (p={pval:.4f}). Roll out self-onboarding kiosks to remaining control branches.",
        hyp2_title="HYP-02-SALARY-ATM-SURGE",
        hyp2_question="Does the 1st–6th & 28th–31st salary window drive a statistically significant surge in ATM cash demand?",
        hyp2_recommendation_template="Salary window days require +{uplift:.1f}% higher ATM cash replenishment (p={pval:.2e}). Pre-position armored CIT cash 24h before the 28th.",
        hyp3_title="HYP-03-BRANCH-TIER-ANOVA",
        hyp3_question="Does total CASA deposit book vary significantly across Metropolitan, Commercial, Urban, and Semi-Urban tiers?",
        persona_names={
            "Affluent_HNW_Wealth": "Affluent_HNW_Wealth",
            "High_Value_SME_Commercial": "High_Value_SME_Commercial",
            "Digital_Active_Growth": "Digital_Active_Growth",
            "Salaried_Core_Retail": "Salaried_Core_Retail",
            "Dormant_Churn_Risk": "Dormant_Churn_Risk",
        },
        persona_actions={
            "Affluent_HNW_Wealth": "Priority Banking upgrade; pitch Islamic/Mutual Fund wealth products and Premier multi-currency card.",
            "High_Value_SME_Commercial": "Assign SME Relationship Manager; pitch Working Capital Finance & POS merchant acquiring fee waiver.",
            "Digital_Active_Growth": "Pre-approve instant digital personal loan / BNPL credit line via Mobile Banking App push notification.",
            "Salaried_Core_Retail": "Cross-sell Bancassurance & Auto-Invest SIP; incentivize utility bill auto-debit for sticky CASA float.",
            "Dormant_Churn_Risk": "Trigger automated 90-day dormancy win-back campaign (zero-fee RAAST cashback + RM outbound call).",
        },
        portfolio_balance_label="Retail & SME CASA Book",
        channels=["RAAST_P2P", "IBFT_Transfer", "Mobile_App_Bill", "ATM_Withdrawal", "POS_Card", "Branch_OTC"],
        anomaly_reasons={
            "AML_STRUCTURING_NEAR_2.5M_THRESHOLD": "AML_STRUCTURING_NEAR_2.5M_THRESHOLD",
            "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR": "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR",
            "MIDNIGHT_HIGH_VALUE_VELOCITY_DRAIN": "MIDNIGHT_HIGH_VALUE_IBFT_DRAIN",
            "MULTIVARIATE_ISOLATION_FOREST_OUTLIER": "MULTIVARIATE_ISOLATION_FOREST_OUTLIER",
        },
        exec_committee_name="ALCO, Retail Banking & AML Compliance",
        rec1_domain="TREASURY_LIQUIDITY",
        rec1_headline="Pre-Position CIT Armored Cash Ahead of 28th–6th Salary Window",
        rec1_action="Automate Treasury CIT dispatch schedules 24 hours before the 28th across top Metropolitan and Commercial branches while sweeping idle off-peak vault buffers into overnight placements.",
        rec1_impact="Reduces ATM cash-out downtime events by ~65% and captures ~PKR 14.2M annualized overnight yield.",
        rec2_domain="BRANCH_NETWORK",
        rec2_headline="Expand Digital Self-Onboarding Kiosks & Remediate Underperforming Peer Branches",
        rec2_action="Deploy digital account-opening kiosks to the 15 control branches immediately and initiate SME deposit mobilization scorecards for branches below 94% peer-adjusted efficiency.",
        rec2_impact="Projected +PKR 1.85B incremental low-cost CASA deposit mobilization over the next two quarters.",
        rec3_domain="RETAIL_CRM",
        rec3_headline="Execute Automated Win-Back on Dormant CASA & Working-Capital Pitch for SME Segment",
        rec3_action="Trigger automated zero-fee RAAST bill-pay incentives for Dormant_Churn_Risk accounts prior to 180-day dormancy lock, and assign Relationship Managers to High_Value_SME_Commercial accounts.",
        rec3_impact="Prevents dormancy lock on ~35% of at-risk accounts and lifts SME cross-sell penetration from 3.4 to 4.2.",
        rec4_domain="AML_RISK",
        rec4_headline="Enforce Step-Up Biometric Auth on Midnight High-Value IBFT & Sub-2.5M Structuring",
        rec4_action="Route transactions with anomaly_risk_score >= 78.0 (00:00–04:00 IBFT drains and 2.40M–2.50M RAAST structuring) into real-time Cooling-Period hold and biometric re-verification.",
        rec4_impact="Mitigates account takeover (ATO) loss exposure and satisfies regulatory AML/CFT monitoring mandates.",
        value_scale=1.0,
    ),
    "ecommerce_retail": DomainPackConfig(
        domain_key="ecommerce_retail",
        display_name="E-Commerce & Omnichannel Retail",
        short_badge="🛒 E-Commerce & Retail",
        description="Store/Fulfillment GMV & SKU replenishment forecasting, store footprint OLS benchmarking, Shopper RFM/CLV clustering, and refund/coupon abuse detection.",
        currency_unit="PKR",
        entity_label_singular="Fulfillment Hub / Store",
        entity_label_plural="Fulfillment Hubs & Stores",
        entity_prefix="RTL-HUB",
        primary_inflow_label="30-Day Gross Merchandise Value (GMV)",
        secondary_demand_label="30-Day Warehouse Stock Replenishment",
        seasonality_spike_label="Payday & Flash-Sale Campaign Window",
        macro_rate_label="Category Price Elasticity / Promo Index (bps)",
        hyp1_title="HYP-01-ONE-CLICK-CHECKOUT",
        hyp1_question="Did enabling 1-Click Express Checkout & Free-Shipping Bundles significantly lift 180-day store GMV?",
        hyp1_recommendation_template="Express Checkout hubs achieved +{uplift:.1f}% higher GMV inflow (p={pval:.4f}). Enable 1-Click Checkout across all regional fulfillment zones.",
        hyp2_title="HYP-02-PAYDAY-FLASH-SURGE",
        hyp2_question="Do Payday & Mega-Sale windows (1st–6th & 28th–31st) drive a statistically significant surge in inventory dispatch?",
        hyp2_recommendation_template="Payday sale days require +{uplift:.1f}% higher SKU safety stock & last-mile courier capacity (p={pval:.2e}). Pre-stage fast-moving inventory 48h prior.",
        hyp3_title="HYP-03-STORE-TIER-ANOVA",
        hyp3_question="Does active buyer wallet spend vary significantly across Flagship Metro, Mall, Urban, and Express dark-store tiers?",
        persona_names={
            "Affluent_HNW_Wealth": "VIP_Brand_Champions",
            "High_Value_SME_Commercial": "B2B_Wholesale_Buyers",
            "Digital_Active_Growth": "High_Frequency_App_Shoppers",
            "Salaried_Core_Retail": "Payday_Basket_Regulars",
            "Dormant_Churn_Risk": "Lapsed_Cart_Abandoners",
        },
        persona_actions={
            "VIP_Brand_Champions": "Enroll in Early-Access Loyalty Tier; offer complimentary express delivery and personalized luxury bundle previews.",
            "B2B_Wholesale_Buyers": "Assign B2B Key Account Manager; unlock tiered bulk-cart invoicing and Net-15 trade credit line.",
            "High_Frequency_App_Shoppers": "Trigger dynamic push upsell for subscription refills and cross-category gamified loyalty points.",
            "Payday_Basket_Regulars": "Send curated payday replenishment bundles on the 28th of each month with threshold voucher incentives.",
            "Lapsed_Cart_Abandoners": "Trigger automated 60-day win-back WhatsApp/Push voucher with expiring free-shipping token.",
        },
        portfolio_balance_label="Active Customer Wallet & Loyalty Float",
        channels=["Mobile_App_Checkout", "Web_Storefront", "Social_Commerce", "POS_In_Store", "COD_Courier", "B2B_Portal"],
        anomaly_reasons={
            "AML_STRUCTURING_NEAR_2.5M_THRESHOLD": "SERIAL_REFUND_AND_COUPON_STACKING_ABUSE",
            "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR": "BOT_SCALPING_HIGH_VELOCITY_INVENTORY_HOARDING",
            "MIDNIGHT_HIGH_VALUE_VELOCITY_DRAIN": "MIDNIGHT_STOLEN_CARD_GIFT_VOUCHER_DRAIN",
            "MULTIVARIATE_ISOLATION_FOREST_OUTLIER": "MULTIVARIATE_RETURN_FRAUD_OUTLIER",
        },
        exec_committee_name="Merchandising, Supply Chain & Trust/Safety",
        rec1_domain="SUPPLY_CHAIN_INVENTORY",
        rec1_headline="Pre-Stage Fast-Moving SKUs & Last-Mile Fleet Ahead of Payday Sale Peaks",
        rec1_action="Automate regional dark-store SKU replenishment 48 hours before the 28th payday window across Flagship Metro hubs while reducing slow-moving safety stock holding costs.",
        rec1_impact="Cuts stock-out lost sales by ~58% and reduces warehouse working-capital holding cost by ~11%.",
        rec2_domain="STORE_OPERATIONS",
        rec2_headline="Roll Out 1-Click Express Checkout & Remediate Underperforming Dark Stores",
        rec2_action="Enable 1-Click Express Checkout across all remaining control hubs and audit picker/packer throughput at fulfillment centers below 94% peer-adjusted OLS efficiency.",
        rec2_impact="Projected +13.9% lift in regional GMV conversion and faster SLA dispatch.",
        rec3_domain="CRM_RETENTION",
        rec3_headline="Automated Win-Back for Lapsed Cart Abandoners & B2B Wholesale Credit",
        rec3_action="Deploy personalized cart-recovery triggers for Lapsed_Cart_Abandoners and extend automated Net-15 credit limits to B2B_Wholesale_Buyers.",
        rec3_impact="Recovers ~28% of at-risk churned shoppers and increases average order value (AOV) by +19%.",
        rec4_domain="TRUST_AND_SAFETY",
        rec4_headline="Block Automated Bot Scalping, Stolen-Card Gift Drains & Serial Return Abuse",
        rec4_action="Hold orders with anomaly_risk_score >= 78.0 (midnight high-value voucher drains and serial promo stacking) for device-fingerprint & OTP verification before dispatch.",
        rec4_impact="Eliminates chargeback & return-abuse leakage across high-value electronics and voucher categories.",
        value_scale=0.45,
    ),
    "telecom": DomainPackConfig(
        domain_key="telecom",
        display_name="Telecom & Network Intelligence",
        short_badge="📡 Telecom & CVM",
        description="Regional cell-site revenue & bandwidth capacity forecasting, tower peer OLS benchmarking, Subscriber ARPU/Churn RFM clustering, and SIM-box bypass fraud detection.",
        currency_unit="PKR",
        entity_label_singular="Regional Network Cluster",
        entity_label_plural="Regional Network Clusters",
        entity_prefix="TEL-RGN",
        primary_inflow_label="30-Day Prepaid/Postpaid Recharge Inflow",
        secondary_demand_label="30-Day Peak Backhaul Bandwidth Provisioning",
        seasonality_spike_label="Monthly Bundle Renewal & Billing Cycle",
        macro_rate_label="Spectrum / Energy Tariff Index (bps)",
        hyp1_title="HYP-01-5G-DYNAMIC-BUNDLES",
        hyp1_question="Did AI-Personalized 4G/5G App Data Bundles significantly lift 180-day regional ARPU inflows?",
        hyp1_recommendation_template="Personalized bundle regions achieved +{uplift:.1f}% higher recharge inflow (p={pval:.4f}). Expand dynamic app bundle offers network-wide.",
        hyp2_title="HYP-02-BILLING-CYCLE-LOAD-SURGE",
        hyp2_question="Does the monthly salary/bundle-renewal window drive a statistically significant surge in core network load?",
        hyp2_recommendation_template="Renewal window days experience +{uplift:.1f}% higher peak traffic & recharge load (p={pval:.2e}). Auto-scale CDN & packet-core capacity 24h prior.",
        hyp3_title="HYP-03-CLUSTER-TIER-ANOVA",
        hyp3_question="Does subscriber wallet balance & ARPU vary significantly across Metro 5G, Commercial, Urban, and Rural coverage tiers?",
        persona_names={
            "Affluent_HNW_Wealth": "Postpaid_Platinum_Roamers",
            "High_Value_SME_Commercial": "Enterprise_B2B_IoT_Lines",
            "Digital_Active_Growth": "Heavy_5G_Streaming_Youth",
            "Salaried_Core_Retail": "Monthly_Combo_Bundle_Users",
            "Dormant_Churn_Risk": "Multi_SIM_Rotational_Churners",
        },
        persona_actions={
            "Postpaid_Platinum_Roamers": "Pitch zero-bill-shock international roaming pass and priority 5G QoS slicing with family shared quota.",
            "Enterprise_B2B_IoT_Lines": "Assign Enterprise Account Lead; bundle SD-WAN, fleet IoT SIM management, and bulk voice trunks.",
            "Heavy_5G_Streaming_Youth": "Offer midnight gaming/streaming unlimited add-on and mobile wallet bill-pay cashback.",
            "Monthly_Combo_Bundle_Users": "Incentivize auto-renewal subscription via mobile app with +15% bonus data rollover.",
            "Multi_SIM_Rotational_Churners": "Trigger proactive 7-day 'Just-For-You' discounted reactivation bundle before 30-day SIM inactivity.",
        },
        portfolio_balance_label="Active Subscriber Prepaid/Wallet Float",
        channels=["SelfCare_Mobile_App", "USSD_Star_Menu", "Retailer_EasyLoad", "Digital_Wallet_Topup", "Auto_Postpaid_Debit", "Franchise_Store"],
        anomaly_reasons={
            "AML_STRUCTURING_NEAR_2.5M_THRESHOLD": "SIM_BOX_INTERNATIONAL_GREY_TRAFFIC_BYPASS",
            "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR": "WANGIRI_AND_PREMIUM_RATE_NUMBER_PUMPING",
            "MIDNIGHT_HIGH_VALUE_VELOCITY_DRAIN": "MIDNIGHT_ACCOUNT_TAKEOVER_BALANCE_TRANSFER",
            "MULTIVARIATE_ISOLATION_FOREST_OUTLIER": "MULTIVARIATE_CDR_VELOCITY_OUTLIER",
        },
        exec_committee_name="Network Ops, Customer Value Management (CVM) & Revenue Assurance",
        rec1_domain="NETWORK_CAPACITY",
        rec1_headline="Auto-Scale Packet Core & Recharge Gateway Ahead of Bundle Renewal Peaks",
        rec1_action="Pre-allocate elastic cloud-native UPF bandwidth and digital top-up gateway workers 24 hours prior to the 28th–6th monthly renewal surge.",
        rec1_impact="Prevents USSD/App recharge timeouts during peak hours and improves congested cell-site throughput by ~24%.",
        rec2_domain="REGIONAL_OPERATIONS",
        rec2_headline="Expand AI Dynamic Data Bundles & Optimize Underperforming Tower Clusters",
        rec2_action="Roll out dynamic app-based bundle pricing to remaining control clusters and audit power/backhaul OPEX on clusters below 94% OLS efficiency.",
        rec2_impact="Drives +13.9% regional recharge uplift while trimming diesel/grid OPEX per GB.",
        rec3_domain="CVM_CHURN_RETENTION",
        rec3_headline="Pre-Emptive Retention for Multi-SIM Churners & Enterprise IoT Upsell",
        rec3_action="Trigger automated USSD/App 'Just-For-You' win-back bundles when recency exceeds 21 days, and pitch SD-WAN/IoT packages to Enterprise_B2B_IoT_Lines.",
        rec3_impact="Reduces monthly prepaid rotational churn by ~2.4 percentage points and lifts blended ARPU.",
        rec4_domain="REVENUE_ASSURANCE",
        rec4_headline="Quarantine SIM-Box Grey Traffic Bypass & Midnight Balance Drains in Real Time",
        rec4_action="Automatically throttle SIMs scoring >= 78.0 on Isolation Forest CDR velocity rules (high-volume international termination & odd-hour balance transfers).",
        rec4_impact="Protects international interconnect settlement revenue and blocks subscriber wallet takeover fraud.",
        value_scale=0.65,
    ),
    "marketing_fintech": DomainPackConfig(
        domain_key="marketing_fintech",
        display_name="Marketing, Growth & Fintech",
        short_badge="📈 Marketing & Fintech",
        description="Omnichannel acquisition revenue & ad-spend pacing forecasting, campaign ROAS OLS benchmarking, Growth LTV cohort clustering, and click/attribution IVT fraud detection.",
        currency_unit="PKR",
        entity_label_singular="Acquisition Channel / Pod",
        entity_label_plural="Acquisition Channels & Growth Pods",
        entity_prefix="MKT-POD",
        primary_inflow_label="30-Day Attributed Gross Revenue Inflow",
        secondary_demand_label="30-Day Performance Ad Spend & Incentive Budget",
        seasonality_spike_label="Payday Conversion & Mega-Campaign Window",
        macro_rate_label="CPM / Auction Bid Inflation Index (bps)",
        hyp1_title="HYP-01-AI-CREATIVE-PERSONALIZATION",
        hyp1_question="Did Dynamic AI Creative Personalization & Lifecycle Triggers significantly lift 180-day attributed revenue?",
        hyp1_recommendation_template="AI-personalized growth pods achieved +{uplift:.1f}% higher attributed revenue (p={pval:.4f}). Scale dynamic creative optimization across all pods.",
        hyp2_title="HYP-02-PAYDAY-CONVERSION-SURGE",
        hyp2_question="Does the monthly payday window (1st–6th & 28th–31st) drive a statistically significant surge in conversion velocity?",
        hyp2_recommendation_template="Payday conversion windows deliver +{uplift:.1f}% higher daily volume (p={pval:.2e}). Shift performance bid caps forward by 24h to capture high-intent traffic.",
        hyp3_title="HYP-03-CHANNEL-TIER-ANOVA",
        hyp3_question="Does cumulative customer LTV vary significantly across Paid Search, Paid Social, Influencer/Affiliate, and Organic Referral tiers?",
        persona_names={
            "Affluent_HNW_Wealth": "High_LTV_Power_Advocates",
            "High_Value_SME_Commercial": "B2B_Merchant_Partner_Accounts",
            "Digital_Active_Growth": "Organic_Habitual_App_Users",
            "Salaried_Core_Retail": "Promo_Responsive_Converters",
            "Dormant_Churn_Risk": "One_And_Done_Bonus_Hunters",
        },
        persona_actions={
            "High_LTV_Power_Advocates": "Invite to VIP Ambassador Referral Program; suppress paid retargeting ads to save wasted ad spend.",
            "B2B_Merchant_Partner_Accounts": "Enroll in co-op marketing rebates and automated settlement API tier.",
            "Organic_Habitual_App_Users": "Cross-sell annual subscription membership and gamified streak rewards.",
            "Promo_Responsive_Converters": "Target strictly during payday high-conversion windows with capped cashback multipliers.",
            "One_And_Done_Bonus_Hunters": "Exclude from paid acquisition lookalike seeds and enforce strict milestone conditions on sign-up bonuses.",
        },
        portfolio_balance_label="Attributed Customer LTV & Retained Wallet Balance",
        channels=["Meta_Performance_Ads", "Google_Search_UAC", "TikTok_Creator_Video", "Affiliate_Partner_Net", "CRM_Push_WhatsApp", "Organic_Referral_Loop"],
        anomaly_reasons={
            "AML_STRUCTURING_NEAR_2.5M_THRESHOLD": "AFFILIATE_ATTRIBUTION_SPOOFER_AND_REFERRAL_RING",
            "DISPROPORTIONATE_TO_DECLARED_INCOME_ODD_HOUR": "BOT_FARM_INVALID_TRAFFIC_IVT_BUDGET_DRAIN",
            "MIDNIGHT_HIGH_VALUE_VELOCITY_DRAIN": "MIDNIGHT_PROMO_CODE_BRUTE_FORCE_EXPLOIT",
            "MULTIVARIATE_ISOLATION_FOREST_OUTLIER": "MULTIVARIATE_CONVERSION_VELOCITY_OUTLIER",
        },
        exec_committee_name="CMO Growth Committee, Marketing Science & Fraud Prevention",
        rec1_domain="BUDGET_PACING",
        rec1_headline="Front-Load Performance Bid Caps 24h Ahead of Payday Conversion Windows",
        rec1_action="Automate programmatic budget pacing using the Hybrid SARIMAX+GBR forecast (3.02% MAPE) to capture payday conversion surges while throttling off-peak CPA inflation.",
        rec1_impact="Improves blended ROAS by +18% and eliminates end-of-month budget exhaustion.",
        rec2_domain="CHANNEL_BENCHMARKING",
        rec2_headline="Scale Dynamic Creative Optimization & Reallocate Spend from Underperforming Pods",
        rec2_action="Expand AI creative personalization across all growth pods and shift budget away from pods scoring below 94% on peer-adjusted OLS incrementality benchmarking.",
        rec2_impact="Generates +13.9% incremental attributed revenue at flat Customer Acquisition Cost (CAC).",
        rec3_domain="LTV_SEGMENTATION",
        rec3_headline="Suppress Paid Ads on VIP Advocates & Exclude Bonus Hunters from Lookalike Seeds",
        rec3_action="Sync K-Means/GMM RFM segments to ad networks: suppress redundant retargeting on High_LTV_Power_Advocates and block One_And_Done_Bonus_Hunters from seed audiences.",
        rec3_impact="Saves ~22% in wasted retargeting spend and improves 90-day cohort retention.",
        rec4_domain="AD_FRAUD_AND_PROMO_SHIELD",
        rec4_headline="Block Click-Farm IVT, Referral Rings & Midnight Promo Exploits in Real Time",
        rec4_action="Quarantine attribution events and promo redemptions scoring >= 78.0 on Isolation Forest velocity rules before payout or cashback settlement.",
        rec4_impact="Prevents performance marketing budget drain and referral bonus abuse.",
        value_scale=0.35,
    ),
}


def get_domain_pack(domain_key: str = "banking") -> DomainPackConfig:
    """Return the requested DomainPackConfig (defaults to 'banking' if unknown)."""
    return DOMAIN_PACKS.get(domain_key, DOMAIN_PACKS["banking"])


def list_domain_packs() -> List[Dict[str, Any]]:
    """Return metadata for all supported industry domain packs."""
    return [
        {
            "domain_key": k,
            "display_name": v.display_name,
            "short_badge": v.short_badge,
            "description": v.description,
            "entity_label_plural": v.entity_label_plural,
            "primary_inflow_label": v.primary_inflow_label,
            "secondary_demand_label": v.secondary_demand_label,
        }
        for k, v in DOMAIN_PACKS.items()
    ]
