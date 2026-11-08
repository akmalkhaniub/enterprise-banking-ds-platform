"""FastAPI Backend & Multi-Domain Decision Intelligence Console (Banking, E-Commerce/Retail, Telecom, Marketing/Fintech)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.domain_adapter import adapt_bundle_to_domain
from src.domain_packs import list_domain_packs
from src.pipeline_runner import ARTIFACTS_DIR, MART_DIR, run_full_banking_ds_pipeline


SNAPSHOT_FILE = ARTIFACTS_DIR / "pipeline_snapshot.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_CACHE: Dict[str, Any] = {}


def get_or_build_snapshot(force_rebuild: bool = False) -> Dict[str, Any]:
    """Load precomputed canonical data science snapshot or run the pipeline on demand."""
    global _CACHE
    if not force_rebuild and _CACHE:
        return _CACHE
    if not force_rebuild and SNAPSHOT_FILE.exists():
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
            return _CACHE
    _CACHE = run_full_banking_ds_pipeline(persist_to_disk=True)
    return _CACHE


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_or_build_snapshot(force_rebuild=False)
    yield


app = FastAPI(
    title="Enterprise Multi-Domain Data Science & Decision Intelligence Platform",
    description=(
        "Unified statistical, econometric, and machine learning platform (SARIMAX + GBR Forecasting, "
        "OLS Peer-Adjusted Benchmarking, Customer 360 RFM Clustering, Isolation Forest Anomaly Detection, "
        "Controlled Enterprise GenAI, and Power BI Star-Schema Marts) across Banking, E-Commerce/Retail, "
        "Telecom, and Marketing/Fintech."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


class WhatIfLiquidityRequest(BaseModel):
    domain: str = Field("banking", description="Target industry domain key")
    kibor_delta_bps: int = Field(0, ge=-500, le=500, description="Change in macro rate / elasticity index in bps (-500 to +500)")
    salary_surge_multiplier: float = Field(1.0, ge=0.8, le=1.6, description="Stress multiplier on seasonal peak demand")
    digital_campaign_rollout_pct: float = Field(50.0, ge=0.0, le=100.0, description="% of network nodes with strategic campaign active")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "enterprise-banking-ds-platform"}


@app.get("/api/v1/domains")
def get_supported_domains() -> Dict[str, Any]:
    """Return all supported industry domain packs."""
    return {"domains": list_domain_packs()}


@app.get("/api/v1/dashboard-bundle")
def get_dashboard_bundle(
    domain: str = Query("banking", description="Industry domain key: banking | ecommerce_retail | telecom | marketing_fintech"),
) -> Dict[str, Any]:
    """Return the complete pre-computed data science snapshot adapted to the selected industry domain."""
    base = get_or_build_snapshot(force_rebuild=False)
    return adapt_bundle_to_domain(base, domain_key=domain)


@app.post("/api/v1/pipeline/refresh")
def trigger_pipeline_refresh(
    domain: str = Query("banking", description="Industry domain key"),
) -> Dict[str, Any]:
    """Re-run the entire SQL + ML + GenAI pipeline and refresh the Power BI Star-Schema CSV exports."""
    snapshot = get_or_build_snapshot(force_rebuild=True)
    adapted = adapt_bundle_to_domain(snapshot, domain_key=domain)
    return {"status": "refreshed", "active_domain": adapted["active_domain"], "kpis": adapted["kpis"]}


@app.post("/api/v1/simulate/alco-stress-test")
def simulate_alco_stress_test(req: WhatIfLiquidityRequest) -> Dict[str, Any]:
    """Interactive Multi-Domain What-If Scenario Stress Simulator."""
    base = get_or_build_snapshot(force_rebuild=False)
    snap = adapt_bundle_to_domain(base, domain_key=req.domain)
    base_atm_30d = float(snap["kpis"]["next_30d_atm_forecast_pkr"])
    base_inflow_30d = float(snap["kpis"]["next_30d_deposit_inflow_forecast_pkr"])

    kibor_inflow_factor = 1.0 + (req.kibor_delta_bps / 100.0) * 0.018
    kibor_atm_factor = 1.0 - (req.kibor_delta_bps / 100.0) * 0.006

    additional_rollout_share = max(0.0, (req.digital_campaign_rollout_pct - 50.0) / 100.0)
    campaign_inflow_factor = 1.0 + additional_rollout_share * 0.142

    atm_stress_factor = (0.67 * 1.0 + 0.33 * req.salary_surge_multiplier) * kibor_atm_factor

    stressed_inflow_30d = round(base_inflow_30d * kibor_inflow_factor * campaign_inflow_factor, 2)
    stressed_atm_30d = round(base_atm_30d * atm_stress_factor, 2)
    incremental_inflow = round(stressed_inflow_30d - base_inflow_30d, 2)
    incremental_atm_cash = round(stressed_atm_30d - base_atm_30d, 2)

    return {
        "scenario_inputs": req.model_dump(),
        "active_domain": snap["active_domain"],
        "baseline_30d_deposit_inflow_pkr": base_inflow_30d,
        "stressed_30d_deposit_inflow_pkr": stressed_inflow_30d,
        "incremental_deposit_inflow_pkr": incremental_inflow,
        "baseline_30d_atm_demand_pkr": base_atm_30d,
        "stressed_30d_atm_demand_pkr": stressed_atm_30d,
        "incremental_atm_cash_required_pkr": incremental_atm_cash,
        "recommended_treasury_buffer_pkr": round(stressed_atm_30d * 0.065, 2),
    }


@app.get("/api/v1/bi-mart/download/{table_name}")
def download_bi_mart_csv(table_name: str) -> FileResponse:
    """Download Star-Schema CSV tables directly for Power BI / Tableau ingestion."""
    allowed = {
        "DimBranchBenchmark": "DimBranchBenchmark.csv",
        "DimCustomerSegment": "DimCustomerSegment.csv",
        "FactLiquidityForecast": "FactLiquidityForecast.csv",
        "FactTransactionAnomaly": "FactTransactionAnomaly.csv",
        "FactExecutiveRecommendation": "FactExecutiveRecommendation.csv",
    }
    if table_name not in allowed:
        raise HTTPException(status_code=404, detail=f"Unknown BI Mart table '{table_name}'. Allowed: {list(allowed)}")
    file_path = MART_DIR / allowed[table_name]
    if not file_path.exists():
        get_or_build_snapshot(force_rebuild=True)
    return FileResponse(path=file_path, filename=allowed[table_name], media_type="text/csv")


@app.get("/", response_class=HTMLResponse)
def serve_console() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
