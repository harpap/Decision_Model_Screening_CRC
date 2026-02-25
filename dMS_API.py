import copy
import os
import threading
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import use_case_new_strategy as strategy_module


app = FastAPI(
    title="Decision Model Screening API",
    description="API endpoint for running use_case_new_strategy and returning analytics.",
    version="1.0.0",
)

cfg_lock = threading.Lock()


class UseCaseRequest(BaseModel):
    file_location: Optional[str] = None
    operational_limit: Optional[Dict[str, Any]] = None
    operational_limit_comp: Optional[Dict[str, Any]] = None
    single_run: Optional[bool] = None
    num_runs: Optional[int] = None
    use_case_new_test: Optional[bool] = Field(default=None, alias="new_test")
    all_variables: Optional[bool] = None
    from_elicitation: Optional[bool] = None
    run_label: str = "run"
    output_dir: str = "logs"
    full_analysis: bool = True
    config_overrides: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_report(path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df.to_dict(orient="index")


def _collect_analytics(output_dir: str) -> Dict[str, Any]:
    report_files = {
        "new_strategy_with_limits": "new_str_w_lim_classification_report.csv",
        "new_strategy_without_limits": "new_str_no_lim_classification_report.csv",
        "new_strategy_multi_run": "new_str_classification_report.csv",
        "old_strategy": "old_str_classification_report.csv",
        "comparison_strategy": "comparison_classification_report.csv",
    }

    reports = {}
    for key, filename in report_files.items():
        report = _read_report(os.path.join(output_dir, filename))
        if report is not None:
            reports[key] = report

    return {
        "reports": reports,
        "generated_files": sorted(os.listdir(output_dir)) if os.path.exists(output_dir) else [],
    }


def _run_use_case(payload: UseCaseRequest) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())[:8]
    run_output_dir = os.path.join(payload.output_dir, f"api_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)

    with cfg_lock:
        original_cfg = copy.deepcopy(strategy_module.cfg)
        try:
            _deep_update(strategy_module.cfg, payload.config_overrides)

            result = strategy_module.use_case_new_strategy(
                file_location=payload.file_location,
                operational_limit=payload.operational_limit
                if payload.operational_limit is not None
                else strategy_module.cfg["operational_limit"],
                operational_limit_comp=payload.operational_limit_comp
                if payload.operational_limit_comp is not None
                else strategy_module.cfg["operational_limit_comp"],
                single_run=payload.single_run
                if payload.single_run is not None
                else strategy_module.cfg["single_run"],
                num_runs=payload.num_runs
                if payload.num_runs is not None
                else strategy_module.cfg["num_runs"],
                use_case_new_test=payload.use_case_new_test
                if payload.use_case_new_test is not None
                else strategy_module.cfg["new_test"],
                all_variables=payload.all_variables
                if payload.all_variables is not None
                else strategy_module.cfg["all_variables"],
                from_elicitation=payload.from_elicitation
                if payload.from_elicitation is not None
                else strategy_module.cfg["from_elicitation"],
                run_label=payload.run_label,
                output_dir=run_output_dir,
                full_analysis=payload.full_analysis,
            )
        finally:
            strategy_module.cfg = original_cfg

    analytics = _collect_analytics(run_output_dir)
    return {
        "run_id": run_id,
        "output_dir": run_output_dir,
        "best_f1_score": result if isinstance(result, dict) else None,
        "analytics": analytics,
    }


@app.post("/analytics/use-case-new-strategy")
async def run_use_case_new_strategy(payload: UseCaseRequest) -> Dict[str, Any]:
    try:
        return await run_in_threadpool(_run_use_case, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
