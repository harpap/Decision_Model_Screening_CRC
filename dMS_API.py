import copy
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import update as update_module
import use_case_new_strategy as strategy_module
from update import update_influence_diagram


app = FastAPI(
    title="Decision Model Screening API",
    description="API endpoint for running use_case_new_strategy and returning analytics.",
    version="1.0.0",
)

cfg_lock = threading.Lock()
jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=2)


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
    build_model: bool = False
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


def _build_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(f"dms_api_build_{uuid.uuid4()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _cleanup_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _build_model_if_requested(payload: UseCaseRequest, run_output_dir: str) -> Optional[str]:
    if not payload.build_model:
        return None

    build_output_dir = os.path.join(run_output_dir, "built_model")
    os.makedirs(build_output_dir, exist_ok=True)

    build_logger = _build_logger(os.path.join(build_output_dir, "build_model.log"))
    try:
        model_type = strategy_module.cfg["model_type"]
        value_function = strategy_module.cfg["value_function"]
        new_test_flag = strategy_module.cfg["new_test"]

        update_influence_diagram(
            model_type=model_type,
            value_function=value_function,
            elicit=strategy_module.cfg["elicit"],
            noise=strategy_module.cfg["noise"],
            calculate_info_values=strategy_module.cfg["calculate_info_values"],
            ref_patient_chars=strategy_module.cfg["patient_chars"],
            new_test=new_test_flag,
            logger=build_logger,
            output_dir=build_output_dir,
        )

        file_suffix = "_new_test" if new_test_flag else ""
        return os.path.join(
            build_output_dir,
            "decision_models",
            f"DM_screening_{value_function}_{model_type}{file_suffix}.xdsl",
        )
    finally:
        _cleanup_logger(build_logger)


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


def _run_use_case(payload: UseCaseRequest, run_id: Optional[str] = None) -> Dict[str, Any]:
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    run_output_dir = os.path.join(payload.output_dir, f"api_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)

    with cfg_lock:
        original_cfg = copy.deepcopy(strategy_module.cfg)
        original_update_cfg = copy.deepcopy(update_module.cfg)
        try:
            _deep_update(strategy_module.cfg, payload.config_overrides)
            _deep_update(update_module.cfg, payload.config_overrides)

            built_file_location = _build_model_if_requested(payload, run_output_dir)
            resolved_file_location = built_file_location or payload.file_location

            result = strategy_module.use_case_new_strategy(
                file_location=resolved_file_location,
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
            update_module.cfg = original_update_cfg

    analytics = _collect_analytics(run_output_dir)
    return {
        "run_id": run_id,
        "output_dir": run_output_dir,
        "built_model": payload.build_model,
        "resolved_file_location": resolved_file_location,
        "best_f1_score": result if isinstance(result, dict) else None,
        "analytics": analytics,
    }


def _execute_async_job(run_id: str, payload: UseCaseRequest) -> None:
    with jobs_lock:
        jobs[run_id]["status"] = "running"
    try:
        result = _run_use_case(payload, run_id=run_id)
        with jobs_lock:
            jobs[run_id]["status"] = "completed"
            jobs[run_id]["result"] = result
    except Exception as exc:
        with jobs_lock:
            jobs[run_id]["status"] = "failed"
            jobs[run_id]["error"] = str(exc)


@app.post("/analytics/use-case-new-strategy")
async def run_use_case_new_strategy(payload: UseCaseRequest) -> Dict[str, Any]:
    try:
        return await run_in_threadpool(_run_use_case, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analytics/use-case-new-strategy/submit")
def submit_use_case_new_strategy(payload: UseCaseRequest) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        jobs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "result": None,
            "error": None,
        }
    executor.submit(_execute_async_job, run_id, payload)
    return {"run_id": run_id, "status": "queued"}


@app.get("/analytics/use-case-new-strategy/{run_id}")
def get_use_case_new_strategy_status(run_id: str) -> Dict[str, Any]:
    with jobs_lock:
        job = jobs.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return job


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
