"""FastAPI routes — REST API for pipeline execution and results retrieval."""

import asyncio
import io
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.auth import require_api_key
from app.config import CHARTS_DIR, MAX_UPLOAD_SIZE, RATE_LIMIT_PER_MINUTE, RAW_DIR
from app.db.models import AnalysisResult, InventorySnapshot, PipelineRun, SessionLocal, get_db
from app.pipeline.orchestrator import PipelineOrchestrator
from app.ws.manager import manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["pipeline"], dependencies=[Depends(require_api_key)])
WATCHDOG_STAGING_DIR = RAW_DIR.parent / "watchdog_staging"
WATCHDOG_STAGING_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_STAGING_RETENTION = timedelta(hours=24)


# ---------- Helpers ----------

def _safe_filename(filename: str) -> str:
    """Sanitize a filename: strip path components and dangerous characters."""
    # Take only the basename (strip directory traversal)
    name = Path(filename).name
    # Remove any remaining path separators and null bytes
    name = re.sub(r'[/\\:\x00]', '', name)
    # Strip leading dots to prevent hidden files
    name = name.lstrip('.')
    if not name:
        name = "upload.csv"
    return f"{uuid.uuid4().hex[:8]}_{name}"


def _safe_path(base: Path, *parts: str) -> Path:
    """Resolve path and ensure it stays within base directory."""
    resolved = (base / Path(*parts)).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(403, "Access denied")
    return resolved


def _validate_csv_bytes(contents: bytes) -> None:
    """Validate uploaded CSV bytes for size and basic schema sanity."""
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

    try:
        test_df = pd.read_csv(io.BytesIO(contents), nrows=1)
        if test_df.empty or len(test_df.columns) < 2:
            raise ValueError("CSV must have at least 2 columns and 1 data row")
    except Exception as exc:
        raise HTTPException(400, "Invalid CSV file") from exc


async def _wait_for_stable_csv(path: Path, checks: int = 3, interval_seconds: float = 1.0) -> bytes:
    """Wait until a CSV file stops changing, then return a validated snapshot."""
    if path.suffix.lower() != ".csv":
        raise HTTPException(400, "Only CSV files are accepted")

    stable_observations = 0
    previous_signature: tuple[int, int] | None = None

    for _ in range(checks + 2):
        if not path.exists() or not path.is_file():
            raise HTTPException(404, "CSV file not found")

        stat = path.stat()
        if stat.st_size == 0:
            stable_observations = 0
            previous_signature = None
            await asyncio.sleep(interval_seconds)
            continue

        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == previous_signature:
            stable_observations += 1
        else:
            stable_observations = 1
            previous_signature = signature

        if stable_observations >= checks:
            contents = await asyncio.to_thread(path.read_bytes)
            final_stat = path.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != signature:
                stable_observations = 0
                previous_signature = None
                await asyncio.sleep(interval_seconds)
                continue
            _validate_csv_bytes(contents)
            return contents

        await asyncio.sleep(interval_seconds)

    raise HTTPException(400, f"CSV file {path.name} is still being written or is unstable")


def _prune_watchdog_staging(now: datetime | None = None) -> None:
    """Delete expired watchdog staging files to prevent unbounded growth."""
    cutoff = (now or datetime.now(timezone.utc)) - WATCHDOG_STAGING_RETENTION
    for staged_file in WATCHDOG_STAGING_DIR.glob("*"):
        try:
            if not staged_file.is_file():
                continue
            modified_at = datetime.fromtimestamp(staged_file.stat().st_mtime, tz=timezone.utc)
            if modified_at < cutoff:
                staged_file.unlink()
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("Failed to prune staged watchdog file %s", staged_file, exc_info=True)


async def _run_pipeline_job(
    source_path: Path,
    batch_id: str,
    progress_cb,
    cleanup_path: Path | None = None,
) -> None:
    """Run the pipeline in a worker thread and clean up any staged snapshot afterwards."""
    orchestrator = PipelineOrchestrator(on_progress=progress_cb)
    try:
        await asyncio.to_thread(orchestrator.run, str(source_path), batch_id)
    finally:
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove staged watchdog file %s", cleanup_path, exc_info=True)
        await asyncio.to_thread(_prune_watchdog_staging)


# ---------- Rate limiter ----------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
MAX_STORE_SIZE = 10_000  # Max tracked IPs before eviction


def _evict_rate_limit_store() -> None:
    """Remove stale entries when the store grows too large."""
    now = time.time()
    window_start = now - 60.0
    # Remove timestamps outside the current window and drop empty keys
    keys_to_delete = []
    for ip, timestamps in _rate_limit_store.items():
        _rate_limit_store[ip] = [t for t in timestamps if t > window_start]
        if not _rate_limit_store[ip]:
            keys_to_delete.append(ip)
    for ip in keys_to_delete:
        del _rate_limit_store[ip]


def _check_rate_limit(request: Request):
    """Simple in-memory rate limiter per client IP."""
    if len(_rate_limit_store) >= MAX_STORE_SIZE:
        _evict_rate_limit_store()

    # NOTE: In production behind a reverse proxy, configure the proxy to set
    # X-Real-IP or use uvicorn's --proxy-headers flag for correct client IPs.
    # Do not trust X-Forwarded-For unconditionally — it is trivially spoofable.
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    # Clean old entries for this IP
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, "Rate limit exceeded. Try again later.")
    _rate_limit_store[client_ip].append(now)


# ---------- Fire-and-forget task logging ----------

_running_tasks: set = set()


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback attached to fire-and-forget tasks to log unhandled exceptions."""
    _running_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Pipeline task failed: %s", exc, exc_info=exc)


# ---------- Progress bridge: sync pipeline thread -> async WS ----------

def _make_ws_progress_callback(batch_id: str, loop: asyncio.AbstractEventLoop):
    """Create a sync callback that bridges to async WS broadcast."""
    stage_index = {s: i for i, s in enumerate(PipelineOrchestrator.STAGES)}
    total = len(PipelineOrchestrator.STAGES)

    def callback(stage: str, status: str, data: dict = None):
        if data is None and isinstance(status, str) and status not in ("running", "completed", "failed"):
            data = {"message": status}
            status = "running"
        data = data or {}
        idx = stage_index.get(stage, 0)
        if status == "completed":
            pct = int(((idx + 1) / total) * 100)
        elif status == "running":
            pct = int((idx / total) * 100)
        else:
            pct = 0

        msg_type = "pipeline:failed" if status == "failed" else f"pipeline:{status}"
        if status in ("running", "completed"):
            is_final = idx == total - 1 and status == "completed"
            msg_type = f"pipeline:{'completed' if is_final else 'progress'}"

        msg = manager.build_message(
            msg_type=msg_type,
            batch_id=batch_id,
            stage=stage,
            status=status,
            progress_pct=pct,
            data=_safe_serialize(data),
        )
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_to_batch(batch_id, msg), loop
        )

    return callback


def _safe_serialize(obj):
    """Make data JSON-safe (strip large arrays, convert numpy)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"summary": str(obj)[:500]}


# ---------- Ingest (non-blocking) ----------

@router.post("/ingest")
async def ingest_csv(request: Request, file: UploadFile = File(...)):
    """Upload a CSV file and trigger the full analysis pipeline (non-blocking)."""
    _check_rate_limit(request)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    # Read and validate size
    contents = await file.read()
    _validate_csv_bytes(contents)

    # Save with sanitized filename
    safe_name = _safe_filename(file.filename)
    raw_path = RAW_DIR / safe_name
    raw_path.write_bytes(contents)
    logger.info("File uploaded: %s", raw_path.name)

    # Create batch ID and queued pipeline run record
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    db = SessionLocal()
    try:
        run_record = PipelineRun(batch_id=batch_id, status="queued", source_file=safe_name)
        db.add(run_record)
        db.commit()
    finally:
        db.close()

    # Launch pipeline in background thread via asyncio
    loop = asyncio.get_running_loop()
    progress_cb = _make_ws_progress_callback(batch_id, loop)
    orchestrator = PipelineOrchestrator(on_progress=progress_cb)

    task = asyncio.create_task(
        asyncio.to_thread(orchestrator.run, str(raw_path), batch_id)
    )
    _running_tasks.add(task)
    task.add_done_callback(_log_task_exception)

    return {
        "batch_id": batch_id,
        "status": "queued",
        "message": "Pipeline started. Connect to WS for real-time progress.",
    }


@router.post("/ingest/existing")
async def ingest_existing(request: Request):
    """Trigger pipeline using an existing CSV file in data/raw/."""
    _check_rate_limit(request)

    candidates = list(RAW_DIR.glob("*Dirty*.csv")) + list(RAW_DIR.glob("*dirty*.csv"))
    if not candidates:
        raise HTTPException(404, "No existing dirty CSV found in data/raw/")
    raw_path = sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )[0]
    contents = await _wait_for_stable_csv(raw_path, checks=2, interval_seconds=0.1)
    logger.info("Using existing file: %s", raw_path.name)

    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    db = SessionLocal()
    try:
        run_record = PipelineRun(batch_id=batch_id, status="queued", source_file=raw_path.name)
        db.add(run_record)
        db.commit()
    finally:
        db.close()

    loop = asyncio.get_running_loop()
    progress_cb = _make_ws_progress_callback(batch_id, loop)
    staged_path = WATCHDOG_STAGING_DIR / _safe_filename(raw_path.name)
    await asyncio.to_thread(staged_path.write_bytes, contents)
    task = asyncio.create_task(
        _run_pipeline_job(staged_path, batch_id, progress_cb, cleanup_path=staged_path)
    )
    _running_tasks.add(task)
    task.add_done_callback(_log_task_exception)
    return {"batch_id": batch_id, "status": "queued", "message": "Pipeline started with existing file."}


# ---------- Ingest from path (for watchdog) ----------

async def trigger_pipeline_from_path(file_path: str):
    """Trigger a pipeline run from a file path (called by watchdog)."""
    source_path = _safe_path(RAW_DIR, Path(file_path).name)
    contents = await _wait_for_stable_csv(source_path)
    await asyncio.to_thread(_prune_watchdog_staging)
    original_name = source_path.name
    staged_name = _safe_filename(original_name)
    staged_path = WATCHDOG_STAGING_DIR / staged_name
    await asyncio.to_thread(staged_path.write_bytes, contents)

    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    db = SessionLocal()
    try:
        run_record = PipelineRun(batch_id=batch_id, status="queued", source_file=original_name)
        db.add(run_record)
        db.commit()
    finally:
        db.close()

    await manager.broadcast_global(
        manager.build_message(
            msg_type="watchdog:detected",
            batch_id=batch_id,
            data={"file": original_name},
        )
    )

    loop = asyncio.get_running_loop()
    progress_cb = _make_ws_progress_callback(batch_id, loop)
    task = asyncio.create_task(
        _run_pipeline_job(staged_path, batch_id, progress_cb, cleanup_path=staged_path)
    )
    _running_tasks.add(task)
    task.add_done_callback(_log_task_exception)
    return batch_id


# ---------- Pipeline Status (WS fallback) ----------

@router.get("/runs/{batch_id}/status")
def get_run_status(batch_id: str, db: Session = Depends(get_db)):
    """Quick status poll for a pipeline run (fallback when WS unavailable)."""
    run = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
    if not run:
        raise HTTPException(404, "Run not found")

    analyses = db.query(AnalysisResult).filter(AnalysisResult.batch_id == batch_id).all()
    completed_stages = [a.analysis_type for a in analyses]
    all_stages = PipelineOrchestrator.STAGES
    progress_pct = int((len(completed_stages) / len(all_stages)) * 100) if all_stages else 0

    return {
        "batch_id": run.batch_id,
        "status": run.status,
        "progress_pct": progress_pct,
        "completed_stages": completed_stages,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error_message,
    }


# ---------- Analysis by type ----------

@router.get("/runs/{batch_id}/analysis/{analysis_type}")
def get_analysis(batch_id: str, analysis_type: str, db: Session = Depends(get_db)):
    """Get KPIs + chart_paths for a specific analysis stage."""
    valid_types = ["etl", "stats", "supply_chain", "ml", "capacity", "sensing", "sop"]
    if analysis_type not in valid_types:
        raise HTTPException(400, f"Invalid analysis_type. Must be one of: {valid_types}")

    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == batch_id,
        AnalysisResult.analysis_type == analysis_type,
    ).first()
    if not result:
        raise HTTPException(404, f"Analysis '{analysis_type}' not found for batch {batch_id}")

    return {
        "batch_id": batch_id,
        "analysis_type": analysis_type,
        "kpis": result.result_json,
        "chart_paths": result.chart_paths,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


# ---------- Inventory data for interactive charts ----------

@router.get("/runs/{batch_id}/data")
def get_inventory_data(
    batch_id: str,
    offset: int = 0,
    limit: int = Query(default=1000, le=10000),
    db: Session = Depends(get_db),
):
    """Get inventory snapshot rows for interactive charting."""
    snapshots = db.query(InventorySnapshot).filter(
        InventorySnapshot.batch_id == batch_id
    ).offset(offset).limit(limit).all()
    if not snapshots:
        raise HTTPException(404, "No data found for this batch")

    return [
        {
            "product_id": s.product_id,
            "category": s.category,
            "unit_cost": s.unit_cost,
            "current_stock": s.current_stock,
            "daily_demand_est": s.daily_demand_est,
            "safety_stock_target": s.safety_stock_target,
            "vendor_name": s.vendor_name,
            "lead_time_days": s.lead_time_days,
            "reorder_point": s.reorder_point,
            "stock_status": s.stock_status,
            "inventory_value": s.inventory_value,
        }
        for s in snapshots
    ]


# ---------- Existing endpoints ----------

@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    """List all pipeline runs."""
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(50).all()
    return [
        {
            "batch_id": r.batch_id,
            "status": r.status,
            "source_file": r.source_file,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/runs/{batch_id}")
def get_run(batch_id: str, db: Session = Depends(get_db)):
    """Get details of a specific pipeline run."""
    run = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
    if not run:
        raise HTTPException(404, "Run not found")

    analyses = db.query(AnalysisResult).filter(AnalysisResult.batch_id == batch_id).all()

    return {
        "batch_id": run.batch_id,
        "status": run.status,
        "source_file": run.source_file,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "etl_stats": run.etl_stats,
        "error": run.error_message,
        "analyses": [
            {
                "type": a.analysis_type,
                "results": a.result_json,
                "chart_paths": a.chart_paths,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in analyses
        ],
    }


@router.get("/runs/{batch_id}/kpis")
def get_kpis(batch_id: str, db: Session = Depends(get_db)):
    """Get KPI results for a specific run."""
    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == batch_id,
        AnalysisResult.analysis_type == "stats",
    ).first()
    if not result:
        raise HTTPException(404, "Stats results not found")
    return result.result_json


@router.get("/runs/{batch_id}/charts")
def list_charts(batch_id: str):
    """List all chart files for a batch."""
    charts_dir = _safe_path(CHARTS_DIR, batch_id)
    if not charts_dir.exists():
        raise HTTPException(404, "Charts directory not found")
    charts = sorted(charts_dir.glob("*.png"))
    return [{"name": c.name, "path": c.name} for c in charts]


@router.get("/runs/{batch_id}/charts/{chart_name}")
def get_chart(batch_id: str, chart_name: str):
    """Serve a specific chart image."""
    chart_path = _safe_path(CHARTS_DIR, batch_id, chart_name)
    if not chart_path.exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(str(chart_path), media_type="image/png")


@router.get("/latest/kpis")
def get_latest_kpis(db: Session = Depends(get_db)):
    """Get KPIs from the most recent completed pipeline run."""
    run = db.query(PipelineRun).filter(
        PipelineRun.status == "completed"
    ).order_by(PipelineRun.completed_at.desc()).first()
    if not run:
        raise HTTPException(404, "No completed runs found")

    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == run.batch_id,
        AnalysisResult.analysis_type == "stats",
    ).first()
    if not result:
        raise HTTPException(404, "Stats results not found")

    return {
        "batch_id": run.batch_id,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "kpis": result.result_json,
    }


@router.get("/history/kpis")
def get_kpi_history(limit: int = 20, db: Session = Depends(get_db)):
    """Get KPI history across multiple pipeline runs for trend analysis."""
    limit = max(1, min(limit, 100))
    rows = (
        db.query(PipelineRun, AnalysisResult)
        .join(AnalysisResult, PipelineRun.batch_id == AnalysisResult.batch_id)
        .filter(
            PipelineRun.status == "completed",
            AnalysisResult.analysis_type == "stats",
        )
        .order_by(PipelineRun.completed_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "batch_id": run.batch_id,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "kpis": result.result_json,
        }
        for run, result in rows
    ]
