"""
routes/metrics.py — Metrics & Telemetry Endpoints
===================================================

This module manages the in-memory telemetry history of the simulation.
It provides endpoints for the dashboard to pull historical data for
charts, compute averages, and export data as CSV.
"""

import csv
import io
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────
#  In-Memory Metrics Storage
# ──────────────────────────────────────────────────────────────────────
# A simple list acting as a circular buffer (max 1000 entries).
# Contains dicts: {step, reward, waiting_time, queue_length, timestamp}
metrics_history: List[Dict[str, Any]] = []
MAX_HISTORY = 1000

# Metadata about the training baseline (if available)
_fixed_timer_baseline_wait = 0.0


# ──────────────────────────────────────────────────────────────────────
#  Data Schemas
# ──────────────────────────────────────────────────────────────────────

class MetricEntry(BaseModel):
    step: int
    reward: float
    waiting_time: float
    queue_length: float
    timestamp: float


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def _load_baseline():
    """Try to load the training baseline from the docs directory."""
    global _fixed_timer_baseline_wait
    summary_path = Path("docs/training_summary.json")
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
                _fixed_timer_baseline_wait = data.get("baselines", {}).get("fixed_timer", {}).get("mean_waiting_time", 0.0)
        except Exception:
            pass


# Initialize baseline on module load
_load_baseline()


# ──────────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.get("/current", summary="Get the latest tracking snapshot")
async def get_current_metrics() -> Dict[str, Any]:
    """Returns the most recent metrics entry in the history."""
    if not metrics_history:
        return {"status": "empty", "message": "No metrics recorded yet."}
    return metrics_history[-1]


@router.get("/history", summary="Get historical telemetry data")
async def get_metrics_history(
    last_n: int = Query(100, ge=1, le=MAX_HISTORY)
) -> List[Dict[str, Any]]:
    """
    Returns the last 'n' entries from the metrics history.
    Used by the dashboard to populate line charts (reward, wait time).
    """
    return metrics_history[-last_n:]


@router.get("/summary", summary="Get aggregate performance statistics")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Computes and returns high-level statistics:
      - Mean reward (last 100 steps)
      - Mean waiting time (last 100 steps)
      - Best reward achieved so far
      - Total steps simulated session
      - Improvement vs fixed-timer baseline (%)
    """
    if not metrics_history:
        return {"status": "empty", "total_steps": 0}

    last_100 = metrics_history[-100:]
    mean_reward = sum(m["reward"] for m in last_100) / len(last_100)
    mean_wait = sum(m["waiting_time"] for m in last_100) / len(last_100)
    best_reward = max(m["reward"] for m in metrics_history)
    total_steps = metrics_history[-1]["step"]

    # Calculate improvement vs baseline
    improvement = 0.0
    if _fixed_timer_baseline_wait > 0:
        improvement = ((_fixed_timer_baseline_wait - mean_wait) / _fixed_timer_baseline_wait) * 100.0

    return {
        "mean_reward_last_100": round(mean_reward, 4),
        "mean_waiting_time_last_100": round(mean_wait, 2),
        "best_reward": round(best_reward, 4),
        "total_steps": total_steps,
        "improvement_vs_baseline_pct": round(improvement, 2),
        "baseline_wait_reference": round(_fixed_timer_baseline_wait, 2)
    }


@router.post("/record", summary="Internal: Append new metrics (Used by Simulation Loop)")
async def record_metrics(entry: MetricEntry) -> Dict[str, str]:
    """
    Internal endpoint called by the simulation loop after every step
    to update the history. Maintains the 1000-entry limit.
    """
    metrics_history.append(entry.model_dump())
    
    # Maintain circular buffer size
    if len(metrics_history) > MAX_HISTORY:
        metrics_history.pop(0)
    
    return {"status": "recorded"}


@router.get("/export", summary="Download metrics history as CSV")
async def export_metrics_csv():
    """
    Generates a CSV file containing the full metrics history and
    triggers a browser download.
    """
    if not metrics_history:
        return Response(content="No data to export.", media_type="text/plain")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["step", "reward", "waiting_time", "queue_length", "timestamp"])
    writer.writeheader()
    writer.writerows(metrics_history)

    # Return as a downloadable CSV response
    content = output.getvalue()
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=traffic_metrics_export.csv"
        }
    )
