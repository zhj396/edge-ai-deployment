"""Latency / throughput statistics helpers.

Centralized so every benchmark surface produces comparable numbers.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def percentiles(values: Iterable[float], qs=(50, 90, 95, 99)) -> Dict[str, float]:
    """Compute percentiles in milliseconds. Returns 0.0 for empty input."""
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    elif values is None:
        arr = np.zeros(0, dtype=np.float64)
    else:
        # Generators don't expose len(); np.fromiter still consumes them. Only
        # fall back to "all zero" if the conversion itself fails (e.g. non-numeric
        # input or an iterator that yields nothing).
        try:
            arr = np.fromiter(values, dtype=np.float64)
        except (TypeError, ValueError):
            return {f"p{q}": 0.0 for q in qs}

    if arr.size == 0:
        return {f"p{q}": 0.0 for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def summarize_runs(per_run_ms: List[float]) -> Dict[str, float]:
    """Return mean/std/min/max/percentiles over per-run timings."""
    if not per_run_ms:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(per_run_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        **percentiles(arr),
    }
