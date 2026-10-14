"""Thread helper for ``ThreadPoolExecutor`` worker capping.

This module provides only worker-count capping; callers own any other
thread knobs.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def clamp_workers(requested: int, ncpu: Optional[int] = None) -> int:
    """Bound a ``ThreadPoolExecutor`` ``max_workers`` argument.

    Cap is 2×ncpu; returns ``max(1, min(requested, 2*ncpu))`` and warns when
    clamping.
    """
    ncpu = ncpu if ncpu is not None else (os.cpu_count() or 1)
    cap = max(1, 2 * ncpu)
    requested = max(1, int(requested))
    if requested > cap:
        logger.warning(
            "num_workers=%d exceeds 2*cpu_count(%d)=%d; clamping to %d",
            requested, ncpu, cap, cap,
        )
        return cap
    return requested
