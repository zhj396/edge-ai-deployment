"""Thread / CPU-affinity helper for ``ThreadPoolExecutor`` worker capping.

The project policy for OpenCV / PyTorch / ORT intra-/inter-op thread counts
is *baked in* at the engine and benchmark entry points
(``YOLOv8Engine.__init__`` / ``Benchmark._make_session_options``), and the
preprocess worker pool already disables OpenCV's internal pool via
``cv2.setNumThreads(0)`` at the top of ``preprocess_imgs``. There is no
project-level "configure everything" helper — each component owns its own
defaults so callers can override per-engine via constructor arguments
(e.g. ``YOLOv8Engine(intra_op_threads=..., inter_op_threads=...)``; the
benchmark bakes its values into its ``SessionOptions``).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def clamp_workers(requested: int, ncpu: Optional[int] = None) -> int:
    """Bound a ``ThreadPoolExecutor`` ``max_workers`` argument.

    Why this exists: in the original ``preprocess_imgs`` we honored ``num_workers`` literally, so
    passing ``--num-workers 64`` on a 4-core box started 64 OS threads, which thrashed the scheduler
    and gave worse throughput than 4 workers. Empirically the sweet spot for cv2 letterbox is
    ``n_cpus``-or-2×``n_cpus`` depending on whether cv2's internal pool is disabled; we cap at
    **2×CPU** because the OpenCV pool is already disabled in this pipeline
    (``cv2.setNumThreads(0)`` in ``preprocess_imgs``), so the executor is the only parallel path.

    Returns ``max(1, min(requested, 2*ncpu))`` and warns when clamping.
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
