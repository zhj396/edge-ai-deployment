import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import logging
import yaml


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/train.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def set_seed(seed=42):
    """Seed the RNGs.

    Note: cudnn stays non-deterministic with benchmark=True on purpose — the
    staged runs favored throughput on the Kaggle 12 h budget (matches
    ``deterministic: false`` in config.yaml). The baseline script makes the
    opposite choice (deterministic=True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return "0,1" if count > 1 else "0"
    return "cpu"


def safe_run(fn, name="operation", retries=3, on_retry=None):
    """Run ``fn`` with exponential-backoff retries (Kaggle session flakiness).

    ``on_retry(attempt)`` fires after a failed attempt and before the next one
    (``attempt`` is the 1-based number of the attempt that just failed), so the
    caller can switch to a cheaper recovery strategy — e.g. resume a training
    stage from the crashed attempt's last checkpoint instead of restarting it
    (see ``checkpoint_retry_runner``).

    Caveat: retries do not distinguish transient failures (CUDA OOM spikes,
    session hiccups) from deterministic ones (bad config) — a config error
    burns all 3 attempts before raising.
    """
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            logger.error(f"[{name}] attempt {i + 1}/{retries} failed: {e}")
            if i == retries - 1:
                raise
            if on_retry is not None:
                on_retry(i + 1)
            time.sleep(2 ** i)
    raise RuntimeError(f"{name} failed after {retries} attempts")


def checkpoint_retry_runner(run_fn, checkpoint, name="operation"):
    """Build the ``(run, on_retry)`` pair that makes ``safe_run`` retries resume.

    ``run_fn`` must accept a single ``resume: bool`` argument. The first attempt
    snapshots the mtime of ``checkpoint`` (which may not exist yet); on a retry,
    ``run_fn`` is called with ``resume=True`` only if the failed attempt actually
    wrote the checkpoint (mtime changed from the snapshot). Without that guard, a
    stale ``last.pt`` left over in an ``exist_ok`` run directory by a *previous*
    run would be mistaken for crash-recovery state, and ultralytics would
    silently "resume" an old, already-complete run instead of retraining.

    If a resumed attempt fails too, the next retry resumes again from the newer
    checkpoint (the baseline snapshot is taken once, before the first attempt).
    """
    path = Path(checkpoint)

    def _mtime():
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    state = {"failed": False, "baseline": _mtime()}

    def run():
        resume = state["failed"] and _mtime() != state["baseline"]
        if state["failed"] and not resume:
            logger.warning(f"[{name}] retry: failed attempt saved no checkpoint "
                           f"({path}) — restarting from scratch")
        return run_fn(resume)

    def on_retry(attempt):
        state["failed"] = True
        logger.info(f"[{name}] attempt {attempt} failed — next attempt resumes "
                    f"from {path} if that attempt checkpointed")

    return run, on_retry


def log_environment():
    """Log runtime versions and warn when ultralytics drifts from the pin.

    Training runs are long and unattended; the pipeline is validated against
    exactly one ultralytics release (``ultralytics==X`` in train/requirements.txt,
    one level above this module — shared by train_baseline.py and train_staged/).
    Surface drift at startup instead of discovering it through a crash 40 epochs
    in, and record the environment for the reproducibility notes in
    docs/TRAINING.md.
    """
    import ultralytics  # heavy dep — function-level import keeps module light

    logger.info(f"Runtime: ultralytics={ultralytics.__version__} "
                f"torch={torch.__version__} numpy={np.__version__} "
                f"cuda_available={torch.cuda.is_available()}")
    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    pinned = None
    try:
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("ultralytics=="):
                pinned = line.split("==", 1)[1].strip()
                break
    except OSError:
        return
    if pinned and ultralytics.__version__ != pinned:
        logger.warning(f"ultralytics {ultralytics.__version__} != pinned {pinned} "
                       f"({req}) — provision with `pip install -r {req}`; "
                       f"results from a drifted environment are not reproducible")


def override_config(cfg, args):
    """Override config.yaml values from command-line arguments."""
    if args.lr0 is not None:
        cfg["optimizer"]["lr0"] = args.lr0
    if args.batch is not None:
        cfg["train"]["batch"] = args.batch
    if args.imgsz is not None:
        cfg["model"]["imgsz"] = args.imgsz
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Per-stage epochs
    if args.epochsA is not None:
        cfg["train"]["epochs"]["A"] = args.epochsA
    if args.epochsB is not None:
        cfg["train"]["epochs"]["B"] = args.epochsB
    if args.epochsC is not None:
        cfg["train"]["epochs"]["C"] = args.epochsC

    # Direct overrides of the key stage parameters (highest priority)
    if args.box is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["box"] = args.box
    if args.cls is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["cls"] = args.cls
    if args.dfl is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["dfl"] = args.dfl
    if args.mosaic is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["mosaic"] = args.mosaic
    if args.copy_paste is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["copy_paste"] = args.copy_paste
    if args.mixup is not None:
        for s in ["A", "B", "C"]:
            cfg["stages"][s]["mixup"] = args.mixup

    # Dotted key=value overrides (most flexible form)
    if args.opts:
        for opt in args.opts:
            if '=' in opt:
                key, value = opt.split('=', 1)
                try:
                    # Coerce to bool/int/float/list when possible.
                    value = yaml.safe_load(value)
                except Exception:
                    pass
                # Set the (possibly nested) value in cfg
                keys = key.split('.')
                d = cfg
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
                logger.info(f"CLI override: {key} = {value}")

    return cfg
