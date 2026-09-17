"""Centralized logging configuration.

All modules obtain their logger via ``get_logger(__name__)``; the resulting child loggers inherit
handlers from the root logger configured here.

Notes:
``setup_logging`` configures the **root** logger so that submodule loggers
(``get_logger("src.engine")``, ``get_logger("utils.common")`` ...) all share the same handlers and
level. Calling ``get_logger`` without first calling ``setup_logging`` yields a logger with the
standard library defaults — sufficient for library code, but the CLI entry point will always call
``setup_logging`` first.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_logging_initialized = False

# ---------------------------------------------------------------------------
# Default format
# ---------------------------------------------------------------------------
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-4s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Initialize logging on the **root** logger.

    Idempotent: re-running with ``force=False`` returns immediately so downstream calls (e.g. from
    tests) stay safe. ``force=True`` tears down existing handlers and rebuilds them — used by the
    CLI to pick up a different ``--log-level``.
    """
    global _logging_initialized

    root = logging.getLogger()
    if _logging_initialized and not force:
        return root

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root.setLevel(level)

    # Detach any existing handlers (avoid duplicates on re-init)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _logging_initialized = True
    root.info(
        "Logging initialized | level=%s | file=%s",
        logging.getLevelName(level),
        log_file,
    )
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger. ``name`` should usually be ``__name__``."""
    return logging.getLogger(name if name else "yolov8s_ort")


def suppress_third_party_logs() -> None:
    """Silence noisy third-party loggers so our INFO-level output stays clean."""
    for lib in ("ultralytics", "torch", "onnx", "onnxruntime",
                "matplotlib", "PIL", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)
