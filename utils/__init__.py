from .logging import setup_logging, get_logger, suppress_third_party_logs
from .common import PathValidationError, existing_validation, load_images
from .comparison import (
    cosine_similarity,
    compare_tensors,
    compare_detections,
    compute_iou,
)
from .visualization import draw_detections, save_annotated_image, color_for_class
from .metrics import percentiles, summarize_runs
from .threading import clamp_workers
from .model_utils import (
    OnnxMetadata,
    sha256_of_file,
    file_size_mb,
    inspect_onnx,
    select_providers,
    compare_models,
)
from src import __version__  # noqa: F401  (re-exported for utils.__version__)


__all__ = [
    "setup_logging",
    "get_logger",
    "suppress_third_party_logs",
    "existing_validation",
    "PathValidationError",
    "load_images",
    "draw_detections",
    "save_annotated_image",
    "color_for_class",
    "percentiles",
    "summarize_runs",
    "clamp_workers",
    "OnnxMetadata",
    "sha256_of_file",
    "file_size_mb",
    "inspect_onnx",
    "select_providers",
    "compare_models",
    # utils.comparison — pure NumPy; safe to import without ultralytics.
    "cosine_similarity",
    "compare_tensors",
    "compare_detections",
    "compute_iou",
]
