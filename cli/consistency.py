from pathlib import Path

from utils import get_logger
from . import (
    ConsistencyConfig,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_FP32,
    DEFAULT_MODEL_PT,
    resolve_path_arg,
)
from src import validate_consistency

logger = get_logger(__name__)


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "consistency",
        help="Validate consistency between two models",
        epilog=(
            "Example: python main.py consistency --model1 models/yolov8s.pt "
            "--model2 models/yolov8s_fp32.onnx --imgs-input data --mode tensor"
        ),
    )
    # ``default=None`` so argparse skips ``existing_validation`` on the documented defaults — see
    # cli/__init__.py for the rationale. The defaults are resolved inside ``run()``.

    parser.add_argument(
        "--model1", type=Path, default=None,
        help=f"Reference model (default: {DEFAULT_MODEL_PT})",
    )
    parser.add_argument(
        "--model2", type=Path, default=None,
        help=f"Comparison model (default: {DEFAULT_MODEL_FP32})",
    )
    parser.add_argument(
        "--imgs-input", type=Path, default=None,
        help=f"Image directory containing data.yaml (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--mode", type=str, choices=["tensor", "detection"], default="tensor")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--img-sizes", type=int, nargs="+", default=[640])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--resnet50", type=Path, default=None,
        help="Optional local ResNet50 checkpoint (default: torchvision auto-download)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Validation device (default: cpu)",
    )
    # Default 1e-3 / 1e-2 covers both PT↔ONNX-FP32 and FP32↔INT8 — INT8 routinely raises p99
    # above 0.05 via noise in low-confidence boxes (advisory tensor stats only), so the stricter
    # 1e-4 / 1e-3 default would false-fail INT8 runs. Tighten explicitly for PT-vs-FP32.
    parser.add_argument(
        "--atol", type=float, default=1e-3,
        help="Absolute tolerance (default: 1e-3 — works for both PT↔FP32 and FP32↔INT8)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-2,
        help="Relative tolerance (default: 1e-2 — works for both PT↔FP32 and FP32↔INT8)",
    )
    parser.add_argument(
        "--report-path", type=Path, default="results/consistency_report.json",
        help="JSON report output path (default: results/consistency_report.json). "
             "Give tensor and detection runs distinct paths (e.g. "
             "results/consistency_tensor.json) so back-to-back runs keep both reports.",
    )

    return parser


# =========================================================
# Run consistency validation command
# =========================================================
def run(args):
    cfg = ConsistencyConfig(
        model1=resolve_path_arg(args.model1, DEFAULT_MODEL_PT),
        model2=resolve_path_arg(args.model2, DEFAULT_MODEL_FP32),
        imgs_input=resolve_path_arg(args.imgs_input, DEFAULT_DATA_DIR),
        mode=args.mode,
        max_images=args.max_images,
        img_sizes=args.img_sizes,
        batch_sizes=args.batch_sizes,
        resnet50=args.resnet50,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
        report_path=args.report_path,
    )
    logger.info(
        f"Starting consistency check: {Path(cfg.model1).name} vs {Path(cfg.model2).name}"
    )
    results = validate_consistency(
        model1=cfg.model1,
        model2=cfg.model2,
        imgs_input=cfg.imgs_input,
        mode=cfg.mode,
        max_images=cfg.max_images,
        img_sizes=cfg.img_sizes,
        batch_sizes=cfg.batch_sizes,
        resnet50=cfg.resnet50,
        device=cfg.device,
        atol=cfg.atol,
        rtol=cfg.rtol,
        report_path=cfg.report_path,
    )
    logger.info("Consistency validation complete")
    # Propagate pass/fail to the process exit code so CI (or any `set -e`
    # driver script) can act on a failed consistency check.
    return 0 if results.get("overall_pass") else 1
