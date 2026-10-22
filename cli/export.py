from pathlib import Path

from utils import existing_validation, get_logger
from . import DEFAULT_MODEL_FP32, ExportConfig

logger = get_logger(__name__)

# Default opset for both ``export`` and cli/openvino.py's auto-export
# fallback.
DEFAULT_OPSET = 17


def export_model(cfg):
    """Run the export workflow from a populated :class:`ExportConfig`.

    Shared by the ``export`` command and the ``openvino convert``
    auto-export fallback.
    """
    # Deferred: src.export pulls in torch + ultralytics + onnx(onnxruntime);
    # keeping it here lets `main.py --help` and parser construction stay cheap.
    from src import model_export

    logger.info(f"Exporting {cfg.model} -> {cfg.output}")
    model_export(
        model_path=cfg.model,
        output_path=cfg.output,
        imgsz=cfg.imgsz,
        opset=cfg.opset,
        dynamic=cfg.dynamic,
        simplify=cfg.simplify,
        device=cfg.device,
        nms=cfg.nms,
        validate=cfg.validate,
    )
    logger.info("Export complete")


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "export",
        help="Export PyTorch model to ONNX FP32",
        epilog=(
            "Example: python main.py export --model models/yolov8s.pt "
            "--output models/yolov8s_fp32.onnx --opset 17"
        ),
    )
    parser.add_argument(
        "--model", type=existing_validation, required=True,
        help="Input .pt model path",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_FP32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-dynamic", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Export device",
    )
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--no-validate", action="store_true")

    return parser


# =========================================================
# Run export command
# =========================================================
def run(args):
    cfg = ExportConfig(
        model=args.model,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        device=args.device,
        nms=args.nms,
        validate=not args.no_validate,
    )
    export_model(cfg)
