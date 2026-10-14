from pathlib import Path

from utils import existing_validation, get_logger
from . import DEFAULT_DATA_DIR, DEFAULT_MODEL_FP32, DEFAULT_MODEL_INT8, QuantizeConfig
from src import quantize_onnx_to_int8

logger = get_logger(__name__)


# =========================================================
# Add CLI subparser
# =========================================================
def add_parser(subparsers):
    parser = subparsers.add_parser(
        "quantize",
        help="Quantize ONNX FP32 model to INT8",
        epilog=(
            "Example: python main.py quantize --model models/yolov8s_fp32.onnx "
            "--output models/yolov8s_int8.onnx --imgs-input data --max-cal-samples 300"
        ),
    )

    # ``default=None``: the defaults for --model/--imgs-input are validated
    # inside ``run()`` via ``existing_validation``.
    parser.add_argument(
        "--model", type=Path, default=None,
        help=f"Input ONNX FP32 model (default: {DEFAULT_MODEL_FP32})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_INT8)
    parser.add_argument(
        "--imgs-input", type=Path, default=None,
        help=f"Calibration image directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-cal-samples", type=int, default=300,
        help="Maximum number of calibration samples (default: 300)",
    )
    parser.add_argument(
        "--method", type=str, default="MinMax",
        choices=["MinMax", "Entropy"],
        help="Calibration method (default: MinMax)",
    )
    # ``default=None`` selects torchvision auto-download; an explicit path
    # must exist (a missing file raises FileNotFoundError).

    parser.add_argument(
        "--resnet50", type=Path, default=None,
        help="Optional local ResNet50 checkpoint (default: torchvision auto-downloads "
             "pretrained weights into its hub cache; a pre-staged file must be passed "
             "explicitly via this flag)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Quantization device (default: cpu)",
    )

    return parser


# =========================================================
# Run quantization command
# =========================================================
def run(args):
    cfg = QuantizeConfig(
        model=(args.model if args.model is not None
               else existing_validation(DEFAULT_MODEL_FP32)),
        output=args.output,
        imgs_input=(args.imgs_input if args.imgs_input is not None
                    else existing_validation(DEFAULT_DATA_DIR)),
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        max_cal_samples=args.max_cal_samples,
        method=args.method,
        resnet50=args.resnet50,
        device=args.device,
    )
    logger.info(f"Quantizing {cfg.model} -> {cfg.output}")
    quantize_onnx_to_int8(
        onnx_fp32_path=cfg.model,
        onnx_int8_path=cfg.output,
        calibration_imgs=cfg.imgs_input,
        imgsz=cfg.imgsz,
        batch_size=cfg.batch_size,
        max_samples=cfg.max_cal_samples,
        method=cfg.method,
        resnet50=cfg.resnet50,
        device=cfg.device,
    )
    logger.info("Quantization complete")
