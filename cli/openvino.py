"""CLI subcommand for OpenVINO conversion + NNCF INT8 quantization.

Workflow
--------
1. ``python main.py openvino convert  --model models/yolov8s_fp32.onnx``
   ONNX → OpenVINO IR (FP16).
2. ``python main.py openvino quantize --model models/yolov8s_fp32.onnx``
   ONNX → FP16 IR → NNCF INT8 IR.
3. ``python main.py openvino run     --model models/yolov8s_int8.xml``
   Load IR and run inference.
"""
import json
from pathlib import Path

from utils import existing_validation, get_logger
from . import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_FP32,
    DEFAULT_MODEL_OPENVINO,
    DEFAULT_MODEL_OPENVINO_INT8,
    DEFAULT_MODEL_PT,
    ExportConfig,
    OpenVINOConvertConfig,
    OpenVINOQuantizeConfig,
    OpenVINORunConfig,
)
from .export import DEFAULT_OPSET, export_model
from src import (
    convert_onnx_to_openvino_ir,
    nncf_available,
    nncf_quantize_openvino,
    openvino_available,
    openvino_conversion_available,
)

logger = get_logger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "openvino",
        help="OpenVINO IR conversion + NNCF INT8 quantization",
    )
    sub = parser.add_subparsers(dest="ov_command", required=True)

    # ---- convert ----
    p_conv = sub.add_parser(
        "convert", help="ONNX → OpenVINO IR (FP16 or FP32)",
    )
    # Resolved at run() time (not via type=existing_validation) so a missing
    # ONNX can fall back to auto-exporting it from the source .pt — the same
    # fresh-clone-friendly pattern cli/__init__.resolve_path_arg documents.
    p_conv.add_argument(
        "--model", type=Path, default=None,
        help=(
            f"Input ONNX file (default: {DEFAULT_MODEL_FP32}). If it is "
            f"missing but {DEFAULT_MODEL_PT} exists, the ONNX is exported "
            "from the .pt first (same defaults as the export command)."
        ),
    )
    p_conv.add_argument("--output", type=Path,
                        default=DEFAULT_MODEL_OPENVINO,
                        help="Output .xml path (.bin written alongside)")
    p_conv.add_argument("--imgsz", type=int, default=640)
    p_conv.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 compression")

    # ---- quantize ----
    p_q = sub.add_parser(
        "quantize", help="ONNX → NNCF INT8 IR",
    )
    p_q.add_argument("--model", type=existing_validation, required=True)
    p_q.add_argument("--output", type=Path,
                     default=DEFAULT_MODEL_OPENVINO_INT8)
    p_q.add_argument("--imgs-input", type=existing_validation,
                     default=DEFAULT_DATA_DIR,
                     help="Calibration dataset dir (needs data.yaml)")
    p_q.add_argument("--imgsz", type=int, default=640)
    # 300 matches the ORT quantize command's default so the two PTQ paths
    # draw from the same-size calibration pool out of the box.
    p_q.add_argument("--max-cal-samples", type=int, default=300)
    p_q.add_argument("--subset-size", type=int, default=64,
                     help="NNCF PTQ subset size, drawn from --max-cal-samples")
    p_q.add_argument("--resnet50", type=Path,
                     default="models/resnet50-11ad3fa6.pth")
    p_q.add_argument("--smooth-quant", action="store_true",
                     help="Enable SmoothQuant. NNCF gates it behind "
                          "model_type=transformer, which this flag passes "
                          "explicitly (preset pinned to performance to keep "
                          "the A/B attributable). Validate the net accuracy "
                          "effect with a consistency run before trusting it.")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run OpenVINO inference")
    p_run.add_argument("--model", type=existing_validation, required=True,
                       help="Path to .xml (auto-loads sibling .bin). "
                            "A .bin is also accepted and redirected to its .xml.")
    p_run.add_argument("--imgs-input", type=existing_validation, required=True)
    p_run.add_argument("--imgsz", type=int, default=640)
    p_run.add_argument(
        "--device", type=str, default="CPU",
        help=(
            "OpenVINO target device, passed to the engine verbatim "
            "(uppercased). Plain plugins: CPU, GPU, AUTO. Compound/"
            "heterogeneous forms are accepted too (deliberately NOT "
            "restricted by a choices list, which cannot express them): "
            "MULTI:CPU,GPU, HETERO:CPU,GPU, MYRIAD (Movidius VPU). A "
            "request no enumerated device can serve (e.g. GPU without an "
            "Intel GPU driver) falls back to CPU at engine init with an "
            "explicit warning naming the unavailable request and a driver "
            "hint (preflight in validate_device_request), instead of an "
            "opaque compile-time exception; it only errors if CPU itself "
            "is unavailable."
        ),
    )
    p_run.add_argument("--num-streams", type=str, default="AUTO")
    p_run.add_argument("--max-imgs", type=int, default=32)
    p_run.add_argument("--batch-size", type=int, default=8)
    p_run.add_argument("--conf", type=float, default=0.25)
    p_run.add_argument("--iou", type=float, default=0.45)
    p_run.add_argument(
        "--data", type=Path,
        default=Path(DEFAULT_DATA_DIR) / "data.yaml",
        help=(
            "data.yaml carrying the class-name list. Needed for an OpenVINO "
            "IR (.xml): ov.convert_model drops the ultralytics 'names' "
            "metadata the ONNX had, so without this the annotated images "
            "show 'class_N' instead of real names. Ignored if the file is "
            "absent (falls back to class_N)."
        ),
    )
    p_run.add_argument("--output-dir", type=Path, default="results/predictions/openvino")

    return parser


def run(args):
    if args.ov_command == "convert":
        _run_convert(args)
    elif args.ov_command == "quantize":
        _run_quantize(args)
    elif args.ov_command == "run":
        _run_inference(args)
    else:
        raise SystemExit(f"Unknown openvino subcommand: {args.ov_command}")


def _resolve_convert_onnx(model_arg, imgsz):
    """Resolve the convert subcommand's ONNX input, auto-exporting if possible.

    - ONNX present → returned validated (same suffix/format checks as before).
    - ONNX missing + ``models/yolov8s.pt`` present → the ONNX is exported
      from the .pt first — same defaults as ``python main.py export`` — then
      returned so the conversion proceeds.
    - Both missing → the standard missing-path error from
      ``existing_validation`` ("Path does not exist: ..." + the ARTIFACTS.md
      hint), i.e. the message this command already produced for a missing
      model.
    """
    onnx_path = (
        Path(model_arg) if model_arg is not None else Path(DEFAULT_MODEL_FP32)
    ).expanduser().resolve()

    if onnx_path.exists():
        return existing_validation(str(onnx_path))

    pt_path = Path(DEFAULT_MODEL_PT).expanduser().resolve()
    if pt_path.exists():
        logger.info(
            "%s not found — exporting it from %s first "
            "(same defaults as `python main.py export`)",
            onnx_path, pt_path,
        )
        export_model(ExportConfig(
            model=pt_path,
            output=onnx_path,
            imgsz=imgsz,
            opset=DEFAULT_OPSET,
            dynamic=True,
            simplify=True,
            device="cpu",
            nms=False,
            validate=True,
        ))
        return onnx_path

    # Neither artifact: keep the standard missing-path message.
    return existing_validation(str(onnx_path))


def _run_convert(args):
    if not openvino_conversion_available():
        raise SystemExit(
            "OpenVINO not installed. Run: pip install -r requirements-openvino.txt"
        )
    onnx_path = _resolve_convert_onnx(args.model, imgsz=args.imgsz)
    cfg = OpenVINOConvertConfig(
        model=onnx_path,
        output=args.output,
        imgsz=args.imgsz,
        fp16=not args.no_fp16,
    )
    out = convert_onnx_to_openvino_ir(
        onnx_path=str(cfg.model),
        output_xml=str(cfg.output),
        fp16=cfg.fp16,
        imgsz=cfg.imgsz,
    )
    logger.info("Conversion complete: %s", out)


def _run_quantize(args):
    if not nncf_available():
        raise SystemExit(
            "NNCF not installed. Run: pip install -r requirements-openvino.txt"
        )
    if not openvino_conversion_available():
        raise SystemExit(
            "OpenVINO conversion tools required for NNCF. "
            "Run: pip install -r requirements-openvino.txt"
        )
    data_yaml = Path(args.imgs_input) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found in {args.imgs_input}")

    cfg = OpenVINOQuantizeConfig(
        model=args.model,
        output=args.output,
        imgs_input=args.imgs_input,
        imgsz=args.imgsz,
        max_cal_samples=args.max_cal_samples,
        subset_size=args.subset_size,
        resnet50=args.resnet50 if Path(args.resnet50).exists() else None,
        smooth_quant=args.smooth_quant,
    )
    out = nncf_quantize_openvino(
        onnx_path=str(cfg.model),
        output_xml=str(cfg.output),
        data_yaml=str(data_yaml),
        imgsz=cfg.imgsz,
        max_samples=cfg.max_cal_samples,
        subset_size=cfg.subset_size,
        smooth_quant=cfg.smooth_quant,
        resnet50=cfg.resnet50,
    )
    logger.info("NNCF INT8 quantization complete: %s", out)


def _run_inference(args):
    if not openvino_available():
        raise SystemExit(
            "OpenVINO not installed. Run: pip install -r requirements-openvino.txt"
        )
    cfg = OpenVINORunConfig(
        model=args.model,
        imgs_input=args.imgs_input,
        imgsz=args.imgsz,
        device=args.device,
        num_streams=args.num_streams,
        max_imgs=args.max_imgs,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        data_yaml=args.data,
        output_dir=args.output_dir,
    )
    from src import OpenVINOEngine

    # Only hand the engine a data_yaml that actually exists; otherwise let
    # the engine fall back to its class_N default (avoids a confusing
    # "could not read" warning when the user has no data.yaml).
    data_yaml = cfg.data_yaml if Path(cfg.data_yaml).exists() else None
    engine = OpenVINOEngine(
        model_path=str(cfg.model),
        device=cfg.device,
        imgsz=cfg.imgsz,
        num_streams=cfg.num_streams,
        data_yaml=data_yaml,
    )
    detections = engine.infer(
        imgs_input=cfg.imgs_input,
        conf=cfg.conf,
        iou=cfg.iou,
        max_imgs=cfg.max_imgs,
        batch_size=cfg.batch_size,
        output_dir=cfg.output_dir,
    )
    for d in detections:
        logger.info(json.dumps(d, ensure_ascii=False))
    logger.info("OpenVINO inference complete")
