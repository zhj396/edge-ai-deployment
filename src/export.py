import numpy as np
import torch
import onnx
import onnxruntime as ort
import shutil
from pathlib import Path
from ultralytics import YOLO

from utils import get_logger, select_providers

logger = get_logger()


# =========================================================
# ONNX FP32 Model Validation
# =========================================================
def validate_onnx_model(onnx_path: Path, imgsz: int, device: str = "cpu"):
    """Validate an exported ONNX model for structural correctness and runtime feasibility."""
    logger.info("Validating ONNX model...")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    # Best-effort shape validation: the inferred model is intentionally not captured
    # — the runtime probe below reloads from ``onnx_path``. We run infer_shapes only
    # for its validation side-effect (raises on broken graphs); make it non-fatal so
    # models whose shape inference fails (e.g. some externally-rewritten graphs)
    # can still be runtime-validated.

    try:
        onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("Shape inference skipped: %s", e)

    # Runtime test — same provider selection as infer / benchmark / quantize
    # (``utils.select_providers``) so the validation session is configured exactly
    # like the deployed path, not ORT's bare defaults.

    providers = select_providers(device)

    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        logger.exception("ORT Session creation failed")
        raise

    actual_providers = session.get_providers()
    logger.info("ORT Providers: %s", actual_providers)

    # Dynamic shape handling
    inputs = {}

    for inp in session.get_inputs():
        shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
        if len(shape) == 4:
            shape[2], shape[3] = imgsz, imgsz
        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    session.run(None, inputs)
    logger.info("ONNX validation passed")


# =========================================================
# ONNX FP32 Model Export
# =========================================================
def model_export(model_path: Path, output_path: Path, **kwargs):
    """Export YOLOv8 PyTorch model to ONNX FP32 format."""
    model = YOLO(str(model_path))

    if kwargs.get("device", "cpu") == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info("Exporting Model ON CUDA")
    else:
        device = "cpu"
        logger.info("Exporting Model ON CPU")

    exported = model.export(
        format="onnx",
        imgsz=kwargs.get("imgsz", 640),
        opset=kwargs.get("opset", 17),
        dynamic=kwargs.get("dynamic", True),
        simplify=kwargs.get("simplify", True),
        quantize=kwargs.get("quantize", "fp32"),
        device=device,
        nms=kwargs.get("nms", False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(exported)
    if exported_path != output_path:
        # Ultralytics exports next to the source .pt (stem + .onnx); move it to the
        # requested --output. Use shutil.move, not Path.rename: Path.rename raises
        # OSError ("cross-device") when source and dest sit on different filesystems
        # / drives (common on Windows when the .pt lives on one drive and --output
        # on another), whereas shutil.move falls back to a copy+delete across mounts.

        shutil.move(str(exported_path), str(output_path))

    if kwargs.get("validate", True):
        validate_onnx_model(output_path, kwargs.get("imgsz", 640), kwargs.get("device", "cpu"))
    logger.info(f"Export successful: {output_path}")
