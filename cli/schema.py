from pathlib import Path
from typing import Union, Dict, Optional
from dataclasses import dataclass


@dataclass
class ExportConfig:
    model: Path
    output: Path
    imgsz: int
    opset: int
    dynamic: bool
    simplify: bool
    device: str
    nms: bool
    validate: bool


@dataclass
class QuantizeConfig:
    model: Path
    output: Path
    imgs_input: Union[str, Path]
    imgsz: int
    batch_size: int
    max_cal_samples: int
    method: str
    resnet50: Optional[Path]
    device: str


@dataclass
class InferConfig:
    model: Path
    backend: str
    imgs_input: Union[str, Path]
    imgsz: int
    max_imgs: int
    batch_size: int
    conf: float
    iou: float
    max_det: int
    save: bool
    output_dir: Path
    device: str


@dataclass
class BenchmarkConfig:
    model: Dict[str, str]
    imgs_input: Union[str, Path]
    imgsz: int
    max_images: int
    batch_size: int
    conf_threshold: float
    iou_threshold: float
    speed_conf: float
    speed_iou: float
    validation: bool
    resnet50: Optional[Path]
    device: str
    warmup: int
    runs: int
    use_sampler: bool
    timestamp_suffix: Optional[str] = None


@dataclass
class ConsistencyConfig:
    # str, not Path: --model1/--model2 may carry an ``openvino(_int8):``
    # backend-selector prefix (see cli/consistency.py).
    model1: Union[str, Path]
    model2: Union[str, Path]
    imgs_input: Union[str, Path]
    mode: str
    max_images: int
    img_sizes: list
    batch_sizes: list
    resnet50: Optional[Path]
    device: str
    atol: float
    rtol: float
    report_path: Path


@dataclass
class OpenVINOConvertConfig:
    model: Path
    output: Path
    imgsz: int
    fp16: bool


@dataclass
class OpenVINOQuantizeConfig:
    model: Path
    output: Path
    imgs_input: Path
    imgsz: int
    max_cal_samples: int
    subset_size: int
    resnet50: Optional[Path]
    smooth_quant: bool


@dataclass
class OpenVINORunConfig:
    model: Path
    imgs_input: Path
    imgsz: int
    device: str
    num_streams: str
    max_imgs: int
    batch_size: int
    conf: float
    iou: float
    output_dir: Path
    # Optional: data.yaml for class names (IR has no ultralytics names meta).
    data_yaml: Optional[Path] = None


@dataclass
class TensorRTBuildConfig:
    model: Path            # the ONNX (Path A) or .pt (Path B)
    output: Path           # the .engine output
    imgsz: int
    precision: str        # fp32 | fp16 | int8
    max_batch: int
    workspace_bytes: int
    device: int
    static: bool = False  # static-batch=1 engine (Turing/sm_75 route)
    # INT8-only, calibrator-ablation path: pin the /model.22/ Detect head to
    # FP32 via layer.precision + OBEY_PRECISION_CONSTRAINTS (mirrors ORT
    # invariant #3). Default-INT8 build uses the QDQ path instead, where the
    # head stays FP32 by QDQ-omission (no calibrator/OBEY) -- proven robust on
    # T4/TRT 10.4 where calibrator+OBEY does NOT hold (ablation wash: head-
    # excluded == whole-net, both collapse detection_fail_rate~50-66%). This
    # flag only matters under --calibrator (the legacy ablation). --no-exclude-
    # head quantizes the whole net (ablation-within-ablation).
    exclude_head: bool = True
    # INT8-only: opt into the legacy calibrator+OBEY INT8 path (plain ONNX +
    # IInt8EntropyCalibrator2). Default False = the QDQ path (reuse ORT's
    # quantize_onnx_to_int8 head-excluded QDQ on the TRT-friendly ONNX; no
    # calibrator; head FP32 by QDQ omission). --calibrator is an ablation that
    # does NOT hold the head FP32 on T4/TRT 10.4 -- produces collapsing engines.
    calibrator: bool = False
    # INT8-only (Path A): calibration image list dir + cache.
    calib_imgs_input: Optional[Path] = None
    calib_cache: Optional[Path] = None
    max_cal_samples: int = 300
    resnet50: Optional[Path] = None
    # INT8 QDQ-path only: calibration method for the quantize_onnx_to_int8 step.
    # MinMax (default) matches the ORT path; Entropy (KL-divergence) may fit
    # tight activation distributions better under the forced SYMMETRIC
    # quantization TRT requires (ActivationSymmetric=True) -- the lever to
    # reduce the symmetric-backbone divergence that drives the ~18-33% recall
    # loss (see docs/TENSORRT.md + CLAUDE.md invariant #16). No-op for the
    # --calibrator ablation (TRT's own IInt8EntropyCalibrator2 runs then).
    calib_method: str = "MinMax"


@dataclass
class TensorRTExportConfig:
    """Path B: .pt -> .engine via Ultralytics native export."""
    model: Path            # the .pt
    output: Path
    imgsz: int
    precision: str         # fp32 | fp16 | int8
    device: int
    data_yaml: Optional[Path] = None  # required for INT8


@dataclass
class TensorRTRunConfig:
    model: Path            # the .engine
    imgs_input: Path
    imgsz: int
    device: int
    max_imgs: int
    batch_size: int
    conf: float
    iou: float
    output_dir: Path
    # Optional: data.yaml for class names (a serialized .engine has no
    # ultralytics names metadata).
    data_yaml: Optional[Path] = None
