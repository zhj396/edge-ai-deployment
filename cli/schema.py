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


@dataclass
class ConsistencyConfig:
    model1: Path
    model2: Path
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
