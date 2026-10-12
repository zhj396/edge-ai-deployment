from pathlib import Path
from typing import Optional, Union
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
