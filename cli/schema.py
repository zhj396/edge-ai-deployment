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
