from pathlib import Path
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
