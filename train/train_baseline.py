"""YOLOv8s training script — COCO 12-class subset (single-stage SGD baseline).

The ablation reference for the staged run in train_staged/: same subset, same
backbone, no long-tail countermeasures. Ran on Kaggle Tesla T4 x2.

Kaggle: provision with `pip install -r train/requirements.txt` (next to this
script, shared with train_staged/) in the first cell before executing — it pins
`ultralytics==8.4.114`, the release this script's train params were validated
against. Do NOT `pip install -U ultralytics`: that floats to the latest
release, and silent version drift has already broken a staged run on this
project (docs/TRAINING.md §5).
"""

import argparse
import shutil
import random
import yaml
import tarfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import List

import torch
import numpy as np
from ultralytics import YOLO


# ====================== Configuration ======================
@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 80
    imgsz: int = 640
    batch: int = 16
    workers: int = 8

    optimizer: str = "SGD"
    lr0: float = 0.008
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Loss weights
    box: float = 7.5
    cls: float = 1.2
    dfl: float = 1.5

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    fliplr: float = 0.5
    mosaic: float = 0.8
    mixup: float = 0.05
    copy_paste: float = 0.2

    # Misc
    val: bool = True
    plots: bool = True
    save: bool = True
    exist_ok: bool = True
    project: str = "yolov8_coco_subset"
    name: str = "yolov8s_12cls_SGD"


# ====================== Global paths (Kaggle, CLI-overridable) ======================
SRC_DATA = Path("/kaggle/input/datasets/zhjing396/coco-subset/coco_subset_12cls")
DST_DATA = Path("/kaggle/working/coco_subset_12cls")
DATA_YAML = Path("/kaggle/working/coco_subset.yaml")
PRETRAINED = "yolov8s.pt"


# ====================== Helpers ======================
def str2bool(v) -> bool:
    """Parse boolean CLI values (--flag yes/no/true/false/1/0)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def set_seed(seed: int = 42):
    """Pin RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> List[int] | str:
    """Detect the available devices (all GPUs if present, else CPU)."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Detected {device_count} GPU(s)")
        return list(range(device_count))
    return "cpu"


def prepare_dataset(src_data: Path, dst_data: Path, data_yaml: Path):
    """Stage the dataset into dst_data and generate the data YAML."""
    if not dst_data.exists():
        print(f"Copying dataset: {src_data} -> {dst_data}")
        shutil.copytree(src_data, dst_data)
        print("Dataset copy complete")
    else:
        print(f"Dataset already exists at {dst_data}, skipping copy")

    # Generate data.yaml
    data_config = {
        'path': str(dst_data),
        'train': str(dst_data / 'images/train'),
        'val': str(dst_data / 'images/val'),
        'names': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'bus', 4: 'truck',
            5: 'motorcycle', 6: 'dog', 7: 'cat', 8: 'chair', 9: 'bottle',
            10: 'backpack', 11: 'traffic light'
        }
    }

    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)

    print(f"Dataset YAML written: {data_yaml}")


def package_results(save_dir: Path, data_yaml: Path, pretrained: str,
                    pack_name: str = "yolov8s_12cls_baseline"):
    """Package the training results into a tarball for notebook download.

    ``save_dir`` is the trainer's actual output directory (``model.trainer.save_dir``)
    — never reconstruct it from project/name strings, which drift across
    ultralytics versions.
    """
    try:
        pack_dir = Path(pack_name)
        pack_dir.mkdir(parents=True, exist_ok=True)

        # Copy required files
        if data_yaml.exists():
            shutil.copy(data_yaml, pack_dir)

        if Path(pretrained).exists():
            shutil.copy(pretrained, pack_dir)

        # Copy training results
        if save_dir.exists():
            shutil.copytree(save_dir, pack_dir / "runs", dirs_exist_ok=True)
            print(f"Copied training results: {save_dir}")

        # Package
        tar_path = Path(f"{pack_name}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pack_dir, arcname=pack_dir.name)

        print(f"Model packaged: {tar_path}")

        # Kaggle Notebook download link
        try:
            from IPython.display import FileLink, display
            display(FileLink(tar_path))
        except ImportError:
            print("Hint: click the link above in the notebook to download")

    except Exception as e:
        print(f"Packaging failed: {e}")


# ====================== Main ======================
def parse_args() -> argparse.Namespace:
    """CLI overrides for every TrainConfig field (None = keep config default),
    plus paths and runtime options."""
    parser = argparse.ArgumentParser(description=__doc__)

    # One flag per TrainConfig field, defaults left to the dataclass
    for f in fields(TrainConfig):
        if isinstance(f.default, bool):
            parser.add_argument(f"--{f.name}", type=str2bool, nargs="?", const=True,
                                default=None, help=f"default: {f.default}")
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=None,
                                help=f"default: {f.default}")

    # Paths / runtime options outside TrainConfig
    parser.add_argument("--src-data", type=Path, default=SRC_DATA,
                        help="Source dataset (Kaggle input path)")
    parser.add_argument("--dst-data", type=Path, default=DST_DATA,
                        help="Working copy of the dataset")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML,
                        help="Where to write the generated data YAML")
    parser.add_argument("--pretrained", default=PRETRAINED,
                        help="Pretrained weights to fine-tune from")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--pack-name", default="yolov8s_12cls_baseline",
                        help="Name of the results tarball (without extension)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8s COCO 12-class training — start")
    print("=" * 60)

    # Environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Seeds
    set_seed(args.seed)

    # Dataset
    prepare_dataset(args.src_data, args.dst_data, args.data_yaml)

    # Model
    print(f"Loading pretrained model: {args.pretrained}")
    model = YOLO(args.pretrained)

    # Training config — TrainConfig defaults, overridden by any CLI flags
    config_names = {f.name for f in fields(TrainConfig)}
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in config_names}
    config = replace(TrainConfig(), **overrides)
    if overrides:
        print(f"CLI overrides: {overrides}")
    print("Training config:")
    for k, v in asdict(config).items():
        print(f"  {k}={v}")
    device = get_device()

    # asdict(config) mirrors every TrainConfig field into train params
    train_params = dict(asdict(config))
    train_params.update({
        "data": str(args.data_yaml),
        "device": device,
    })

    print("Training...")
    model.train(**train_params)

    # Best weights path — taken from the trainer itself, not reconstructed.
    best_weights = Path(model.trainer.best)
    print(f"Training complete — best weights: {best_weights}")

    # Final validation
    print("Running final validation...")
    final_model = YOLO(best_weights)
    metrics = final_model.val(data=str(args.data_yaml), imgsz=config.imgsz)
    print(metrics)

    # Visualization example
    val_imgs = list((args.dst_data / "images/val").glob("*.jpg"))
    if val_imgs:
        test_img = random.choice(val_imgs)
        print(f"Visualization inference example: {test_img}")
        results = final_model.predict(
            source=test_img,
            conf=0.25,
            iou=0.7,
            save=True
        )
        print(f"Predictions saved to: {results[0].save_dir}")

    # Package results
    package_results(Path(model.trainer.save_dir), args.data_yaml,
                    args.pretrained, args.pack_name)

    print("=" * 60)
    print("All steps finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
