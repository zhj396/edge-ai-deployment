import os
import shutil
import yaml
from pathlib import Path
from glob import glob
from collections import Counter
import numpy as np
from utils import logger


def prepare_dataset(src, dst):
    """Copy the subset from the (read-only Kaggle input) src to a writable dst.

    Caveat: if dst exists the copy is skipped entirely — a partially copied
    dst from an interrupted session would be reused as-is. Delete dst to force
    a fresh copy.
    """
    if os.path.exists(dst):
        logger.info(f"Dataset already exists: {dst}")
        return
    logger.info(f"Copying dataset {src} → {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def create_yaml(dst, yaml_path):
    data_config = {
        'path': str(Path(dst).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'bus', 4: 'truck',
            5: 'motorcycle', 6: 'dog', 7: 'cat', 8: 'chair',
            9: 'bottle', 10: 'backpack', 11: 'traffic light'
        }
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Dataset config written: {yaml_path}")


def compute_class_weights(label_dir, num_classes=12, power=0.7, smooth=1e-5):
    """Inverse-frequency class weights from YOLO label files, normalized to mean 1.

    ``power`` controls the weighting strength:
        0.0 → no weighting at all
        0.5 → mild
        0.7 → recommended (used for the staged runs)
        1.0 → strong
    """
    counts = Counter()
    files = glob(f"{label_dir}/*.txt")
    logger.info(f"Found {len(files)} label files")

    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    cls = int(line.split()[0])
                    counts[cls] += 1

    arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=float)
    weights = 1.0 / (np.power(arr + smooth, power))
    weights = weights / weights.mean()          # normalize to mean 1

    logger.info(f"Class counts: {arr.astype(int).tolist()}")
    logger.info(f"Class weights (power={power}): {np.round(weights, 4).tolist()}")
    return weights.tolist()
