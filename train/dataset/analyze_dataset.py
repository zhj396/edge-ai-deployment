"""Instance-distribution analysis for the COCO 12-class subset.

Reads the YOLO label files produced by build_coco_subset.py, reports per-class
instance counts, imbalance ratio, long-tail head/tail split, train-vs-val KL
divergence, and writes distribution plots + results.json to data/analysis_results/.
Its output is the diagnostic that motivated the staged long-tail training
schedule (docs/TRAINING.md §3).
"""

import json
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def load_class_names(dataset_root: Path, fallback: List[str]) -> List[str]:
    """Class names from the subset's ``data.yaml`` (the authoritative source).

    Supports both list-style and ``{id: name}``-mapping ``names`` fields. Falls
    back to the hardcoded 12-class list when the yaml is absent or unreadable,
    so the script stays runnable standalone against any already-built subset.
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return fallback
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        if isinstance(names, list) and names:
            print(f"[INFO] class names loaded from {yaml_path}")
            return [str(n) for n in names]
        print(f"[WARNING] no usable 'names' in {yaml_path} — using fallback list")
    except Exception as e:
        print(f"[WARNING] failed to read {yaml_path}: {e} — using fallback list")
    return fallback


def count_instances(label_dir: Path) -> Counter:
    counts = Counter()
    if not label_dir.exists():
        print(f"[WARNING] path does not exist: {label_dir}")
        return counts

    for txt_file in label_dir.glob("*.txt"):
        try:
            with open(txt_file, "r") as f:
                for line in f:
                    if line.strip():
                        cls = int(line.split()[0])
                        counts[cls] += 1
        except Exception as e:
            print(f"[ERROR] failed to read {txt_file}: {e}")

    return counts


def compute_stats(counts: Counter, num_classes: int) -> Tuple[Dict[int, Dict], int]:
    total = sum(counts.values())
    stats = {}

    for i in range(num_classes):
        count = counts.get(i, 0)
        pct = (count / total * 100) if total > 0 else 0
        stats[i] = {
            "count": count,
            "percentage": pct
        }

    return stats, total


def print_stats(title: str, stats: Dict, class_names: List[str], total: int):
    print(f"\n=== {title} (total instances: {total}) ===")
    print(f"{'ID':>3} {'Class':<15} {'Count':>10} {'Percent':>10}")
    print("-" * 45)

    for i, name in enumerate(class_names):
        count = stats[i]["count"]
        pct = stats[i]["percentage"]
        print(f"{i:>3} {name:<15} {count:>10} {pct:>9.2f}%")

    print("-" * 45)


def compute_imbalance(counts: dict) -> float:
    values = [v for v in counts.values() if v > 0]
    if not values:
        return 0
    return max(values) / min(values)


def compute_long_tail(counts: dict):
    sorted_counts = sorted(counts.values(), reverse=True)
    n = len(sorted_counts)

    head = sorted_counts[: max(1, int(0.2 * n))]
    tail = sorted_counts[int(0.5 * n):]

    return {
        "head_sum": sum(head),
        "tail_sum": sum(tail),
        "head_ratio": sum(head) / sum(sorted_counts) if sorted_counts else 0,
        "tail_ratio": sum(tail) / sum(sorted_counts) if sorted_counts else 0,
    }


def compute_kl_divergence(p_counts: dict, q_counts: dict, num_classes: int):
    p = np.array([p_counts.get(i, 0) for i in range(num_classes)], dtype=float)
    q = np.array([q_counts.get(i, 0) for i in range(num_classes)], dtype=float)

    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)

    kl = np.sum(p * np.log((p + 1e-9) / (q + 1e-9)))
    return float(kl)


# Class-distribution bar chart
def plot_distribution(stats, class_names, title, save_path=None):
    counts = [stats[i]["count"] for i in range(len(class_names))]

    plt.figure()
    plt.bar(range(len(class_names)), counts)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# Train vs Val
def plot_train_val_compare(train_stats, val_stats, class_names, save_path=None):
    x = np.arange(len(class_names))
    train_counts = [train_stats[i]["count"] for i in range(len(class_names))]
    val_counts = [val_stats[i]["count"] for i in range(len(class_names))]

    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, train_counts, width, label="train")
    plt.bar(x + width / 2, val_counts, width, label="val")

    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.title("Train vs Val Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# Long-tail curve
def plot_long_tail(counts, title, save_path=None):
    sorted_counts = sorted(counts.values(), reverse=True)

    plt.figure()
    plt.plot(sorted_counts)
    plt.title(title)
    plt.xlabel("Class Rank")
    plt.ylabel("Instance Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# Fallback so this script stays runnable standalone against any already-built
# subset. The authoritative source is the subset's data.yaml, which is
# preferred whenever it exists next to the labels.
FALLBACK_CLASS_NAMES = ["person", "bicycle", "car", "bus", "truck", "motorcycle",
                        "dog", "cat", "chair", "bottle", "backpack", "traffic light"]


def resolve_class_names(dataset_root: Path) -> List[str]:
    """Read class names from the subset's data.yaml; fall back if absent."""
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        print(f"[INFO] no data.yaml under {dataset_root} — using fallback class names")
        return list(FALLBACK_CLASS_NAMES)

    with open(data_yaml, encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")

    if isinstance(names, dict):
        # YOLO also accepts an {id: name} mapping — normalize to id order.
        names = [names[i] for i in sorted(names)]
    if not names:
        print("[WARNING] data.yaml has no usable 'names' — using fallback class names")
        return list(FALLBACK_CLASS_NAMES)
    return [str(n) for n in names]


class _Tee:
    """Duplicate stdout writes into the text report file.

    Every print from the analysis (per-class tables, diagnostics, long-tail
    dicts) is captured verbatim into the text report next to the plots and
    results.json — committable and diffable, unlike terminal scrollback.
    """

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description="Instance-distribution analysis for the COCO subset")

    parser.add_argument("--root", type=str, default="data/coco_subset_12cls", help="Dataset root")

    args = parser.parse_args()

    ana_results = Path("data/analysis_results")
    ana_results.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(ana_results / "report.txt")

    dataset_root = Path(args.root)

    class_names = load_class_names(dataset_root, FALLBACK_CLASS_NAMES)

    num_classes = len(class_names)

    train_counts = count_instances(dataset_root / "labels/train")
    val_counts = count_instances(dataset_root / "labels/val")

    train_stats, train_total = compute_stats(train_counts, num_classes)
    val_stats, val_total = compute_stats(val_counts, num_classes)

    print_stats("Train set", train_stats, class_names, train_total)
    print_stats("Val set", val_stats, class_names, val_total)

    # ===== Diagnostics =====
    train_imbalance = compute_imbalance(train_counts)
    val_imbalance = compute_imbalance(val_counts)

    train_long_tail = compute_long_tail(train_counts)
    val_long_tail = compute_long_tail(val_counts)

    kl_div = compute_kl_divergence(train_counts, val_counts, num_classes)

    print("\n=== Diagnostics ===")
    print(f"Train imbalance ratio (max/min): {train_imbalance:.2f}")
    print(f"Val imbalance ratio (max/min):   {val_imbalance:.2f}")
    print(f"KL divergence (train || val):    {kl_div:.4f}")

    print("\n[Train long tail]")
    print(train_long_tail)

    print("\n[Val long tail]")
    print(val_long_tail)

    ana_results = Path("data/analysis_results")
    plot_distribution(train_stats, class_names, "Train Distribution", ana_results / "train_dist.png")
    plot_distribution(val_stats, class_names, "Val Distribution", ana_results / "val_dist.png")
    plot_train_val_compare(train_stats, val_stats, class_names, ana_results / "compare.png")
    plot_long_tail(train_counts, "Train Long Tail", ana_results / "train_long_tail.png")

    # JSON output
    output = {
        "train": train_stats,
        "val": val_stats
    }
    with open(ana_results / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] JSON written: {ana_results / 'results.json'}")


if __name__ == "__main__":
    main()
