"""Calibration image sampler for INT8 post-training quantization.

Pipeline
--------
1. **Sqrt-frequency stratified sampling** — quotas proportional to ``sqrt(class_count)`` so rare
classes are not drowned by common ones.
2. **Perceptual-hash deduplication** — phash Hamming distance <= threshold drops visually similar
images.
3. **Greedy farthest-first traversal** on L2-normalized ResNet-50 features — picks a
diverse subset from the candidates.

Performance notes
-----------------
* Feature extraction is **batched** (configurable, default 32) so a single ResNet-50 forward
amortizes over many images.
* Farthest-first traversal uses a vectorized distance update so each iteration is O(N) rather than
recomputing the full pairwise matrix.
* Phash dedup uses an O(N^2) loop, which is fine for the few hundred candidate images typical of
calibration; larger sets should pre-bin via BK-tree / LSH.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import imagehash
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from utils import get_logger

logger = get_logger(__name__)


class CalibrationSampler:
    """Diverse-image sampler for INT8 calibration."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        data_yaml: str,
        calibration_size: int = 300,
        phash_threshold: int = 5,
        local_weights: Optional[Path] = None,
        seed: int = 42,
        device: str = "cpu",
        feature_batch_size: int = 32,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.data_yaml = Path(data_yaml)
        self.calibration_size = calibration_size
        self.phash_threshold = phash_threshold
        self.local_weights = local_weights
        self.seed = seed
        self.feature_batch_size = feature_batch_size
        self.cache_path = Path(cache_path) if cache_path else None

        random.seed(seed)
        np.random.seed(seed)

        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self._load_dataset()
        self._load_feature_extractor()

    # dataset
    def _load_dataset(self) -> None:
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        yaml_dir = self.data_yaml.parent
        val_path = cfg["val"]
        self.image_dir = (yaml_dir / val_path).resolve()
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Val image dir not found: {self.image_dir}")

        # Convention: images/val -> labels/val
        # Derive the label dir by replacing the "images" path *segment* rather than doing a
        # substring replace — str(Path) on Windows uses backslashes ('\images\val'), so
        # replace("/images/", "/labels/") silently no-ops and leaves label_dir == image_dir,
        # which makes every label lookup miss.
        parts = list(self.image_dir.parts)
        if "images" in parts:
            i = parts.index("images")
            parts[i] = "labels"
            self.label_dir = Path(*parts)
        else:
            # No "images" segment to swap — fall back to the sibling "labels" directory
            # next to the image dir's parent, if it exists.

            self.label_dir = self.image_dir.parent.parent / "labels" / self.image_dir.name
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")

        names = cfg["names"]
        self.class_names = (
            names if isinstance(names, dict)
            else {i: n for i, n in enumerate(names)}
        )

        self.image_to_classes: dict = defaultdict(set)
        self.class_to_images: dict = defaultdict(set)

        image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_SUFFIXES
        )

        for img_path in image_paths:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            classes = self._parse_label(label_path)
            if not classes:
                continue
            self.image_to_classes[img_path] = classes
            for c in classes:
                self.class_to_images[c].add(img_path)

        self.valid_images = list(self.image_to_classes.keys())
        logger.info("Valid images: %d", len(self.valid_images))

    @staticmethod
    def _parse_label(label_path: Path) -> set:
        """Parse a YOLO-format ``cls cx cy w h`` label file."""
        classes: set = set()
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    classes.add(int(parts[0]))
        except Exception:
            pass
        return classes

    # feature model
    def _load_feature_extractor(self) -> None:
        """Load ResNet-50 for farthest-first feature extraction.

        Contract:
        * ``local_weights=None`` → use ``ResNet50_Weights.DEFAULT`` (auto-downloads on first use,
        then cached by torchvision).
        * ``local_weights=Path`` → load that checkpoint **strictly**; raise if the file is missing
        or un-loadable. A supplied-but-missing path must never fall through to a
        *randomly-initialized* ResNet-50 — random features would silently destroy calibration-set
        representativeness.

        ``ResNet50_Weights.DEFAULT.transforms()`` is always used for preprocessing regardless of the
        weight source — only the parametric
        ``weights=`` argument to ``resnet50(...)`` varies.
        """
        logger.info("Loading ResNet-50 feature extractor...")
        weights = ResNet50_Weights.DEFAULT  # canonical — also the source of transforms()

        if self.local_weights is not None:
            weights_path = Path(self.local_weights)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"--resnet50 specified but file is missing: {weights_path}. "
                    f"Either drop a real checkpoint there, or omit --resnet50 "
                    f"to let torchvision download ResNet50_Weights.DEFAULT."
                )
            try:
                state_dict = torch.load(
                    str(weights_path), map_location="cpu", weights_only=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load --resnet50 checkpoint at {weights_path}: {e}"
                ) from e
            model = resnet50(weights=None)
            model.load_state_dict(state_dict)
            logger.info("Loaded local weights from %s", weights_path)
        else:
            model = resnet50(weights=weights)
            logger.info("Using torchvision ResNet50_Weights.DEFAULT (auto-download if uncached)")

        model.fc = torch.nn.Identity()
        self.feature_model = model.to(self.device).eval()
        self.transform = weights.transforms()

    # sampling
    def sample(self, use_cache: bool = True) -> List[Path]:
        """Run the full sampling pipeline and return the chosen image paths."""
        if use_cache and self.cache_path and self.cache_path.exists():
            cached = self._load_cache()
            if cached:
                logger.info("Loaded %d cached calibration images", len(cached))
                return cached

        candidates = self._stratified_sampling()
        logger.info("After stratified sampling: %d", len(candidates))

        candidates = self._deduplicate(candidates)
        logger.info("After phash dedup: %d", len(candidates))

        final = self._diverse_sampling(candidates, self.calibration_size)
        logger.info("Final calibration images: %d", len(final))

        if use_cache and self.cache_path:
            self._save_cache(final)
        return final

    # cache helpers
    def _save_cache(self, paths: List[Path]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump([str(p) for p in paths], f, indent=2)

    def _load_cache(self) -> Optional[List[Path]]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            paths = [Path(p) for p in data]
            if all(p.exists() for p in paths):
                return paths
            logger.warning("Cache references missing files; rebuilding")
            return None
        except Exception:
            return None

    # stratified stage
    def _stratified_sampling(self) -> List[Path]:
        class_freq = {c: len(imgs) for c, imgs in self.class_to_images.items()}
        sqrt_weights = {c: np.sqrt(f) for c, f in class_freq.items()}
        total = sum(sqrt_weights.values()) or 1.0

        # Raw proportional quota (float), floored to int with a floor of 1 so every class is
        # represented even when sqrt-weighted to <1 (rare classes). The floor can push the sum above
        # calibration_size, so normalize back down: greedily trim the most-represented classes (>=2)
        # until within budget. This keeps the logged quota table honest and avoids handing the
        # downstream farthest-first stage a grossly oversized candidate set. If num_classes >
        # calibration_size, every class is stuck at floor 1 and the sum cannot drop — that case is
        # logged and farthest-first trims to the target.

        raw = {
            c: self.calibration_size * w / total
            for c, w in sqrt_weights.items()
        }
        class_quota = {c: max(1, int(v)) for c, v in raw.items()}

        surplus = sum(class_quota.values()) - self.calibration_size
        if surplus > 0:
            order = sorted(
                class_quota, key=lambda c: class_quota[c], reverse=True
            )
            while surplus > 0:
                reduced = False
                for c in order:
                    if class_quota[c] > 1:
                        class_quota[c] -= 1
                        surplus -= 1
                        reduced = True
                        if surplus == 0:
                            break
                if not reduced:
                    break  # every class at floor 1; cannot trim further

        quota_total = sum(class_quota.values())
        logger.info(
            "========== Class Quota (target=%d, allocated=%d) ==========",
            self.calibration_size, quota_total,
        )
        for cls_id, quota in sorted(class_quota.items()):
            logger.info("%-15s: %d",
                        self.class_names.get(cls_id, str(cls_id)), quota)
        if quota_total > self.calibration_size:
            logger.warning(
                "Quota total %d > calibration_size %d (num_classes exceeds "
                "calibration_size); farthest-first will trim to the target.",
                quota_total, self.calibration_size,
            )

        selected: set = set()
        for cls_id, quota in sorted(class_quota.items()):
            # sorted() is load-bearing: class_to_images values are sets, and
            # list(set) iteration order depends on PYTHONHASHSEED — without the
            # sort, the seeded shuffle below draws from a different input order
            # in every process and two runs select different calibration sets.
            imgs = sorted(self.class_to_images[cls_id])
            random.shuffle(imgs)
            selected.update(imgs[:quota])
        return sorted(selected)

    # phash dedup stage
    @staticmethod
    def _compute_phash(img_path: Path):
        try:
            with Image.open(img_path) as img:
                return imagehash.phash(img.convert("RGB"))
        except Exception as e:
            # Unreadable images are dropped from the calibration set by _deduplicate; the explicit
            # skip log keeps that reduction traceable, consistent with _extract_features_batched.
            logger.info("Skip phash-unreadable %s: %s", img_path.name, e)
            return None

    def _deduplicate(self, imgs: List[Path]) -> List[Path]:
        unique: List[Path] = []
        hashes = []
        for img_path in imgs:
            h = self._compute_phash(img_path)
            if h is None:
                continue
            is_dup = any(abs(h - oh) <= self.phash_threshold for oh in hashes)
            if not is_dup:
                unique.append(img_path)
                hashes.append(h)
        return unique

    # feature extraction
    @torch.no_grad()
    def _extract_features_batched(self, imgs: List[Path]) -> Tuple[np.ndarray, List[Path]]:
        """Run ResNet-50 in mini-batches; return (features [N,D], valid paths)."""
        features: List[np.ndarray] = []
        valid: List[Path] = []

        batch_imgs: List[torch.Tensor] = []
        batch_paths: List[Path] = []

        def flush():
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs).to(self.device, non_blocking=True)
            f = self.feature_model(x)
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            features.append(f.cpu().numpy())
            valid.extend(batch_paths)
            batch_imgs.clear()
            batch_paths.clear()

        for idx, p in enumerate(imgs):
            try:
                with Image.open(p) as im:
                    t = self.transform(im.convert("RGB"))
                batch_imgs.append(t)
                batch_paths.append(p)
            except Exception as e:
                logger.info("Skip %s: %s", p.name, e)
                continue

            if len(batch_imgs) >= self.feature_batch_size:
                flush()
        flush()

        if not features:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.concatenate(features, axis=0), valid

    # diverse sampling
    def _diverse_sampling(self, imgs: List[Path], max_samples: int) -> List[Path]:
        if len(imgs) <= max_samples:
            return imgs

        logger.info("Extracting feature embeddings...")
        features, valid_imgs = self._extract_features_batched(imgs)
        if features.shape[0] <= max_samples:
            return valid_imgs

        selected = self._greedy_farthest(features, max_samples)
        return [valid_imgs[i] for i in selected]

    @staticmethod
    def _greedy_farthest(features: np.ndarray, k: int) -> List[int]:
        """Greedy farthest-first traversal.

        Each iteration updates a min-distance vector in O(N), so total
        cost is O(kN) instead of the naive O(kN^2).
        """
        n = features.shape[0]
        if n <= k:
            return list(range(n))

        selected: List[int] = [random.randint(0, n - 1)]
        # Min distance from any candidate to the selected set so far.
        last = features[selected[0]]
        distances = np.linalg.norm(features - last, axis=1)
        distances[selected[0]] = -1.0

        for _ in range(k - 1):
            next_idx = int(np.argmax(distances))
            selected.append(next_idx)
            new = features[next_idx]
            d = np.linalg.norm(features - new, axis=1)
            distances = np.minimum(distances, d)
            distances[next_idx] = -1.0
        return selected
