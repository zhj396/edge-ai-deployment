"""Build the COCO 12-class subset (12k train / 1k val) from full COCO 2017.

Selection controls: per-class quotas (person-capped, tail-boosted), phash dedup,
ResNet-18 embedding diversity, person-pollution & multi-label filters, and a
val split that mirrors the realized train distribution (with a look-ahead
quota guard). Emits YOLO-format labels + data.yaml consumed by the training
scripts here and, downstream, by src/sampler.py's CalibrationSampler.
"""

import os
import json
import time
import random
import shutil
import argparse
import logging
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash

import torch
import torchvision.models as models
import torchvision.transforms as transforms


# ===============================
# Logging
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===============================
# Embedding cache
# ===============================
class EmbeddingCache:
    def __init__(self, path):
        self.path = Path(path)
        self.db = np.load(self.path, allow_pickle=True).item() if os.path.exists(self.path) else {}

    def get(self, k):
        return self.db.get(k)

    def set(self, k, v):
        self.db[k] = v

    def save(self):
        # Atomic write (tmp file + os.replace): an interrupted save must not
        # leave a truncated cache that would crash the next build on load.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp.npy")
        np.save(tmp, self.db)
        os.replace(tmp, self.path)


# ===============================
# pHash dedup
# ===============================
# 8-bit popcount lookup table for the vectorized Hamming-distance scan.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class FastDedup:
    """pHash dedup with a vectorized Hamming-distance scan.

    Each 64-bit perceptual hash is packed into one uint64 and stored in a
    single array; a duplicate check is one XOR + byte-wise popcount-table
    lookup over ALL stored hashes (numpy-vectorized), instead of a Python
    loop over ImageHash objects (~10^8 Python-level Hamming computations
    at the 12k scale — the slowest stage of the builder).

    Checks are pure: registration happens in ``add()``, called only on
    ACCEPT, so a candidate rejected by a downstream filter never blocks
    near-duplicates of itself (see docs/TRAINING.md §2).
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self._hashes = np.zeros(0, dtype=np.uint64)

    @staticmethod
    def hash_of(path):
        """Pack the file's pHash into a uint64; None if the image is unreadable."""
        try:
            h = imagehash.phash(Image.open(path))
        except Exception:
            return None
        return int.from_bytes(np.packbits(h.hash.flatten()).tobytes(), "big")

    def is_dup(self, packed):
        if self._hashes.size == 0:
            return False
        xor = np.bitwise_xor(self._hashes, np.uint64(packed))
        dist = _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        return bool((dist < self.threshold).any())

    def add(self, packed):
        self._hashes = np.append(self._hashes, np.uint64(packed))


# ===============================
# Lightweight embedding
# ===============================
class Embedder:
    def __init__(self):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(m.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.feats = []

    def encode(self, path):
        img = Image.open(path).convert("RGB")
        x = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            f = self.model(x).flatten()
        f = f / f.norm()
        return f.numpy()

    def is_similar(self, feat, th=0.9):
        # NOTE: O(N^2) over a growing Python list — np.dot re-converts the list
        # on every call. Acceptable at the 12k scale of this builder; a stacked
        # preallocated array would be the optimization if this grows.
        if not self.feats:
            return False
        sims = np.dot(self.feats, feat)
        return sims.max() > th

    def add(self, feat):
        self.feats.append(feat)


# ===============================
# Parallel candidate prefetch
# ===============================
# The slow per-candidate work (pHash + ResNet-18 encode) is computed in
# worker processes a chunk ahead of the selection loop, while the
# accept/reject decisions stay sequential in the parent: dedup registration
# and embedding-diversity state must evolve in selection order. A chunk is
# at most ~64 wasted encodes when a quota fills mid-chunk.
_PREFETCH_CHUNK = 64
_WORKER = {}


def _init_worker(cache_snapshot):
    # One ResNet-18 per worker; single-threaded intra-op so N workers scale
    # instead of each fighting for all cores.
    torch.set_num_threads(1)
    _WORKER["embedder"] = Embedder()
    _WORKER["cache"] = cache_snapshot


def _prep_candidate(path):
    packed = FastDedup.hash_of(path)
    feat = _WORKER["cache"].get(path)
    if feat is None and packed is not None:
        feat = _WORKER["embedder"].encode(path)
    return packed, feat


# ===============================
# BalancedSampler
# ===============================
class BalancedSampler:
    def __init__(self, classes, size, person_cap=0.18):
        self.classes = classes
        self.size = size
        self.num_classes = len(classes)

        base = size // self.num_classes

        self.quotas = {c: int(base * 1.2) for c in range(self.num_classes)}

        self.person_id = 0
        self.quotas[self.person_id] = int(size * person_cap)

        # Long-tail classes get a larger quota (subset class ids, not COCO ids)
        tail = {1, 3, 4, 5, 6, 7, 9, 10, 11}
        for c in tail:
            self.quotas[c] = int(base * 1.5)

    def build_index(self, coco, classes):
        cat_map = {c["name"]: c["id"] for c in coco["categories"]}
        old2new = {cat_map[n]: i for i, n in enumerate(classes)}

        img2cids = defaultdict(list)
        annos_by_img = defaultdict(list)

        for a in coco["annotations"]:
            if a["category_id"] not in old2new:
                continue
            cid = old2new[a["category_id"]]
            img2cids[a["image_id"]].append(cid)
            annos_by_img[a["image_id"]].append(a)

        return img2cids, annos_by_img, old2new

    def pick_primary(self, cids):
        # The rarest class present in the image is its "primary" class —
        # rare-class images are claimed by the rare class first.
        cnt = Counter(cids)
        return min(cnt, key=lambda x: cnt[x])

    def sample(self, coco, root, img_dir, dedup, embed, cache, is_val=False):
        img2cids, annos_by_img, old2new = self.build_index(coco, self.classes)
        img_map = {img["id"]: img for img in coco["images"]}

        primary_index = defaultdict(list)
        for img_id, cids in img2cids.items():
            p = self.pick_primary(cids)
            primary_index[p].append(img_id)

        selected = set()
        counts = Counter()
        scanned = 0
        last_save = time.time()
        t0 = time.time()

        pool = Pool(cpu_count(), initializer=_init_worker,
                    initargs=(dict(cache.db),))
        try:
            # Largest quota first
            for cls in sorted(self.quotas, key=lambda x: self.quotas[x], reverse=True):

                candidates = primary_index[cls]
                random.shuffle(candidates)
                accepted_before = len(selected)

                pbar = tqdm(total=len(candidates),
                            desc=f"cls {self.classes[cls]}", leave=False)
                for lo in range(0, len(candidates), _PREFETCH_CHUNK):
                    if len(selected) >= self.size or counts[cls] >= self.quotas[cls]:
                        break

                    chunk = candidates[lo:lo + _PREFETCH_CHUNK]
                    paths = [str(root / img_dir / img_map[i]["file_name"])
                             for i in chunk]
                    results = pool.map(_prep_candidate, paths)

                    for img_id, path, (packed, feat) in zip(chunk, paths, results):
                        scanned += 1
                        pbar.update(1)

                        if len(selected) >= self.size or counts[cls] >= self.quotas[cls]:
                            break

                        # pHash is checked here but only REGISTERED on accept
                        # below: candidates rejected by any later filter must
                        # not block near-duplicates of themselves (dedup set
                        # == accepted set).
                        if packed is None or dedup.is_dup(packed):
                            continue

                        # Workers only encode on a cache miss; merge anything
                        # they computed so an interrupt keeps it (cache is
                        # checkpointed below).
                        if feat is not None and path not in cache.db:
                            cache.set(path, feat)

                        if embed.is_similar(feat):
                            continue

                        cids = img2cids[img_id]

                        # ===== person-pollution control =====
                        if 0 in cids and cls != 0 and len(cids) > 2:
                            continue

                        # ===== multi-label control =====
                        if len(set(cids)) > 4:
                            continue

                        # ===== val only: prevent quota blow-up (look-ahead) =====
                        if is_val:
                            would_overflow = False
                            for cid in cids:
                                future = counts[cid] + 1
                                if future > self.quotas.get(cid, 0) * 1.3:
                                    would_overflow = True
                                    break
                            if would_overflow:
                                continue

                        # accept
                        selected.add(img_id)
                        dedup.add(packed)
                        embed.add(feat)

                        for cid in cids:
                            counts[cid] += 1

                        # Periodic embedding-cache checkpoint: an interrupt
                        # must not discard every embedding computed so far
                        # (CPU re-encode of all scanned images on next run).
                        if time.time() - last_save > 300:
                            self.save_cache(cache)
                            last_save = time.time()
                pbar.close()
                logger.info(f"[{'val' if is_val else 'train'}] {self.classes[cls]}: "
                            f"sampled {len(selected) - accepted_before}/{self.quotas[cls]}")

            self.save_cache(cache)
            logger.info(f"[{'val' if is_val else 'train'}] sampling done: scanned {scanned} "
                        f"candidates in {time.time() - t0:.0f}s, accepted {len(selected)}")
            return selected, counts, annos_by_img, old2new
        finally:
            pool.terminate()

    def save_cache(self, cache):
        if cache.db:
            cache.save()


# ===============================
# Builder
# ===============================
class Builder:

    def __init__(self, args):
        self.args = args
        self.root = Path(args.coco_root)
        self.out = Path(args.out_root)
        self.classes = args.classes

        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (self.out / sub).mkdir(parents=True, exist_ok=True)

        self.cache = EmbeddingCache(args.cache)

    def load(self, p):
        logger.info(f"Loading {p} (~1-3 min for train2017)...")
        with open(p) as f:
            return json.load(f)

    def build_dataset(self, coco, size, img_dir):

        logger.info(f"Sampling train split (target {size})...")
        sampler = BalancedSampler(self.classes, size)

        dedup = FastDedup(self.args.phash_th)
        embed = Embedder()

        selected, counts, annos_by_img, old2new = sampler.sample(
            coco, self.root, img_dir, dedup, embed, self.cache
        )

        print("=== Distribution (Train) ===")
        total = sum(counts.values())
        for i, name in enumerate(self.classes):
            c = counts[i]
            pct = c / total * 100 if total else 0.0
            print(f"{name:<15} {c:5d} ({pct:.2f}%)")

        # Rebuild the COCO dicts with dense image/annotation/category ids
        img_map = {img["id"]: img for img in coco["images"]}

        new_imgs = []
        idmap = {}

        for i, oid in enumerate(selected):
            img = img_map[oid].copy()
            img["id"] = i
            idmap[oid] = i
            new_imgs.append(img)

        new_ann = []
        aid = 0
        for oid in selected:
            for a in annos_by_img[oid]:
                new_ann.append({
                    **a,
                    "id": aid,
                    "image_id": idmap[oid],
                    "category_id": old2new[a["category_id"]]
                })
                aid += 1

        return {
            "images": new_imgs,
            "annotations": new_ann,
            "categories": [{"id": i, "name": n} for i, n in enumerate(self.classes)]
        }, counts

    # ===== val: sample to mirror the realized train distribution =====
    def build_val(self, coco, train_counts, size, img_dir):
        logger.info(f"Sampling val split (target {size})...")
        total = sum(train_counts.values())
        ratios = {c: train_counts[c] / total for c in train_counts}

        sampler = BalancedSampler(self.classes, size)
        val_quotas = {}
        for c in ratios:
            val_quotas[c] = max(30, int(ratios[c] * size))

        sampler.quotas = val_quotas

        dedup = FastDedup(self.args.phash_th)
        embed = Embedder()

        selected, counts, annos_by_img, old2new = sampler.sample(
            coco, self.root, img_dir, dedup, embed, self.cache, is_val=True
        )

        print("=== Distribution (Val) ===")
        total_val = sum(counts.values())
        for i, name in enumerate(self.classes):
            c = counts[i]
            pct = c / total_val * 100 if total_val else 0.0
            print(f"{name:<15} {c:5d} ({pct:.2f}%)")

        img_map = {img["id"]: img for img in coco["images"]}

        new_imgs = []
        idmap = {}

        for i, oid in enumerate(selected):
            img = img_map[oid].copy()
            img["id"] = i
            idmap[oid] = i
            new_imgs.append(img)

        new_ann = []
        aid = 0
        for oid in selected:
            for a in annos_by_img[oid]:
                new_ann.append({
                    **a,
                    "id": aid,
                    "image_id": idmap[oid],
                    "category_id": old2new[a["category_id"]]
                })
                aid += 1

        return {
            "images": new_imgs,
            "annotations": new_ann,
            "categories": [{"id": i, "name": n} for i, n in enumerate(self.classes)]
        }

    def convert(self, coco, split):
        """Write YOLO-format `cls cx cy w h` label files (normalized)."""
        label_dir = self.out / "labels" / split
        annos_by_img = defaultdict(list)

        for a in coco["annotations"]:
            annos_by_img[a["image_id"]].append(a)

        for img in coco["images"]:
            w, h = img["width"], img["height"]
            lines = []
            for a in annos_by_img[img["id"]]:
                x, y, bw, bh = a["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                lines.append(f"{a['category_id']} {cx} {cy} {bw / w} {bh / h}")

            if lines:
                with open(label_dir / Path(img["file_name"]).with_suffix(".txt"), "w") as f:
                    f.write("\n".join(lines))

    def copy(self, coco, split, img_dir):
        src = self.root / img_dir
        dst = self.out / "images" / split

        tasks = [(src / img["file_name"], dst / img["file_name"]) for img in coco["images"]]

        with Pool(cpu_count()) as p:
            list(tqdm(p.imap_unordered(_copy_file, tasks), total=len(tasks)))


# ===============================
def _copy_file(p):
    src, dst = p
    if src.exists():
        shutil.copy2(src, dst)


# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", default="data/COCO")
    parser.add_argument("--out_root", default="data/coco_subset_12cls")
    parser.add_argument("--train_size", type=int, default=12000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--phash_th", type=int, default=6)
    parser.add_argument("--cache", default=None,
                        help="Embedding cache path (default: <out_root>/embed_cache.npy)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible subset selection")

    args = parser.parse_args()

    random.seed(args.seed)

    # Cache tied to the output root so the dataset and its cache move
    # together and a cwd change can't silently miss the cache.
    if args.cache is None:
        args.cache = str(Path(args.out_root) / "embed_cache.npy")

    args.classes = [
        "person", "bicycle", "car", "bus", "truck",
        "motorcycle", "dog", "cat", "chair", "bottle",
        "backpack", "traffic light"
    ]

    start_time = time.time()

    b = Builder(args)

    train_coco = b.load(Path(args.coco_root) / "annotations/instances_train2017.json")
    val_coco = b.load(Path(args.coco_root) / "annotations/instances_val2017.json")

    train_set, counts = b.build_dataset(train_coco, args.train_size, "train2017")
    val_set = b.build_val(val_coco, counts, args.val_size, "val2017")

    b.convert(train_set, "train")
    b.copy(train_set, "train", "train2017")

    b.convert(val_set, "val")
    b.copy(val_set, "val", "val2017")

    with open(Path(args.out_root) / "data.yaml", "w") as f:
        yaml.dump({
            "path": str(Path(args.out_root).absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(args.classes),
            "names": args.classes
        }, f)

    b.cache.save()
    build_time = time.time() - start_time
    logger.info(f"DONE in {build_time:.2f} s")


if __name__ == "__main__":
    main()
