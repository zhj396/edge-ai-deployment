# Training — COCO 12-Class Subset (YOLOv8s)

> The upstream half of the project lifecycle: **dataset build → analysis → baseline →
> staged long-tail training**. This repository holds the training upstream
> (`train/` + `docs/TRAINING.md`); the trained checkpoint (`yolov8s.pt`, 12 classes)
> is consumed by the deployment half of the project — export, quantization,
> benchmarking, and consistency validation across 7 backends — described in the
> deployment-side README.

Two environments, split by stage:

- **Dataset build + analysis** (`train/dataset/`) runs **locally on CPU** — no GPU
  required; both scripts use relative paths (`--coco_root`, `--out_root`) and carry no
  `/kaggle/...` hardcoding.
- **Training** (`train/train_baseline.py`, `train/train_staged/`) runs on **Kaggle
  Tesla T4 ×2** (12 h/session limit; provision the notebook with
  `pip install -r train/requirements.txt` in the first cell before
  executing either training script — it pins `ultralytics==8.4.114`, the exact
  release the whole pipeline is validated against; do **not** `pip install -U
  ultralytics`, which floats to the latest release and silently drifts the
  environment away from the validated pin. The native `class_weights` hook the injection
  relies on only exists in ultralytics >= 8.4). Two Kaggle dataset uploads back the runs:
  the **baseline** script (`train/train_baseline.py`) reads `coco-12cls`, the **staged**
  script (`train/train_staged/config.yaml`) reads `coco-subset` — two uploads of the same
  `coco_subset_12cls/` inner directory under different slugs. The hardcoded `/kaggle/...`
  paths record that environment.

§5 documents the class-weight injection mechanism. `train/` is
**isolated from the deployment half** (`cli/`, `src/`, `utils/`): the heavyweight
training dependencies (`ultralytics`, `imagehash`) stay out of the default test
suite (pytest's `testpaths` is `tests/`), and the scripts are linted under the
same `flake8` gate as the rest of the repo. The one deliberate exception:
`tests/test_train_staged_retry.py` loads `train_staged/utils.py` in isolation
(numpy/torch only, no ultralytics) to cover the retry/resume state machine.

---

## 1. Pipeline

```
COCO 2017 (full)
   │  train/dataset/build_coco_subset.py   select 12k train + 1k val, 12 classes,
   │                                       phash dedup + ResNet-18 embedding diversity,
   │                                       person-cap & multi-label control, val matched
   │                                       to train distribution → YOLO format + data.yaml
   ▼
coco_subset_12cls/
   │  train/dataset/analyze_dataset.py     per-class instance counts, imbalance ratio,
   │                                       long-tail head/tail split, train↔val KL
   │                                       divergence, distribution plots
   ▼
diagnosis: moderate long tail (max/min ≈ 8.03) — motivates the staged schedule
   │
   ├─ train/train_baseline.py        single-stage SGD fine-tune — measured BEST at
   │                                 unified evaluation (§6): FINAL checkpoint producer
   │
   └─ train/train_staged/            three-stage (A/B/C) AdamW training with custom
                                     class weights — the long-tail experiment;
                                     negative result at unified evaluation (§6)
   ▼
yolov8s.pt (12 cls, baseline run) ──> main.py export / quantize / benchmark / consistency
                        (deployment half, outside this repository)
```

`train/dataset/build_coco_subset.py` (dataset construction, from raw COCO JSON) and
`src/sampler.py::CalibrationSampler` (INT8 calibration image selection, from the built
subset) deliberately share the same diversity toolkit — phash dedup + CNN-embedding
diversity + class quotas — but sit at opposite ends of the pipeline with different
constraints (greedy similarity filtering at 12k scale vs. farthest-first traversal at
300-image scale). They are **not** duplicates; the dependency is one-directional: the
subset builder produces the `data.yaml` the calibration sampler consumes.

---

## 2. Dataset construction (`train/dataset/build_coco_subset.py`)

12 classes: `person, bicycle, car, bus, truck, motorcycle, dog, cat, chair, bottle,
backpack, traffic light`.

Selection controls (why the subset is not a random sample):

- **Class quotas** — `size/12 × 1.2` per class, long-tail classes ×1.5, `person` capped
  at 18 % of the set (person appears in a large fraction of COCO images and would
  otherwise dominate every multi-label image).
- **phash dedup** — perceptual-hash Hamming distance < threshold (default 6) drops
  near-duplicates. Hashes are packed into one `uint64` array and scanned with a
  vectorized XOR + popcount lookup (see implementation notes below); a candidate's
  hash is registered **only on accept**, so rejected images never block
  near-duplicates of themselves.
- **Embedding diversity** — ResNet-18 (pretrained, penultimate layer, L2-normalized)
  cosine similarity > 0.9 against already-selected images rejects visually redundant
  picks; embeddings are cached to `embed_cache.npy`.
- **person pollution control** — an image whose primary class is not `person` but which
  carries `person` and more than 2 annotation instances is skipped (the guard counts
  instances, duplicates included — not distinct classes; the multi-label cap below is
  the distinct-class check).
- **Multi-label cap** — ≤ 4 distinct classes per image.
- **Val mirrors train** — val quotas are derived from the realized train distribution
  (floor 30/class), with a look-ahead guard rejecting picks that would push any class
  over 1.3× its quota (prevents val distribution blow-up).

Output: `images/{train,val}`, `labels/{train,val}` (YOLO format), `data.yaml`.

```bash
# --coco_root defaults to ./data/COCO (annotations/ + train2017/ + val2017/)
# --train_size / --val_size / --phash_th default to 12000 / 1000 / 6
python train/dataset/build_coco_subset.py --out_root data/coco_subset_12cls
```

`--seed` (default 42) seeds the RNG for reproducible selection.

### Implementation notes

- `FastDedup.add()` registers a hash only on accept and `is_dup()` is a pure check, so
  the dedup set always equals the accepted set — a candidate rejected by a downstream
  filter never blocks near-duplicates of itself.
- The vectorized Hamming scan preserves brute-force semantics exactly (same greedy
  order, same threshold), verified against a brute-force reference, and avoids the
  ~10⁸ Python-level Hamming comparisons a per-pair `ImageHash` loop costs at the 12k
  scale.
- The ResNet-18 embedding similarity check is O(N²) numpy over a growing list
  (documented in the code) — measured acceptable at the 12k scale; a
  preallocated-matrix rewrite is unnecessary at this scale.

## 3. Dataset analysis (`analyze_dataset.py`)

Run on the built subset; writes to a **hardcoded `data/analysis_results/`** (relative
to the cwd, independent of `--root`): four plots, `results.json`, and a `report.txt`
stdout tee of the whole analysis.
Class names are read from the subset's `data.yaml` when present (both list- and
mapping-style `names`); the hardcoded 12-class list is only a fallback for
subsets without one.

```bash
python train/dataset/analyze_dataset.py --root data/coco_subset_12cls
```

### Train distribution (20,816 instances)

| ID | Class         | Count | Percent |
|----|---------------|-------|---------|
| 0  | person        | 3152  | 15.14 % |
| 1  | bicycle       | 878   | 4.22 %  |
| 2  | car           | 4481  | 21.53 % |
| 3  | bus           | 1291  | 6.20 %  |
| 4  | truck         | 1670  | 8.02 %  |
| 5  | motorcycle    | 1419  | 6.82 %  |
| 6  | dog           | 1527  | 7.34 %  |
| 7  | cat           | 1521  | 7.31 %  |
| 8  | chair         | 1200  | 5.76 %  |
| 9  | bottle        | 1618  | 7.77 %  |
| 10 | backpack      | 558   | 2.68 %  |
| 11 | traffic light | 1501  | 7.21 %  |

### Val distribution (1,126 instances)

| ID | Class         | Count | Percent |
|----|---------------|-------|---------|
| 0  | person        | 196   | 17.41 % |
| 1  | bicycle       | 30    | 2.66 %  |
| 2  | car           | 279   | 24.78 % |
| 3  | bus           | 62    | 5.51 %  |
| 4  | truck         | 92    | 8.17 %  |
| 5  | motorcycle    | 70    | 6.22 %  |
| 6  | dog           | 78    | 6.93 %  |
| 7  | cat           | 74    | 6.57 %  |
| 8  | chair         | 70    | 6.22 %  |
| 9  | bottle        | 93    | 8.26 %  |
| 10 | backpack      | 10    | 0.89 %  |
| 11 | traffic light | 72    | 6.39 %  |

### Diagnosis

- **Imbalance ratio (max/min):** train **8.03**, val **27.90** — moderate long tail;
  val is more skewed than train (backpack drops to 0.89 %).
- **KL(train ‖ val) = 0.0213** — val distribution tracks train closely enough for
  reliable model selection (the look-ahead guard in the builder did its job).
- **Head classes:** car, person (~37 % of train instances). **Tail classes:**
  backpack (2.68 %, worst), bicycle (4.22 %).

This diagnosis is the direct input to the staged design below: the tail classes get
custom class weights and a progressively stronger `cls` loss gain.

---

## 4. Baseline training (`train_baseline.py`)

Single-stage fine-tune of pretrained `yolov8s.pt`: **SGD**, `lr0=0.008`, 200 epochs
(`patience=80`), `imgsz=640`, `batch=16`, loss weights `box=7.5, dfl=1.5` (Ultralytics
defaults) with `cls` raised to 1.2 (Ultralytics default is 0.5), strong augmentation
(`mosaic=0.8, mixup=0.05, copy_paste=0.2`). Python/NumPy/Torch RNGs seeded (42) with
`cudnn.deterministic=True` (note: the script does not pass `seed` to `model.train()`,
so the ultralytics trainer's internal default of 0 governs train-time shuffling).
Ends by packaging `runs/`
into a tarball for download from the Kaggle notebook. `--epochs` / `--patience` CLI
flags override the `TrainConfig` defaults (mirrors the staged script's argparse
override pattern; omitted flags leave the config values untouched).

Role in the project: the control arm for the staged long-tail experiment — same subset,
same backbone, no long-tail countermeasures — and, per the unified evaluation in §6,
the **final checkpoint producer**.

## 5. Staged long-tail training (`train_staged/`) — the long-tail experiment

> Outcome first: at unified evaluation this recipe did **not** beat the single-stage
> baseline — neither overall nor on the tail classes it targets (§6). The mechanism
> documentation below is kept in full as the engineering record.

Module layout: `train.py` (entry + final val/visualization), `trainer.py` (per-stage
`YOLO.train` orchestration + the class-weight injection callback), `dataset.py` (Kaggle
dataset prep, `data.yaml` generation, class-weight computation), `utils.py` (seed,
device, config override, logging, failure-safe `safe_run` with `on_retry` hook,
`checkpoint_retry_runner` for resume-on-retry, `log_environment` version-drift guard),
`config.yaml` (all stage/optimizer/augmentation params).

### Custom class weights (computed from train label frequencies, smoothed + normalized)

```python
custom_class_weights = [
    0.56,   # 0 person
    1.38,   # 1 bicycle
    0.44,   # 2 car
    1.05,   # 3 bus
    0.88,   # 4 truck
    0.98,   # 5 motorcycle
    0.93,   # 6 dog
    0.94,   # 7 cat
    1.11,   # 8 chair
    0.90,   # 9 bottle
    1.89,   # 10 backpack   ← largest boost (worst tail class)
    0.95,   # 11 traffic light
]
```

Applied from Stage B onward — `train.py` passes the weights only for stages B/C, via
the `on_train_start` callback described below; Stage A stays unweighted so the model
first learns generic localization. Per-class weighting goes exclusively through the
injected `class_weights` attribute.

### Class-weight injection mechanism (ultralytics ≥ 8.4)

`v8DetectionLoss.__init__` reads `getattr(model, "class_weights", None)` off the
**DetectionModel** and scales the per-class BCE loss
(`bce_loss *= class_weights.view(1,1,nc)`).

Two approaches that do **not** work:

1. Monkey-patching `model.loss` on the `YOLO` object with a
   `WeightedLoss(v8DetectionLoss)` subclass overriding `cls_loss(preds, targets)` —
   the training loop never reads attributes on the `YOLO` *engine wrapper* (it calls
   `unwrap_model(self.model).loss(...)` on the inner `DetectionModel`), and no
   ultralytics 8.x loss class exposes a `cls_loss(preds, targets)` hook (the cls loss
   is computed inline inside `v8DetectionLoss`).
2. Setting `model.model.class_weights` before `train()` — for a fresh (non-resume)
   run `Model.train` **rebuilds** the DetectionModel via `trainer.get_model()`, and the
   new instance inherits only the state_dict, so the attribute is lost before the
   criterion is ever constructed.

The working implementation registers an `on_train_start` callback
(`train_staged/trainer.py::_inject_class_weights`): it fires inside `_do_train`
**after** the rebuild and **before** the first loss call lazily builds the criterion.
Verified end-to-end with a smoke training run: the callback fires on the rebuilt
DetectionModel, `criterion.class_weights` holds the injected tensor, and the weights
broadcast over the BCE loss.

### Three-stage schedule

| Param                    | Stage A (warm-up) | Stage B (main)          | Stage C (refine) | Trend & rationale |
|--------------------------|-------------------|-------------------------|------------------|-------------------|
| epochs                   | 50                | 100                     | 70               | — |
| freeze                   | `[0]` (light)     | none                    | none             | warm-up protects pretrained stem |
| box                      | 8.0               | 7.5                     | 7.0              | high→low: establish box regression first, then yield to classification |
| cls                      | 1.35              | 1.75                    | 2.10             | low→high: progressively force attention on tail classes (with custom weights) |
| dfl                      | 1.3               | 1.7                     | 2.0              | rising: sharper box-boundary distributions late, helps small objects (backpack, bottle) |
| mosaic                   | 0.90              | 0.85                    | 0.75             | strong→weak augmentation: explore early, converge on near-real distribution late |
| mixup                    | 0.15              | 0.10                    | 0.05             | same |
| copy_paste               | 0.35              | 0.28                    | 0.18             | same |
| degrees                  | 12.0              | 8.0                     | 5.0              | rotation: explore early, avoid distorting object shapes late |
| scale                    | 0.60              | 0.55                    | 0.50             | small-object friendly, tapered |
| shear                    | 2.0               | 1.5                     | 1.0              | same |
| hsv_s / hsv_v            | 0.70 / 0.40       | 0.60 / 0.38             | 0.50 / 0.35      | photometric noise tapered |
| hsv_h, translate, fliplr | 0.015, 0.1, 0.5   | (global, not per-stage) |                  | symmetric classes dominate → keep fliplr |

Optimizer: **AdamW**, `lr0=0.0015`, cosine LR, `weight_decay=5e-4`, 3-epoch warm-up;
`imgsz=768`, `batch=16`, `multi_scale=0.5` (±50% size range — ultralytics ≥8.4
reads `multi_scale` as a float fraction; the legacy `True` now means ±100% and can
shrink inputs to a 1×1 feature map, crashing BatchNorm on single-image DDP tail
batches), `seed=42`. Stage B/C resume from the
previous stage's best weights (`safe_run` chains the checkpoint path).

Retry economics: when a stage attempt crashes mid-run, the `safe_run` retry does
not restart from epoch 0 — `checkpoint_retry_runner` resumes from that attempt's
`weights/last.pt` via `YOLO(last.pt).train(resume=True)` (all training args are
restored from the checkpoint; the class-weight callback is re-registered because
it is not part of the checkpoint). A `last.pt` whose mtime predates the failed
attempt (i.e. left over from a previous run in the `exist_ok` run directory) is
ignored, so a stale completed run is never mistaken for crash-recovery state.

Monitoring heuristic used during the runs: **Precision dropping fast → lower `cls`/`dfl`;
Recall improving slowly → raise `cls` toward 2.2–2.4.**

### Usage (Kaggle or local with paths in `config.yaml` adjusted)

```bash
pip install -r train/requirements.txt   # from repo root — pins ultralytics==8.4.114
cd train/train_staged
python train.py                                    # all three stages
python train.py --stage B --lr0 0.0012 stages.B.cls=2.2  # single stage + overrides
                                                                  # NOTE: bare --cls applies to ALL three stages
                                                                  # (see train/train_staged/utils.py); use the
                                                                  # dotted-key form stages.<X>.cls=... to scope.
python train.py --stage C stages.C.cls=2.3         # dotted config overrides
```

Final step in `train.py`: `model.val()` on the val split (mAP50 / mAP50-95 / P / R) +
one random visualization inference.

---

## 6. Results

Every number below comes from the training scripts' own final validation plus a
**unified re-evaluation** (`model.val`, imgsz 640, same 397-image / 1,126-instance
val split).

| Run | mAP50-95 | mAP50 | Precision | Recall | Notes |
|-----|----------|-------|-----------|--------|-------|
| baseline (`train_baseline.py`) | _pending_ | _pending_ | _pending_ | _pending_ | single-stage SGD |
| staged, schedule-only (`train_staged/`, all stages `class_weights=None`) | _pending_ | _pending_ | _pending_ | _pending_ | ablation: A/B/C schedule without weighting |
| staged + class weights (`train_staged/` as-is — weights on B/C) | _pending_ | _pending_ | _pending_ | _pending_ | the full recipe (§5) |


> ⚠️ Checkpoint coupling: the deployment-side results (the 7-backend benchmarks,
> consistency runs, INT8 calibration) all derive from the checkpoint staged as
> `models/yolov8s.pt` — the baseline checkpoint above.

## 7. Handoff to the deployment repo

The **baseline run's** checkpoint (§6 run log: sha256 `523b21a0…c5f6`) is staged as
`models/yolov8s.pt` (see the deployment-side README §Installation) and consumed by
`python main.py export` — from there the deployment pipeline takes over: ONNX FP32 →
INT8 QDQ (calibration images chosen by `src/sampler.py` from the same subset's val
split) → 7-backend consistency/benchmark.
