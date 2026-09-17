# Artifacts

Two binary artifacts are required to run the deployment pipeline on real data.
Neither is in git — `models/` and `data/` are git-ignored by design so the
repository stays a fast, text-only clone and the test suite remains model-free
(`pytest -q` passes on a fresh checkout with zero downloads). Both ship as assets
on the same GitHub Release (tag `v0.1.0` at the time of writing), each pinned by
sha256 below.

| Artifact | Expected path | Size | Source |
|----------|---------------|------|--------|
| YOLOv8s 12-class checkpoint | `models/yolov8s.pt` | 21.5 MB | GitHub Release, sha256-pinned below |
| COCO 12-class subset (val split) | `data/` (must contain `data.yaml`) | ~65 MB | GitHub Release, sha256-pinned below |

## Checkpoint — `models/yolov8s.pt`

Produced by the baseline training run recorded in
[docs/TRAINING.md §6](docs/TRAINING.md#6-results) (single-stage SGD, imgsz 640,
early-stopped at 161/200 epochs). Download and verify:

```bash
mkdir -p models
curl -L -o models/yolov8s.pt https://github.com/zhj396/edge-ai-deployment/releases/download/v0.1.0/yolov8s.pt
echo "523b21a075a67eccd06071a6313c49e6acb252677984e3ee02ebc8f9e265c5f6 models/yolov8s.pt" | sha256sum -c
```

(Windows PowerShell: `Get-FileHash models/yolov8s.pt -Algorithm SHA256`.)

The checksum must match the run-log entry in TRAINING.md §6 — if it does not,
you are holding a different checkpoint than every number in this repository
was measured against.

## Dataset — `data/`

The COCO 2017 12-class subset built by `train/dataset/build_coco_subset.py`
(seeded, phash-deduplicated, embedding-diverse; construction documented in
[docs/TRAINING.md §2](docs/TRAINING.md#2-dataset-construction-traindatasetbuild_coco_subsetpy)).
Its val split (397 images / 1,126 instances) is the evaluation set behind
every number in TRAINING.md §6 and the input dataset for the deployment
pipeline's INT8 calibration, consistency validation, and benchmarking.

The deployment pipeline reads only the val split from `data/`. INT8
calibration additionally requires the **labels** — the calibration sampler
reads per-image class annotations from `labels/val/` for stratified
sampling. The Release asset therefore packages
`data.yaml` + `images/val/` + `labels/val/` — all three are required.

**Download the val split from GitHub Release:**

```bash
mkdir -p data
curl -L -o coco12-val.zip https://github.com/zhj396/edge-ai-deployment/releases/download/v0.1.0/coco12-val.zip
echo "c7fa7a71dd8ddfb2cc47a32704de1e3c1484c54ca6375307706d993a60f8a8b2  coco12-val.zip" | sha256sum -c
unzip -q coco12-val.zip -d data   # zip root: data.yaml, images/val/, labels/val/
```

(Windows PowerShell: `Expand-Archive coco12-val.zip -DestinationPath data`;
`Get-FileHash coco12-val.zip -Algorithm SHA256`.)

## Expected layout after setup

```text
models/
└── yolov8s.pt            # the checkpoint above; derived model files are
                          # generated locally by the deployment pipeline
data/
├── data.yaml             # from the Release asset; val path + 12-class names
├── images/val/           # 397 images — calibration, consistency, benchmark
└── labels/val/           # required by INT8 calibration sampling
```
