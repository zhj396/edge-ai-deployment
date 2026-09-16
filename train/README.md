# `train/` — Dataset Build & Training Scripts (Upstream of the Deployment Pipeline)

Upstream half of the project lifecycle: builds the COCO 12-class subset, diagnoses its
long tail, and produces the `yolov8s.pt` checkpoint that the deployment half of the
project exports, quantizes, consistency-validates, and benchmarks across 7 backends.
This repository holds the training upstream; the deployment half lives downstream.

**Full narrative, distribution tables, stage schedule, and results live in
[`docs/TRAINING.md`](../docs/TRAINING.md).** This directory is code-only.

| Script | Stage | Role | Env |
|--------|-------|------|-----|
| `dataset/build_coco_subset.py` | 1. build | COCO 2017 → 12k/1k 12-class subset (phash dedup + ResNet-18 embedding diversity + person-cap + val-mirrors-train quotas) → YOLO labels + `data.yaml` | local CPU |
| `dataset/analyze_dataset.py` | 2. measure | per-class counts, imbalance ratio, long-tail split, train↔val KL divergence, plots → `data/analysis_results/` | local CPU |
| `train_baseline.py` | 3a. baseline | single-stage SGD fine-tune — measured best at unified evaluation, **final checkpoint producer** ([`docs/TRAINING.md`](../docs/TRAINING.md) §6) | Kaggle T4 ×2 |
| `train_staged/` | 3b. experiment | three-stage (A/B/C) AdamW training with custom long-tail class weights on stages B/C (injection mechanism: [`docs/TRAINING.md`](../docs/TRAINING.md) §5; negative result at unified evaluation: §6) | Kaggle T4 ×2 |
| `requirements.txt` | — | shared pinned dependency set for both training scripts — provision Kaggle notebooks with `pip install -r train/requirements.txt` (never `pip install -U ultralytics`; the `ultralytics==8.4.114` pin is the validated release, and `train_staged/utils.py::log_environment` warns on drift at startup) | Kaggle T4 ×2 |

## Design notes — environments & dependency boundaries

- **Two environments, split by directory:** `dataset/` runs **locally on CPU**
  (relative paths, no GPU needed); the training scripts (`train_baseline.py`,
  `train_staged/`) run on **Kaggle Tesla T4 ×2**, provisioned from the shared
  `train/requirements.txt`. The hardcoded `/kaggle/...` paths record the
  environment the published results were produced in — keeping them makes the
  dataset→train→quantize→benchmark pipeline reproducible end to end.
- **`train/` is isolated from the deployment half** (`cli/`, `src/`, `utils/`):
  heavy training dependencies never enter the default `pytest -q` suite
  (model-free, GPU-free, seconds), while the scripts stay under the same
  `flake8` gate as the rest of the repo. The one deliberate exception:
  `tests/test_train_staged_retry.py` loads `train_staged/utils.py` in isolation
  (numpy/torch only, no ultralytics) to cover the retry/resume state machine.
- Any change to training behavior requires re-validation with a real training
  run; the class-weight injection mechanism is documented in
  [`docs/TRAINING.md`](../docs/TRAINING.md) §5.
