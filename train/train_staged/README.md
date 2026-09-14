# `train_staged/` — Three-Stage Long-Tail Training (final checkpoint)

Entry: `python train.py` (see `config.yaml` for all stage/optimizer/augmentation
params; `--stage A|B|C|all`, dotted config overrides supported).

Design rationale, class weights, stage schedule, and results:
[`docs/TRAINING.md`](../../docs/TRAINING.md) §5.
