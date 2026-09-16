"""YOLOv8s multi-stage long-tail training (staged A/B/C, AdamW + class weights).

Kaggle: provision the environment with `pip install -r train/requirements.txt`
in the first cell before executing — it pins the exact ultralytics release this
pipeline is validated against (the class-weight injection in
docs/TRAINING.md §5 and the fractional `multi_scale` semantics both depend on
it). Do NOT `pip install -U ultralytics`: that floats to the latest release
and silently invalidates the pinned-environment guarantee. `log_environment()`
logs the runtime versions and warns on drift at startup.
"""

import os
import glob
import random

import yaml
import argparse
from dataset import compute_class_weights, create_yaml, prepare_dataset
from trainer import stage_checkpoint, train_stage
from utils import (checkpoint_retry_runner, get_device, log_environment,
                   logger, override_config, safe_run, set_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8s multi-stage long-tail training")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the config file")
    parser.add_argument("--stage", type=str, choices=["A", "B", "C", "all"], default="all",
                        help="Run a single stage or all of them")

    # ==================== Key tunable parameters ====================
    # 1. Global parameters
    parser.add_argument("--lr0", type=float, default=None,
                        help="Initial learning rate (e.g. 0.001)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size (-1 = auto)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Input image size")
    parser.add_argument("--epochsA", type=int, default=None,
                        help="Stage A epochs")
    parser.add_argument("--epochsB", type=int, default=None,
                        help="Stage B epochs")
    parser.add_argument("--epochsC", type=int, default=None,
                        help="Stage C epochs")

    # 2. Loss gains (most frequently tuned)
    parser.add_argument("--box", type=float, default=None,
                        help="box loss gain")
    parser.add_argument("--cls", type=float, default=None,
                        help="cls loss gain")
    parser.add_argument("--dfl", type=float, default=None,
                        help="dfl loss gain")

    # 3. Augmentation (frequently tuned)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--copy_paste", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)

    # Other common knobs
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Dotted config overrides, e.g. optimizer.lr0=0.002 stages.A.cls=1.5")

    return parser.parse_args()


def load_config():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg = override_config(cfg, args)

    return cfg, args


def main():
    cfg, args = load_config()

    set_seed(cfg.get("seed", 42))
    cfg["train"]["device"] = get_device()
    log_environment()

    logger.info("Starting YOLOv8s multi-stage training")
    logger.info(f"Config: Stage={args.stage}")

    # Data preparation
    prepare_dataset(cfg["data"]["src"], cfg["data"]["dst"])
    create_yaml(cfg["data"]["dst"], cfg["data"]["yaml"])

    # Class weights from train-label frequencies
    class_weights = compute_class_weights(f"{cfg['data']['dst']}/labels/train")

    # Multi-stage training
    weights = cfg["model"]["base"]
    stages = ["A", "B", "C"] if args.stage == "all" else [args.stage]

    for stage in stages:
        cw = class_weights if stage != "A" else None
        w_in = weights  # bind per-iteration: stage N starts from stage N-1's best

        def _run_stage(resume, s=stage, w=w_in, cw=cw):
            return train_stage(cfg, s, w, cw, resume=resume)

        # Retry economics: a mid-stage crash resumes from that attempt's
        # last.pt instead of restarting the stage from epoch 0 (a stale
        # last.pt from a previous run is ignored — see checkpoint_retry_runner).
        run_stage, on_retry = checkpoint_retry_runner(
            _run_stage, stage_checkpoint(cfg, stage), name=f"Stage {stage}")
        weights = safe_run(run_stage, name=f"Stage {stage}", on_retry=on_retry)

    logger.info(f"All stages complete — final weights: {weights}")

    # ====================== Final validation & visualization ======================
    try:
        from ultralytics import YOLO

        DATA_YAML = cfg["data"]["yaml"]
        DST_DATA = cfg["data"]["dst"]

        print("\n" + "=" * 60)
        print("Running final validation...")
        print("=" * 60)

        final_model = YOLO(weights)

        # Val-split evaluation
        metrics = final_model.val(
            data=DATA_YAML,
            imgsz=cfg["model"].get("imgsz", 768),
            batch=16,
            device=cfg["train"]["device"],
            plots=True
        )

        print("\nFinal validation metrics:")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")

        # ====================== Visualization inference ======================
        val_img_dir = os.path.join(DST_DATA, "images/val")
        val_imgs = glob.glob(os.path.join(val_img_dir, "*.jpg")) + glob.glob(os.path.join(val_img_dir, "*.png"))

        if val_imgs:
            test_img = random.choice(val_imgs)
            print(f"\nRandom visualization example: {test_img}")

            results = final_model.predict(
                source=test_img,
                conf=0.25,
                iou=0.7,
                save=True,
                save_conf=True,
                line_width=2
            )

            save_dir = results[0].save_dir if hasattr(results[0], 'save_dir') else "runs/detect/predict"
            print(f"Predictions saved to: {save_dir}")
        else:
            print("No val images found — skipping visualization inference")

    except Exception as e:
        logger.error(f"Final validation/visualization failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("All steps finished!")


if __name__ == "__main__":
    main()
