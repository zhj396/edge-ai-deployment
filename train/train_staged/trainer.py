from pathlib import Path

from ultralytics import YOLO
import torch
from utils import logger


def stage_checkpoint(cfg, stage):
    """Path to the stage's ``weights/last.pt``.

    Mirrors the ``project``/``name`` overrides passed to ``model.train``
    (``<project>/<experiment>_<stage>``). Centralized so the retry/resume
    logic and the trainer cannot disagree about the run-directory layout.
    """
    return Path(cfg["project"]) / f"{cfg['experiment']}_{stage}" / "weights" / "last.pt"


def _inject_class_weights(class_weights):
    """Build an ``on_train_start`` callback that installs per-class weights.

    Mechanism (ultralytics >= 8.4): ``v8DetectionLoss.__init__`` reads
    ``getattr(model, "class_weights", None)`` off the DetectionModel and scales
    the per-class BCE cls loss by it (``bce_loss *= class_weights.view(1,1,nc)``).

    Why a callback and not a plain attribute set before ``model.train()``:
    for a fresh (non-resume) run ``Model.train`` REBUILDS the DetectionModel
    via ``trainer.get_model(weights=..., cfg=...)`` — a fresh instance that
    only inherits the state_dict, so any attribute set on the pre-train model
    is lost. ``on_train_start`` fires inside ``_do_train`` after that rebuild
    and before the first loss call lazily constructs the criterion, so the
    attribute is present exactly when ``v8DetectionLoss`` reads it.
    """
    cw = torch.tensor(class_weights, dtype=torch.float32)

    def _callback(trainer):
        trainer.model.class_weights = cw
        logger.info(f"class_weights injected into {type(trainer.model).__name__}")

    return _callback


def train_stage(cfg, stage: str, weights: str, class_weights=None, resume: bool = False):
    if resume:
        # Crash-recovery path: continue the stage from the failed attempt's
        # checkpoint. `resume=True` restores every training arg (epochs,
        # optimizer, augmentation, DDP layout) from the checkpoint itself, so
        # none of the overrides below are re-applied — this continues the run
        # that crashed rather than starting a differently-configured one.
        last_ckpt = stage_checkpoint(cfg, stage)
        logger.info(f"=== Stage {stage}: resume from {last_ckpt} ===")
        model = YOLO(str(last_ckpt))
        # The class-weight injection callback lives on the wrapper, not in the
        # checkpoint; on_train_start also fires on the resume path (inside
        # _do_train, before the criterion is lazily rebuilt), so re-registering
        # keeps stages B/C weighting intact across a resume.
        if class_weights is not None:
            model.add_callback("on_train_start", _inject_class_weights(class_weights))
            logger.info(f"Stage {stage}: custom class weights re-registered for resume")
        model.train(resume=True)
        best_path = str(model.trainer.best)
        logger.info(f"Stage {stage}: done (resumed) — best weights: {best_path}")
        return best_path

    logger.info(f"=== Stage {stage}: start ===")
    model = YOLO(weights)

    # ====================== Stage-specific parameters ======================
    stage_cfg = cfg["stages"][stage]

    overrides = {
        "data": cfg["data"]["yaml"],
        "imgsz": cfg["model"]["imgsz"],
        "device": cfg["train"]["device"],
        "batch": cfg["train"]["batch"],
        "workers": cfg["train"]["workers"],
        "multi_scale": cfg["train"]["multi_scale"],
        "epochs": cfg["train"]["epochs"][stage],
        "patience": cfg["train"].get("patience", 50),
        "save_period": cfg["train"].get("save_period", 10),
        "amp": True,

        # Optimizer
        "optimizer": cfg["optimizer"]["name"],
        "lr0": cfg["optimizer"]["lr0"],
        "lrf": cfg["optimizer"]["lrf"],
        "weight_decay": cfg["optimizer"]["weight_decay"],
        "cos_lr": cfg["optimizer"].get("cos_lr", True),
        "warmup_epochs": cfg["optimizer"].get("warmup_epochs", 3.0),
        "warmup_momentum": cfg["optimizer"].get("warmup_momentum", 0.8),
        "warmup_bias_lr": cfg["optimizer"].get("warmup_bias_lr", 0.1),

        # Loss gains — read from the per-stage config (highest priority)
        "box": stage_cfg["box"],
        "cls": stage_cfg["cls"],
        "dfl": stage_cfg["dfl"],

        # Augmentation — per-stage values override the global defaults
        "mosaic": stage_cfg["mosaic"],
        "mixup": stage_cfg["mixup"],
        "copy_paste": stage_cfg["copy_paste"],
        "hsv_h": cfg["augment"].get("hsv_h", 0.015),
        "hsv_s": stage_cfg.get("hsv_s", cfg["augment"].get("hsv_s", 0.7)),
        "hsv_v": stage_cfg.get("hsv_v", cfg["augment"].get("hsv_v", 0.4)),
        "degrees": stage_cfg.get("degrees", cfg["augment"].get("degrees", 10.0)),
        "translate": cfg["augment"].get("translate", 0.1),
        "scale": stage_cfg.get("scale", cfg["augment"].get("scale", 0.5)),
        "shear": stage_cfg.get("shear", cfg["augment"].get("shear", 2.0)),
        "fliplr": cfg["augment"].get("fliplr", 0.5),

        # Project settings
        "project": cfg["project"],
        "name": f"{cfg['experiment']}_{stage}",
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", False),
        "val": cfg.get("val", True),
        "plots": cfg.get("plots", True),
        "save": cfg.get("save", True),
        "exist_ok": cfg.get("exist_ok", True),
    }

    # ====================== Freeze setting (light or off) ======================
    if "freeze" in stage_cfg:
        overrides["freeze"] = stage_cfg["freeze"]
    else:
        overrides["freeze"] = None  # freeze nothing

    # ====================== Custom class weights ======================
    if class_weights is not None:
        model.add_callback("on_train_start", _inject_class_weights(class_weights))
        logger.info(f"Stage {stage}: custom class weights registered "
                    f"(applied via on_train_start)")

    # ====================== Train ======================
    model.train(**overrides)

    # Best weights path — taken from the trainer itself, not reconstructed from
    # project/name strings (which drift across ultralytics versions).
    best_path = str(model.trainer.best)
    logger.info(f"Stage {stage}: done — best weights: {best_path}")
    return best_path
