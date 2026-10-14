"""Per-image FP32-vs-INT8 detection diff on the consistency-FAILED images.

Mirrors the consistency pipeline exactly (same models, same preprocess,
same post_process conf/iou, same greedy 1:1 IoU matching as
``utils/comparison.py::_match_predictions_to_gt``), then reports per image:

* matched pairs: conf on both sides + IoU;
* FP32-only / INT8-only boxes at conf 0.25, each linked to its best-IoU
  counterpart on the other side at conf 0.10 (conf + IoU) — where the
  unmatched box went, including below the 0.25 threshold;
* the surviving boxes in the [0.10, 0.25) band on each side.

Writes ``results/consistency_failed_analysis.json`` (atomic) and prints a
table. Run from the repo root with the project venv active:

    python diag_consistency_failed.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import yaml

from src.consistency import ModelWrapper, atomic_json_dump
from src import post_process, preprocess_imgs
from utils.comparison import compute_iou
from utils.model_utils import sha256_of_file

REPO = Path(__file__).resolve().parent

IMAGES = [
    "data/images/val/000000022892.jpg",
    "data/images/val/000000477227.jpg",
    "data/images/val/000000546976.jpg",
]
MODEL_FP32 = "models/yolov8s_fp32.onnx"
MODEL_INT8 = "models/yolov8s_int8.onnx"

CONF = 0.25        # consistency conf_thres (cli default)
IOU_NMS = 0.45     # consistency iou_thres (cli default)
MATCH_IOU = 0.5    # consistency det_iou_thresh
CONF_PROBE = 0.10  # below-cliff probe


# Greedy 1:1 matching, mirroring _match_predictions_to_gt but tracking d1
# indices so unmatched boxes can be reported. Semantics: iterate d1 in order,
# take the best-IoU unused d2, match only if IoU >= MATCH_IOU.
def match_tracked(dets1, dets2):
    used = set()
    matches = []          # (i1, j2, iou, score_diff, class_match)
    for i, d1 in enumerate(dets1):
        best_iou, best_j = 0.0, -1
        for j, d2 in enumerate(dets2):
            if j in used:
                continue
            iou = compute_iou(d1[:4], d2[:4])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= MATCH_IOU:
            used.add(best_j)
            matches.append((i, best_j, best_iou,
                            abs(d1[4] - dets2[best_j][4]),
                            int(int(d1[5]) == int(dets2[best_j][5]))))
    return matches


def best_iou_to(box, dets):
    """Best (iou, conf, cls) over ``dets`` regardless of matching state."""
    best = (0.0, None, None)
    for d in dets:
        iou = compute_iou(box[:4], d[:4])
        if iou > best[0]:
            best = (iou, float(d[4]), int(d[5]))
    return best


def probe_below_cliff(out, pre, conf_probe, cliff):
    """Detections in [conf_probe, cliff) — the near-cliff band."""
    dets = post_process(
        out, orig_shapes=pre["orig_shapes"],
        conf_thres=conf_probe, iou_thres=IOU_NMS, imgsz=640,
        ratios=pre.get("ratios"), pads=pre.get("pads"),
    )[0]
    return [d for d in dets if conf_probe <= float(d[4]) < cliff]


def main():
    with open(REPO / "data" / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, list):  # names as list vs id->name dict
        names = dict(enumerate(names))

    fp32 = ModelWrapper(MODEL_FP32, "cpu")
    int8 = ModelWrapper(MODEL_INT8, "cpu")

    report = {
        "date": str(date.today()),
        "device": "cpu",
        "models": {
            "fp32": {"path": MODEL_FP32, "sha256": sha256_of_file(REPO / MODEL_FP32)},
            "int8": {"path": MODEL_INT8, "sha256": sha256_of_file(REPO / MODEL_INT8)},
        },
        "params": {"conf": CONF, "iou_nms": IOU_NMS, "match_iou": MATCH_IOU,
                   "conf_probe": CONF_PROBE},
        "images": [],
    }

    for rel in IMAGES:
        pre = preprocess_imgs([str(REPO / rel)], imgsz=640, device="cpu")
        x = pre["images"]
        out1, out2 = fp32.forward(x), int8.forward(x)
        kw = dict(orig_shapes=pre["orig_shapes"], conf_thres=CONF,
                  iou_thres=IOU_NMS, imgsz=640,
                  ratios=pre.get("ratios"), pads=pre.get("pads"))
        d1 = post_process(out1, **kw)[0]
        d2 = post_process(out2, **kw)[0]

        matches = match_tracked(d1, d2)
        m1 = {i for i, *_ in matches}
        m2 = {j for _, j, *_ in matches}

        entry = {
            "image": rel,
            "n_fp32": len(d1), "n_int8": len(d2), "matched": len(matches),
            "recall": len(matches) / max(len(d1), 1),
            "matched_pairs": [
                {"cls": names.get(int(d1[i][5]), int(d1[i][5])),
                 "conf_fp32": round(float(d1[i][4]), 4),
                 "conf_int8": round(float(d2[j][4]), 4),
                 "conf_diff": round(sd, 4), "iou": round(iou, 4),
                 "class_match": bool(cm)}
                for i, j, iou, sd, cm in matches
            ],
            "fp32_only": [], "int8_only": [],
            "below_cliff_fp32": [], "below_cliff_int8": [],
        }

        for i, d in enumerate(d1):
            if i in m1:
                continue
            iou, conf2, _ = best_iou_to(d, d2)
            entry["fp32_only"].append({
                "cls": names.get(int(d[5]), int(d[5])),
                "conf_fp32": round(float(d[4]), 4),
                "best_iou_to_int8": round(iou, 4),
                "best_int8_conf": None if conf2 is None else round(conf2, 4),
            })
        for j, d in enumerate(d2):
            if j in m2:
                continue
            iou, conf1, _ = best_iou_to(d, d1)
            entry["int8_only"].append({
                "cls": names.get(int(d[5]), int(d[5])),
                "conf_int8": round(float(d[4]), 4),
                "best_iou_to_fp32": round(iou, 4),
                "best_fp32_conf": None if conf1 is None else round(conf1, 4),
            })

        # Below-cliff probe at conf 0.10: where did each unmatched box go on
        # the other side?
        lo1 = post_process(
            out1, orig_shapes=pre["orig_shapes"],
            conf_thres=CONF_PROBE, iou_thres=IOU_NMS, imgsz=640,
            ratios=pre.get("ratios"), pads=pre.get("pads"))[0]
        lo2 = post_process(
            out2, orig_shapes=pre["orig_shapes"],
            conf_thres=CONF_PROBE, iou_thres=IOU_NMS, imgsz=640,
            ratios=pre.get("ratios"), pads=pre.get("pads"))[0]

        for r, d in zip(entry["fp32_only"], [d for i, d in enumerate(d1) if i not in m1]):
            iou, conf2, cls2 = best_iou_to(d, lo2)
            r["int8_counterpart_at_conf010"] = {
                "iou": round(iou, 4),
                "conf": None if conf2 is None else round(conf2, 4),
                "cls": names.get(cls2, cls2) if cls2 is not None else None,
            }
        for r, d in zip(entry["int8_only"], [d for j, d in enumerate(d2) if j not in m2]):
            iou, conf1, cls1 = best_iou_to(d, lo1)
            r["fp32_counterpart_at_conf010"] = {
                "iou": round(iou, 4),
                "conf": None if conf1 is None else round(conf1, 4),
                "cls": names.get(cls1, cls1) if cls1 is not None else None,
            }
        for d in lo1:
            if CONF_PROBE <= float(d[4]) < CONF:
                entry["below_cliff_fp32"].append(
                    {"cls": names.get(int(d[5]), int(d[5])),
                     "conf": round(float(d[4]), 4)})
        for d in lo2:
            if CONF_PROBE <= float(d[4]) < CONF:
                entry["below_cliff_int8"].append(
                    {"cls": names.get(int(d[5]), int(d[5])),
                     "conf": round(float(d[4]), 4)})

        report["images"].append(entry)

        # ---- console table ----
        print(f"\n=== {Path(rel).name}  (FP32 {len(d1)} dets, INT8 {len(d2)} dets, "
              f"matched {len(matches)}, recall {entry['recall']:.3f})")
        for i, j, iou, sd, cm in matches:
            print(f"  matched  {names.get(int(d1[i][5])):<12} "
                  f"conf {float(d1[i][4]):.4f} -> {float(d2[j][4]):.4f} "
                  f"(diff {sd:.4f}, IoU {iou:.4f})")
        for r in entry["fp32_only"]:
            cp = r["int8_counterpart_at_conf010"]
            print(f"  FP32-ONLY {r['cls']:<12} conf {r['conf_fp32']:.4f} "
                  f"best-IoU@0.25 {r['best_iou_to_int8']} to INT8 conf {r['best_int8_conf']}"
                  f" | @conf0.10: IoU {cp['iou']} conf {cp['conf']} cls {cp['cls']}")
        for r in entry["int8_only"]:
            cp = r["fp32_counterpart_at_conf010"]
            print(f"  INT8-ONLY {r['cls']:<12} conf {r['conf_int8']:.4f} "
                  f"best-IoU@0.25 {r['best_iou_to_fp32']} to FP32 conf {r['best_fp32_conf']}"
                  f" | @conf0.10: IoU {cp['iou']} conf {cp['conf']} cls {cp['cls']}")
        print(f"  below-cliff [{CONF_PROBE},{CONF}) — FP32: {entry['below_cliff_fp32']}"
              f"  INT8: {entry['below_cliff_int8']}")

    out_path = REPO / "results" / "consistency_failed_analysis.json"
    atomic_json_dump(report, out_path)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
