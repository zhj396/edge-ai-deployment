#!/usr/bin/env python
"""Diagnose whether the TRT-bound QDQ ONNX actually excluded the Detect head.

The default TRT INT8 path (`tensorrt build --precision int8`, no `--calibrator`)
reuses `src.quantize.quantize_onnx_to_int8` to produce a QDQ ONNX whose
`/model.22/` Detect head should carry **no** QuantizeLinear/DequantizeLinear
nodes — the head stays FP32 by QDQ omission, and TRT 10 consumes the graph with
the INT8 flag and no calibrator (CLAUDE.md invariant #16).

The head exclusion keys on `_resolve_node_names_by_prefix(model_for_quant,
["/model.22/"])` resolved from the **pre-processed** graph (the one
`quantize_static` sees), because `quant_pre_process` may rename/restructure
nodes (invariant #3). The canonical ORT INT8 path was validated on the opset-17
+simplify ONNX; the TRT path feeds it the **opset-13, simplify=False, static**
TRT-friendly ONNX. If `quant_pre_process` restructures head nodes on the
opset-13 graph so the `/model.22/` prefix no longer matches, the head is NOT in
`nodes_to_exclude` → `quantize_static` inserts Q/DQ into the head → TRT
quantizes it → the unbounded box-decode accumulation collapses →
`max_diff`≈300-560 with a nonzero `detection_fail_rate` (the signature that
mimics calibrator-path collapse even though `--calibrator` was not passed).

This script is the **definitive** check for the QDQ path and it is PURE — it
loads the ONNX graph with `onnx` only, no tensorrt / no CUDA / no ORT session —
so it runs on any laptop, not just the Kaggle build box. (The companion
`scripts/diag_trt_head.py` inspects the built `.engine` via the TRT engine
inspector, which per invariant #16 carries no per-layer precision field on TRT
10.4 — inconclusive for the QDQ path by design. The QDQ path's ground truth
lives in the ONNX, not the engine.)

Run it on the box where the QDQ ONNX lives (it need not be the GPU box):

    python scripts/diag_qdq_head.py models/yolov8s_trt_int8.onnx
    # optional: also inspect the FP32 source to see pre-quant head node names
    python scripts/diag_qdq_head.py models/yolov8s_trt_int8.onnx models/yolov8s_trt.onnx

Exit code: 0 if the head is clean (no Q/DQ under /model.22/), 1 if head Q/DQ
nodes are present (the head was NOT excluded — root cause of the 18.75% cliff
collapse on a "QDQ path" run).
"""
import sys
from pathlib import Path

HEAD_PREFIX = "/model.22/"
QDQ_TYPES = ("QuantizeLinear", "DequantizeLinear")


def _load_nodes(onnx_path: Path):
    """Return the list of graph nodes from an ONNX model (no external data)."""
    import onnx
    return list(onnx.load_model(str(onnx_path), load_external_data=False).graph.node)


def _is_head(name: str) -> bool:
    return bool(name) and name.startswith(HEAD_PREFIX)


def _summarize(onnx_path: Path) -> int:
    """Print a head-QDQ summary for one ONNX. Returns head QDQ node count."""
    nodes = _load_nodes(onnx_path)
    qdq_nodes = [n for n in nodes if n.op_type in QDQ_TYPES]
    head_qdq = [n for n in qdq_nodes if _is_head(n.name)]
    head_total = [n for n in nodes if _is_head(n.name)]

    print(f"\n=== {onnx_path} ===")
    print(f"  total nodes: {len(nodes)}")
    print(f"  Q/DQ nodes (whole graph): {len(qdq_nodes)}")
    print(f"  /model.22/ head nodes (any type): {len(head_total)}")
    print(f"  /model.22/ head Q/DQ nodes: {len(head_qdq)}")

    if head_total:
        print("  sample head node names (first 8):")
        for n in head_total[:8]:
            print(f"    {n.op_type:20s}  {n.name}")
    else:
        print("  WARNING: no nodes start with '/model.22/' in this graph —")
        print("          the head-exclusion prefix matches NOTHING here, so")
        print("          quantize_static excluded nothing by prefix. If this")
        print("          is the QDQ ONNX, the head was quantized wholesale.")
        print("          Dump the first 15 node names to see the naming scheme:")
        for n in nodes[:15]:
            print(f"    {n.op_type:20s}  {n.name}")

    if head_qdq:
        print(f"  *** {len(head_qdq)} Q/DQ node(s) landed inside /model.22/ —")
        print("      the head was NOT excluded and TRT quantized it. This is")
        print("      the root cause of the cliff-collapse signature")
        print("      (max_diff~300-560, detection_fail_rate>0) on a run that")
        print("      did NOT pass --calibrator. Head FP32 protection failed at")
        print("      the QDQ-ONNX step, not the TRT build step. ***")
        print("  head Q/DQ nodes (first 12):")
        for n in head_qdq[:12]:
            print(f"    {n.op_type:20s}  {n.name}")
    else:
        print("  OK: no Q/DQ nodes under /model.22/ — the head was excluded;")
        print("      TRT ran it FP32 by construction. A nonzero")
        print("      detection_fail_rate with this clean is genuine backbone-")
        print("      INT8 divergence at the conf cliff (not head collapse), and")
        print("      the 'QDQ -> 0%' expectation in the docs is over-stated.")

    return len(head_qdq)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    worst = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        try:
            worst = max(worst, _summarize(p))
        except Exception as e:  # pragma: no cover — best-effort per file
            print(f"  could not load {p}: {e}", file=sys.stderr)
    return 1 if worst > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
