#!/usr/bin/env python
"""Diagnose whether the TRT-bound QDQ ONNX actually excluded the Detect head.

The default TRT INT8 path (`tensorrt build --precision int8`, no `--calibrator`)
reuses `src.quantize.quantize_onnx_to_int8` to produce a QDQ ONNX whose
`/model.22/` Detect head should carry **no** QuantizeLinear/DequantizeLinear
nodes — the head stays FP32 by QDQ omission, and TRT 10 consumes the graph with
the INT8 flag and no calibrator.

The head exclusion keys on `_resolve_node_names_by_prefix(model_for_quant,
["/model.22/"])` resolved from the **pre-processed** graph (the one
`quantize_static` sees), because `quant_pre_process` may rename/restructure
nodes. The canonical ORT INT8 path was validated on the opset-17
+simplify ONNX; the TRT path feeds it the **opset-13, simplify=False**
TRT-friendly ONNX (static-batch on the `STATIC=1` Turing/T4 route). If
`quant_pre_process` restructures head nodes on the
opset-13 graph so that no node name carries the `/model.22/` prefix, the head
is NOT in `nodes_to_exclude` → `quantize_static` inserts Q/DQ into the head →
TRT quantizes it → the unbounded box-decode accumulation collapses →
a large `max_diff` with a nonzero `detection_fail_rate` (recorded magnitude
`_TBD_` — see docs/TENSORRT.md Provenance; the same signature a head-quantized
collapse is expected to produce on either INT8 path).

This script is the ground-truth check for the QDQ path and it is PURE — it
loads the ONNX graph with `onnx` only, no tensorrt / no CUDA / no ORT session —
so it runs on any laptop, not just the Kaggle build box. (The companion
`scripts/diag_trt_head.py` inspects the built `.engine` via the TRT engine
inspector, which carries no per-layer precision field on some TRT
10.x builds — inconclusive for the QDQ path by design. The QDQ path's ground truth
lives in the ONNX graph.)

Run it on the box where the QDQ ONNX lives (it need not be the GPU box):

    python scripts/diag_qdq_head.py models/yolov8s_trt_int8.onnx
    # optional: also inspect the FP32 source to see pre-quant head node names
    python scripts/diag_qdq_head.py models/yolov8s_trt_int8.onnx models/yolov8s_trt.onnx

Exit code: 0 if every checked file has a clean head (no Q/DQ under /model.22/),
1 if head Q/DQ nodes are present (the head was NOT excluded — with head Q/DQ in
place TRT quantizes the unbounded box-decode, which surfaces as cliff flips on a
"QDQ path" run), 2 on usage error, 3 if no file could be checked (missing or
unloadable) — a run that checked nothing never reports clean.
"""
import re
import sys
from pathlib import Path

HEAD_PREFIX = "/model.22/"
# Slashless names ("model.22.cv2.0.weight") must have "22" followed by a
# non-digit segment ("model.22.0.*" is a sibling module, not the head).
_HEAD_SLASHLESS_RE = re.compile(r"^\.?model\.22\.(?!\d)")
QDQ_TYPES = ("QuantizeLinear", "DequantizeLinear")


def _load_nodes(onnx_path: Path):
    """Return the list of graph nodes from an ONNX model (no external data)."""
    import onnx
    return list(onnx.load_model(str(onnx_path), load_external_data=False).graph.node)


def _is_head(name: str) -> bool:
    """Head-node match under both naming schemes.

    PyTorch-exported node names carry the leading slash
    (``/model.22/...``); names derived from initializers or the
    quantization tensors they feed often do not
    (``model.22.cv2.0.weight``). Matching both keeps a head-side Q/DQ
    node named after its tensor from being missed here (a missed Q/DQ
    would read as a clean head). ``model.22.<digit>...`` (e.g.
    ``/model.22.0/``) is a sibling module, not the head.
    """
    if not name:
        return False
    if name.startswith(HEAD_PREFIX):
        return True
    return bool(_HEAD_SLASHLESS_RE.match(name.replace("/", ".")))


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
        print("  WARNING: no nodes match the head prefixes ('/model.22/',")
        print("          'model.22.') in this graph — the head-exclusion")
        print("          scope matches NOTHING here, so quantize_static")
        print("          excluded nothing by prefix. If this is the QDQ")
        print("          ONNX, the head was quantized wholesale.")
        print("          Dump the first 15 node names to see the naming scheme:")
        for n in nodes[:15]:
            print(f"    {n.op_type:20s}  {n.name}")

    if head_qdq:
        print(f"  *** {len(head_qdq)} Q/DQ node(s) landed inside /model.22/ —")
        print("      the head was NOT excluded and TRT quantized it. The FP32")
        print("      protection failed at the QDQ-ONNX step (signature: a ")
        print("      large max_diff with detection_fail_rate>0) on a ")
        print("      run that did NOT pass --calibrator. ***")
        print("  head Q/DQ nodes (first 12):")
        for n in head_qdq[:12]:
            print(f"    {n.op_type:20s}  {n.name}")
    else:
        print("  OK: no Q/DQ nodes under /model.22/ — the head was excluded;")
        print("      TRT ran it FP32 by construction. A nonzero")
        print("      detection_fail_rate with this clean result points to ")
        print("      genuine backbone-INT8 divergence at the conf cliff.")

    return len(head_qdq)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    worst = 0    # max head Q/DQ count over the files that were checked
    checked = 0  # files actually loaded and summarized
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        try:
            worst = max(worst, _summarize(p))
            checked += 1
        except Exception as e:  # pragma: no cover — best-effort per file
            print(f"  could not load {p}: {e}", file=sys.stderr)
    if checked == 0:
        return 3  # nothing was checked — "clean" cannot be concluded
    return 1 if worst > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
