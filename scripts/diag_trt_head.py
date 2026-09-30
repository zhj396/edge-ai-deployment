#!/usr/bin/env python
"""Diagnose whether a TRT INT8 engine's Detect head actually ran FP32.

.. deprecated-in-favor-of:: scripts/diag_qdq_head.py

    For the **default QDQ path** (no ``--calibrator``), this engine-inspector
    approach is **inconclusive** on TRT 10.4 — the inspector returns only layer
    names, no per-layer precision field (CLAUDE.md invariant #16), so it cannot
    tell you whether the head stayed FP32. The authoritative check for the QDQ
    path is the pure ONNX diagnostic ``scripts/diag_qdq_head.py`` (counts Q/DQ
    nodes under ``/model.22/``; pure ``onnx``, no TRT/GPU): head Q/DQ=0 ⟹ the
    head stayed FP32 by QDQ omission. Use THIS script (``diag_trt_head.py``)
    only for the legacy ``--calibrator`` ablation path, where there is no QDQ
    ONNX to inspect and the engine inspector is all you have.

The build log line "Head-exclusion: pinned 70 layer(s) ..." only proves the
precision constraints were *set* before build_serialized_network — not that
the builder honored them at tactic-selection. This script loads the *built*
``.engine`` and uses the TRT engine inspector to dump every layer's real,
post-tactic-selection execution precision, then tallies the precision of the
/model.22/ head layers specifically.

Run on the GPU box that built the engine:

    python scripts/diag_trt_head.py models/yolov8s_int8.engine

NOTE on formats: ONELINE in some TRT builds omits the precision token (or
puts it past char 200 of a long fused-layer name). We therefore prefer JSON
(which carries per-layer precision), write the full dump to
``<engine_stem>_inspector.json``, and print head lines UNtruncated with a
precision-token tally so nothing is hidden by truncation.
"""
import json
import sys
from pathlib import Path

HEAD_PREFIX = "/model.22/"
PRECISION_TOKENS = ("INT8", "FP16", "FP32", "FLOAT", "HALF", "INT4")


def _fmt(trt, name):
    LIF = getattr(trt, "LayerInformationFormat", None)
    if LIF is None:
        return None
    return getattr(LIF, name, None)


def _engine_info(inspector, trt):
    """Return (format_name, text) preferring JSON (carries precision)."""
    order = ["JSON", "VERBOSE", "ONELINE"]
    last_err = None
    for nm in order:
        m = _fmt(trt, nm)
        if m is None:
            continue
        try:
            s = inspector.get_engine_information(m)
            if s:
                return nm, s
        except Exception as e:
            last_err = e
    if last_err:
        print(f"  get_engine_information failed on all formats: {last_err}",
              file=sys.stderr)
    return None, None


def _head_lines(lines):
    return [ln for ln in lines if HEAD_PREFIX in ln]


def _tally(lines):
    counts = {t: 0 for t in PRECISION_TOKENS}
    for ln in lines:
        up = ln.upper()
        for t in PRECISION_TOKENS:
            if t in up:
                counts[t] += 1
    return counts


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt not installed — run this on the GPU build box.")
        return 1
    print(f"TRT version: {getattr(trt, '__version__', '?')}")

    for eng_path in sys.argv[1:]:
        p = Path(eng_path)
        print(f"\n=== {p} ===")
        if not p.exists():
            print(f"  missing: {p}")
            continue
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(p.read_bytes())
        if engine is None:
            print("  failed to deserialize engine")
            continue

        inspector = engine.create_engine_inspector()
        try:
            ctx = engine.create_execution_context()
            for setter in ("set_execution_context",):
                if hasattr(inspector, setter):
                    getattr(inspector, setter)(ctx)
                    break
            else:
                if hasattr(inspector, "execution_context"):
                    inspector.execution_context = ctx
        except Exception as e:
            print(f"  (context setup note: {e})", file=sys.stderr)

        fmt_nm, dump = _engine_info(inspector, trt)
        if not dump:
            print("  inspector exposed no layer info via known APIs.")
            continue

        # Persist the full dump so nothing is lost to terminal width/truncation.
        out_json = p.with_name(f"{p.stem}_inspector.json")
        out_json.write_text(dump, encoding="utf-8")
        print(f"  inspector format={fmt_nm}; full dump -> {out_json}")

        # Try to parse as JSON for structured per-layer precision; if it's not
        # JSON (VERBOSE/ONELINE), fall back to line scanning.
        parsed = None
        if fmt_nm == "JSON":
            try:
                parsed = json.loads(dump)
            except Exception:
                parsed = None

        if parsed is not None and isinstance(parsed, dict):
            layers = parsed.get("Layers") or parsed.get("layers") or []
            print(f"  parsed JSON layers: {len(layers)}")
            head_layers = [
                layer for layer in layers
                if any(HEAD_PREFIX in str(v) for v in _flatten(layer))
            ]
            print(f"  /model.22/ head layers: {len(head_layers)}")
            tok = _tally([json.dumps(layer) for layer in head_layers])
            print(f"  head precision-token tally: {tok}")
            print("\n  --- head layer entries (full, JSON) ---")
            for layer in head_layers[:30]:
                print(f"    {json.dumps(layer, ensure_ascii=False)}")
            if len(head_layers) > 30:
                print(f"    ... +{len(head_layers)-30} more (see {out_json})")
        else:
            # Line-based scan (VERBOSE/ONELINE) — full, untruncated.
            lines = [ln for ln in dump.splitlines() if ln.strip()]
            head = _head_lines(lines)
            tok = _tally(head)
            print(f"  lines: {len(lines)}  head lines: {len(head)}")
            print(f"  head precision-token tally: {tok}")
            print("\n  --- head lines (full, untruncated) ---")
            for ln in head[:40]:
                print(f"    {ln}")
            if len(head) > 40:
                print(f"    ... +{len(head)-40} more (see {out_json})")

        # Crisp verdict: does any head layer mention INT8?
        head_text = (
            "".join(json.dumps(layer) for layer in head_layers)
            if parsed else "\n".join(head)
        )
        if "INT8" in head_text.upper():
            print("\n  *** HEAD LAYERS MENTION INT8 — constraints may NOT have "
                  "held (inspect the dump). ***")
        else:
            print("\n  No 'INT8' token in head layer info — but verify the "
                  "format actually carries precision (check the dump): if "
                  "precision is absent, this verdict is inconclusive.")

    return 0


def _flatten(obj):
    """Yield all string leaves of a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten(v)
    else:
        yield obj


if __name__ == "__main__":
    raise SystemExit(main())
