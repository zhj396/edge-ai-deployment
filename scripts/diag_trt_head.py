#!/usr/bin/env python
"""Diagnose whether a TRT INT8 engine's Detect head actually ran FP32.

Scope: the ``--calibrator`` ablation path, where no QDQ ONNX exists to
inspect and the engine inspector is the available probe. For the default QDQ
path (no ``--calibrator``), the ground-truth check is the pure ONNX
diagnostic ``scripts/diag_qdq_head.py`` (counts Q/DQ nodes under
``/model.22/``; pure ``onnx``, no TRT/GPU): head Q/DQ=0 means the head stayed
FP32 by QDQ omission.

On some TRT 10.x builds the inspector dump carries layer names without a
per-layer precision field, which makes this script's tally inconclusive on
such a build; builds whose JSON dump carries per-layer precision give a real
verdict. The build log line "Head-exclusion (calibrator path): pinned N
layer(s) under ['/model.22/'] to FP32 (constraint=...)" records
that the precision constraints were set before build_serialized_network;
whether the builder honored them at tactic selection is visible only in the
built engine. This script loads the built ``.engine``, dumps the inspector
output, and tallies the precision tokens of the /model.22/ head layers.

Run on the GPU box that built the engine:

    python scripts/diag_trt_head.py models/yolov8s_int8.engine

NOTE on formats: ONELINE in some TRT builds omits the precision token (or
puts it past char 200 of a long fused-layer name). The JSON format is
preferred (it carries per-layer precision on builds that expose it), the
full dump is written to ``<engine_stem>_inspector.json`` (``.txt`` when the
format is not JSON), and head lines are
printed UNtruncated with a precision-token tally so nothing is hidden by
truncation.

Exit code: 0 if every checked engine's head layer info carries no INT8
token, 1 if tensorrt is missing or at least one checked engine's head
mentions INT8 (constraints may not have held — inspect the dump), 2 on
usage error, 3 if no engine could be checked — a run that checked nothing
never reports clean.
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

    flagged = 0    # engines whose head info mentions INT8
    checked = 0    # engines whose inspector dump was actually examined
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
        fmt_nm, dump = _engine_info(inspector, trt)
        if not dump:
            print("  inspector exposed no layer info via known APIs.")
            continue
        checked += 1

        # Persist the full dump so nothing is lost to terminal
        # width/truncation; only a JSON-format dump is a JSON file.
        dump_suffix = ".json" if fmt_nm == "JSON" else ".txt"
        out_dump = p.with_name(f"{p.stem}_inspector{dump_suffix}")
        out_dump.write_text(dump, encoding="utf-8")
        print(f"  inspector format={fmt_nm}; full dump -> {out_dump}")

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
                print(f"    ... +{len(head_layers)-30} more (see {out_dump})")
            head_text = "".join(json.dumps(layer) for layer in head_layers)
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
                print(f"    ... +{len(head)-40} more (see {out_dump})")
            head_text = "\n".join(head)

        # Crisp verdict: does any head layer mention INT8?
        if "INT8" in head_text.upper():
            print("\n  *** HEAD LAYERS MENTION INT8 — constraints may NOT have "
                  "held (inspect the dump). ***")
            flagged += 1
        else:
            print("\n  No 'INT8' token in head layer info — but verify the "
                  "format actually carries precision (check the dump): if "
                  "precision is absent, this verdict is inconclusive.")

    if checked == 0:
        return 3  # nothing was checked — "clean" cannot be concluded
    return 1 if flagged else 0


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
