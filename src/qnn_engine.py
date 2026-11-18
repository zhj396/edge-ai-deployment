"""QNN (Qualcomm AI Engine) backend for YOLOv8s — on-device HTP execution.

The backend executes on the AR1+ smart-glasses SoC's Hexagon HTP (v73) through
a *host-driven device workflow*: QAIRT host tools convert the FP32 ONNX to a
PTQ context binary (``qnn-onnx-converter`` → ``qnn-model-lib-generator`` →
``qnn-context-binary-generator``), the context + inputs are pushed over adb,
and ``qnn-net-run`` executes on the device. Unlike every other backend in this
toolchain, the accelerator is *not on this host* — the v73 Hexagon context
only deserializes on the device's DSP — so there is no in-process engine and
no subprocess exe: the integration surface is the device filesystem plus adb.

Two structural facts shape the API:

1. **NHWC graph layout.** QNN's native tensor layout is NHWC (the HTP-optimized
   layout). ``qnn-net-run`` consumes ``--input_list`` raws *in graph layout*
   (`[1, 640, 640, 3]`); NCHW-ordered raws get pixel-interleaved and the graph
   computes "a correct forward of the wrong input" (outputs look statistically
   sane but correlate ~0 with the reference — verified with a single-Conv
   minimal graph, bit-exact once re-ordered). All input raws are therefore
   written NHWC (``nchw.transpose(1, 2, 0)``).

2. **Split box / cls outputs.** A single merged output tensor carries ONE
   per-tensor quantization encoding: dominated by the box range (0–640), the
   cls scores (sigmoid, ≤ ~0.12 on this val set) quantize below 1 LSB and
   dequant to exactly 0.0. The conversion therefore names two outputs
   (``--out_node /model.22/Mul_5_output_0`` for box, ``--out_node
   /model.22/Sigmoid_output_0`` for cls), each with its own encoding; this
   engine re-concatenates them into the ``[bs, 4+nc, N]`` layout
   ``post_process`` consumes.

Quantization route: PTQ via ``--input_list`` (the shared
:class:`src.sampler.CalibrationSampler`-selected images, letterboxed through
the shared :func:`src.preprocess.preprocess_imgs`, written NHWC). QDQ-import
and FP32 host compilation are blocked by the QAIRT 2.42 host compiler
(``q::flat_from_vtcm`` at the Resize lowering; ``q::ForceFormat_Crouton`` on a
ConvTranspose rewrite) — see docs/QNN.md. W8A16 (``--act_bitwidth 16``) is the
head-protection configuration: weights stay u8, activations go fp16.

Optional-dep posture mirrors the OpenVINO/TensorRT engines: this module
imports cleanly without QAIRT; the availability predicate probes the *host
toolchain* (converter binaries + a reachable adb device). Constructing the
runner without them raises ``ImportError`` with the setup hint, never
``AttributeError``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from utils import get_logger
from .openvino_engine import _class_names_from_data_yaml

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------
# QAIRT SDK root: the zip-layout install (qairt/<ver>/{bin,lib}/...). QNN_SDK_ROOT
# also honors Qualcomm Package Manager's layout when it exists.
def resolve_qnn_sdk_root() -> str:
    """Locate the QAIRT SDK root (host converters + aarch64-android libs).

    Search order: ``QNN_SDK_ROOT`` env → ``~/qairt/qairt/<latest-version>``.
    Raises ``RuntimeError`` with the setup hint when absent.
    """
    env = os.environ.get("QNN_SDK_ROOT")
    if env and Path(env).is_dir():
        return str(Path(env).resolve())

    home = Path.home() / "qairt" / "qairt"
    if home.is_dir():
        versions = sorted(p for p in home.iterdir() if p.is_dir())
        if versions:
            return str(versions[-1].resolve())

    raise RuntimeError(
        "QAIRT SDK not found. Set QNN_SDK_ROOT to the SDK root (the "
        "directory containing bin/x86_64-linux-clang/qnn-onnx-converter "
        "and lib/aarch64-android/), or extract the QAIRT zip under "
        "~/qairt/qairt/<version>/. See docs/QNN.md."
    )


def resolve_adb() -> str:
    """Locate an adb client. ``QNN_ADB`` env → ``adb`` on PATH."""
    env = os.environ.get("QNN_ADB")
    if env and Path(env).is_file():
        return env
    adb = shutil.which("adb")
    if adb:
        return adb
    raise RuntimeError(
        "adb not found. Install platform-tools and put adb on PATH, or set "
        "QNN_ADB to the adb binary path. See docs/QNN.md."
    )


def adb_device_serial() -> str:
    """First ``device``-state serial from adb, or ``QNN_DEVICE_SERIAL`` pin.

    Raises ``RuntimeError`` with a "no device" hint when nothing is attached.
    """
    pin = os.environ.get("QNN_DEVICE_SERIAL")
    adb = resolve_adb()
    out = subprocess.run(
        [adb, "devices"], capture_output=True, text=True, timeout=30
    )
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device":
            if pin is None or parts[0] == pin:
                return parts[0]
    raise RuntimeError(
        "no ADB device in 'device' state"
        + (f" (serial {pin!r} pinned via QNN_DEVICE_SERIAL)" if pin else "")
        + ". Connect the device and authorize USB debugging. See docs/QNN.md."
    )


def qnn_available() -> bool:
    """True iff the host toolchain (SDK + adb) and a device are reachable.

    Cached probe semantics match the other engines: the result is cached at
    process level; use ``reset_qnn_probe()`` to re-probe after the
    environment changes.
    """
    try:
        resolve_qnn_sdk_root()
        resolve_adb()
        adb_device_serial()
        return True
    except Exception:
        return False


def reset_qnn_probe() -> None:
    """Clear the cached ``qnn_available()`` result."""
    # The predicate recomputes every call (no cache storage); kept for API
    # parity with trt_cpp_available/reset so callers need no special case.
    return None


def _qnn_env_hint() -> str:
    return (
        "QNN host toolchain not available (QAIRT SDK and/or adb missing, or "
        "no device attached). Set QNN_SDK_ROOT, install platform-tools, "
        "connect the device and authorize USB debugging. See docs/QNN.md."
    )


# ---------------------------------------------------------------------------
# Raw tensor layout helpers (shared by conversion staging and output loading)
# ---------------------------------------------------------------------------
def nchw_batch_to_nhwc_raws(
    batch_nchw: np.ndarray, out_dir: Union[str, Path], stem: str = "input",
) -> List[Path]:
    """Write an NCHW batch ``(bs, C, H, W)`` as one NHWC ``.raw`` per image.

    QNN graph input layout is NHWC (`[1, 640, 640, 3]`); ``qnn-net-run`` reads
    ``--input_list`` raws in graph layout with no transposition, so a NCHW
    byte order would be pixel-interleaved into garbage. The file order in
    ``out_dir`` follows the batch order (zero-padded ``<stem>_NNN.raw``).
    Returns the written paths in batch order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = batch_nchw.shape[0]
    width = max(3, len(str(n - 1)))
    paths: List[Path] = []
    for i in range(n):
        nhwc = np.ascontiguousarray(batch_nchw[i].transpose(1, 2, 0))
        p = out_dir / f"{stem}_{i:0{width}d}.raw"
        nhwc.tofile(str(p))
        paths.append(p)
    return paths


def load_split_raw_outputs(
    result_dir: Union[str, Path], nc: int, num_anchors: int = 8400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one image's split raw outputs → ``(box, cls)`` arrays.

    The context binary carries two outputs (box ``(1, 4, N)`` from
    ``/model.22/Mul_5_output_0``, cls ``(1, nc, N)`` from
    ``/model.22/Sigmoid_output_0``); ``qnn-net-run`` writes them as
    ``Result_N/<tensor_name>.raw`` (float32, dequantized). Tensor names are
    globbed (the IR flattening ``/`` → ``_`` varies with the source graph).
    The output raws come back **channel-first** — measured both ways on the
    same bytes (channel-first corr 0.9988 vs NHWC-transpose −0.111 against
    the ORT FP32 reference), in contrast to the NHWC input contract.
    """
    result_dir = Path(result_dir)
    box_files = sorted(result_dir.glob("*Mul_5_output_0.raw"))
    cls_files = sorted(result_dir.glob("*Sigmoid_output_0.raw"))
    if len(box_files) != 1 or len(cls_files) != 1:
        raise RuntimeError(
            f"expected one box + one cls raw in {result_dir}, found "
            f"{[p.name for p in result_dir.glob('*.raw')]}"
        )
    box = np.fromfile(str(box_files[0]), dtype=np.float32)
    cls = np.fromfile(str(cls_files[0]), dtype=np.float32)
    if box.size != 4 * num_anchors:
        raise RuntimeError(
            f"box raw size {box.size} != 4*{num_anchors} in {result_dir}"
        )
    if cls.size != nc * num_anchors:
        raise RuntimeError(
            f"cls raw size {cls.size} != nc({nc})*{num_anchors} in {result_dir}"
        )
    # Output raws come back channel-first (measured: channel-first corr 0.9988
    # vs NHWC-transpose −0.111 against the ORT FP32 reference — the opposite of
    # the NHWC input contract).
    box = box.reshape(4, num_anchors).copy()
    cls = cls.reshape(nc, num_anchors).copy()
    return box, cls


def split_outputs_to_4nc(
    result_dirs: List[Union[str, Path]], nc: int, num_anchors: int = 8400,
) -> np.ndarray:
    """Stack per-image split outputs into the ``[bs, 4+nc, N]`` layout."""
    boxes: List[np.ndarray] = []
    clss: List[np.ndarray] = []
    for d in result_dirs:
        b, c = load_split_raw_outputs(d, nc, num_anchors)
        boxes.append(b)
        clss.append(c)
    return np.stack(
        [np.concatenate([b, c], axis=0) for b, c in zip(boxes, clss)], axis=0
    )


# ---------------------------------------------------------------------------
# Device runner
# ---------------------------------------------------------------------------
class QNNEngineRunner:
    """Drive one qnn-net-run execution on the connected device.

    Not a ``YOLOv8Engine``-shaped engine (the HTP is not on this host): the
    runner stages inputs, runs ``qnn-net-run`` over adb, and loads the split
    raw outputs back. The benchmark / consistency harnesses call
    :meth:`forward_batch` through the adapter described in docs/QNN.md;
    ``infer``-style end-to-end runs go through the same method inside the
    harness's own batch loop.

    Parameters mirror the other engines where they apply (``imgsz``,
    ``class_names`` / ``data_yaml``); ``device`` is accepted-and-ignored —
    the accelerator selection is the context binary itself (HTP), and
    ``--device`` semantics ("where preprocessing runs") stay host-side CPU.
    """

    DEVICE_PKG_DIR = "/data/local/tmp/qnn_pkg"

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        device: Union[str, int] = "cpu",
        class_names: Optional[dict] = None,
        data_yaml: Optional[Union[str, Path]] = None,
        device_serial: Optional[str] = None,
    ) -> None:
        if not qnn_available():
            raise ImportError(_qnn_env_hint())

        self.model_path = str(model_path)
        self.imgsz = int(imgsz)
        self.device_serial = device_serial or adb_device_serial()
        # The accelerator is the context binary's target (HTP); --device has
        # no device-side meaning here. Kept as an attribute for harness
        # parity (the rows report the real target through ``describe_device``).
        self.device = device
        self.class_names = class_names or (
            _class_names_from_data_yaml(data_yaml) if data_yaml else None
        )
        self.nc = len(self.class_names) if self.class_names else None
        if not self.nc:
            raise ValueError(
                "QNN needs class names: pass class_names or data_yaml (a "
                "context binary carries no names metadata)"
            )
        self._adb = resolve_adb()

    # ─── adb helpers ──────────────────────────────────────────────────────
    def _serial_args(self) -> List[str]:
        return ["-s", self.device_serial]

    def _shell(self, command: str, timeout: int = 600) -> str:
        proc = subprocess.run(
            [self._adb, *self._serial_args(), "shell", command],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"adb shell failed (exit {proc.returncode}): {command}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )
        return proc.stdout

    def _push(self, local: str, remote: str) -> None:
        proc = subprocess.run(
            [self._adb, *self._serial_args(), "push", local, remote],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"adb push failed ({local!r} → {remote!r}): {proc.stderr[-2000:]}"
            )

    def _pull(self, remote: str, local: str) -> None:
        proc = subprocess.run(
            [self._adb, *self._serial_args(), "pull", remote, local],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"adb pull failed ({remote!r} → {local!r}): {proc.stderr[-2000:]}"
            )

    def describe_device(self) -> str:
        """Human-readable execution-target string for result rows."""
        model = self._shell("getprop ro.soc.model").strip() or "unknown-soc"
        return f"qnn_htp ({model}, adb {self.device_serial})"

    # ─── staging ──────────────────────────────────────────────────────────
    def ensure_context_staged(self, context_bin: Union[str, Path]) -> str:
        """Push the context binary + input list skeleton; return the remote
        context path. Idempotent (a re-push overwrites in place)."""
        self._shell(
            f"mkdir -p {self.DEVICE_PKG_DIR} && "
            f"chmod 777 {self.DEVICE_PKG_DIR}"
        )
        remote_ctx = (
            f"{self.DEVICE_PKG_DIR}/{Path(context_bin).name}"
        )
        self._push(str(context_bin), remote_ctx)
        self._shell(f"chmod 666 {remote_ctx}")
        return remote_ctx

    # ─── forward ──────────────────────────────────────────────────────────
    def forward_batch(
        self, batch_nchw: np.ndarray, context_bin: Union[str, Path],
        num_anchors: int = 8400, keep_outputs: bool = False,
    ) -> np.ndarray:
        """Run one NHWC-staged forward on the device, return ``[bs, 4+nc, N]``.

        The batch (``(bs, C, imgsz, imgsz)`` float32 NCHW, the shared
        preprocess output) is written as NHWC raws (see
        :func:`nchw_batch_to_nhwc_raws`), pushed, executed through one
        ``qnn-net-run`` invocation, and the split ``Result_N`` outputs are
        pulled back and re-assembled channel-first.
        """
        nc = self.nc
        bs = batch_nchw.shape[0]
        self.ensure_context_staged(context_bin)

        with tempfile.TemporaryDirectory(prefix="qnn_fwd_") as tmp:
            local_in = Path(tmp) / "inputs"
            local_out = Path(tmp) / "outputs"
            raws = nchw_batch_to_nhwc_raws(batch_nchw, local_in)

            remote_in = f"{self.DEVICE_PKG_DIR}/fwd_in"
            remote_out = f"{self.DEVICE_PKG_DIR}/fwd_out"
            self._shell(
                f"rm -rf {remote_in} {remote_out} && mkdir -p {remote_in}"
            )
            for i, p in enumerate(raws):
                self._push(str(p), f"{remote_in}/{p.name}")
            list_lines = "\n".join(
                f"{remote_in}/{p.name}" for p in raws
            )
            self._shell(f"printf '%s\\n' {list_lines!r} > "
                        f"{self.DEVICE_PKG_DIR}/fwd_list.txt")

            ld = f"LD_LIBRARY_PATH={self.DEVICE_PKG_DIR}:/system_ext/lib64"
            adsp = (
                f"ADSP_LIBRARY_PATH={self.DEVICE_PKG_DIR};/system_ext/lib64;"
                f"/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
            )
            cmd = (
                f"export {ld} && export {adsp!r} && "
                f"cd {self.DEVICE_PKG_DIR} && "
                f"./qnn-net-run --retrieve_context {Path(context_bin).name} "
                f"--input_list fwd_list.txt --output_dir fwd_out "
                f"--backend libQnnHtp.so"
            )
            out = self._shell(cmd, timeout=900)
            if "Finished Executing Graphs" not in out:
                raise RuntimeError(
                    f"qnn-net-run did not complete on device:\n{out[-2000:]}"
                )
            self._pull(remote_out, str(local_out))

            result_dirs = sorted(local_out.glob("Result_*"))[:bs]
            if len(result_dirs) != bs:
                raise RuntimeError(
                    f"expected {bs} Result_* dirs, got {len(result_dirs)}"
                )
            full = split_outputs_to_4nc(
                [d for d in result_dirs], nc, num_anchors
            )
            if not keep_outputs:
                self._shell(f"rm -rf {remote_in} {remote_out}")
            return full
