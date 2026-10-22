"""OpenVINO inference backend for YOLOv8s.

OpenVINO loads either:
* **OpenVINO IR** (``.xml`` + ``.bin``)
* **ONNX** (``.onnx``) — automatically converted at load time

The engine exposes the same ``infer()`` / ``infer_frames()`` API surface as
``YOLOv8Engine``. ``benchmark`` drives it through that surface via the
``openvino:`` / ``openvino_int8:`` model prefixes; the docker server's
``infer_frames`` hot path calls either engine identically — only the
constructor differs (``YOLOv8Engine(backend=...)`` vs ``OpenVINOEngine(device=...)``).

Served devices follow the installed OpenVINO build's plugin set: CPU, GPU
(Intel iGPU / Arc, enumerated through the Level Zero driver), and the
``AUTO`` / ``MULTI:`` / ``HETERO:`` selection forms. Device availability
is preflighted at engine init (see :func:`validate_device_request`).
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from utils import get_logger, load_images, save_annotated_image
from . import post_process, preprocess_imgs, preprocess_frames

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional import — OpenVINO is a heavy dep that some users don't want.
#
# Core / AsyncInferQueue are resolved by attribute (top-level first, then the
# runtime submodule) so both version layouts import cleanly:
#   - <= 2022.x : openvino.runtime.Core / openvino.runtime.AsyncInferQueue
#   - 2023.1+   : openvino.Core / openvino.AsyncInferQueue  (top-level)
# ``import openvino as ov`` works on every version — the converter uses the
# same form.
# ---------------------------------------------------------------------------
try:
    import openvino as ov  # used for Core/AsyncInferQueue attr resolution + fallback
    Core = getattr(ov, "Core", None)
    if Core is None:  # <= 2022 layout
        Core = getattr(getattr(ov, "runtime", None), "Core", None)
    AsyncInferQueue = (
        getattr(ov, "AsyncInferQueue", None)
        or getattr(getattr(ov, "runtime", None), "AsyncInferQueue", None)
    )
    if Core is None:
        raise ImportError("openvino is installed but Core was not found")
    _OPENVINO_AVAILABLE = True
except ImportError:  # pragma: no cover
    ov = None
    Core = None
    AsyncInferQueue = None
    _OPENVINO_AVAILABLE = False


def openvino_available() -> bool:
    """True if the ``openvino`` package is importable *and* exposes Core."""
    return _OPENVINO_AVAILABLE


def _component_is_enumerated(name: str, available: List[str]) -> bool:
    """True when ``name`` is served by an enumerated device.

    Multi-tile devices enumerate per tile (``GPU.0`` / ``GPU.1``); a plain
    ``GPU`` request compiles against the first tile, so a tile-suffixed
    enumeration serves the plain name.
    """
    return any(d == name or d.startswith(name + ".") for d in available)


def _request_is_servable(requested: str, available: List[str]) -> bool:
    """True when OpenVINO can serve ``requested`` given ``available``.

    Rules:

    * ``AUTO`` and the ``AUTO:`` priority form are always satisfiable —
      they dispatch across the enumerated devices (CPU included), so no
      preflight applies.
    * A plain name (``CPU`` / ``GPU`` / ...) is served when an enumerated
      device matches it exactly or as a tile instance (``GPU`` is served
      by ``GPU.0`` / ``GPU.1``).
    * ``MULTI:`` / ``HETERO:`` / ``BATCH:`` compound forms need **every
      named component** enumerated, so a partially-available compound
      fails the preflight with an actionable message; ``compile_model``'s
      own error for an absent device is an opaque C++ exception.

    Pure function (no Core needed) so the contract is unit-testable without
    openvino hardware — see tests/test_openvino_optional_imports.py.
    """
    prefix, sep, rest = requested.partition(":")
    if requested == "AUTO":
        return True
    if not sep:
        return _component_is_enumerated(requested, available)
    if prefix == "AUTO":
        return True
    if prefix in ("MULTI", "HETERO", "BATCH"):
        parts = [p for p in rest.split(",") if p]
        # Every named component must be enumerated — a partially-available
        # compound (e.g. MULTI:CPU,GPU with no GPU) fails here, at
        # preflight time, with an actionable message.
        return bool(parts) and all(
            _component_is_enumerated(p, available) for p in parts
        )
    return _component_is_enumerated(requested, available)


def validate_device_request(
    requested: str,
    available: List[str],
    fallback: Optional[str] = "CPU",
) -> str:
    """Resolve ``requested`` against ``available``, falling back explicitly.

    ``compile_model`` raises an opaque C++ exception for an absent device
    (e.g. ``[GPU] Can't get PERFORMANCE_HINT property as no supported
    devices found``), even though ``core.available_devices`` already showed
    the request was impossible. This preflight returns the device string
    the caller should actually compile for:

    * ``requested`` itself when it is servable (see
      :func:`_request_is_servable`);
    * otherwise ``fallback`` (default ``"CPU"``) — with a
      ``logger.warning`` naming the unavailable request, the enumerated
      devices, and a driver hint. The fallback is explicit and logged:
      ``engine.device`` records what actually ran, so a benchmark or
      consistency run is identifiable as a fallback run from its logs;
    * ``RuntimeError`` when neither the request nor the fallback is
      available (or ``fallback=None``, which restores the strict
      reject-on-absent-device behavior).

    Pure function (no Core needed) so the contract is unit-testable without
    openvino hardware — see tests/test_openvino_optional_imports.py.
    """
    if _request_is_servable(requested, available):
        return requested

    hint = ""
    if requested == "GPU":
        # The overwhelmingly common cause on a dev box: no Intel GPU
        # driver visible to the OpenVINO GPU plugin (Level Zero).
        hint = (
            " For Intel GPU: the plugin needs a driver-enumerated device"
            " — on WSL2 install the Windows Intel GPU driver with WSL"
            " compute support and check that /dev/dxg exists; on bare"
            " Linux check /dev/dri/renderD*."
        )
    prefix, sep, rest = requested.partition(":")
    is_compound = bool(sep) and prefix in ("MULTI", "HETERO")

    if fallback and fallback != requested and fallback in available:
        logger.warning(
            "Device %r is not available to OpenVINO on this host "
            "(enumerated devices: %s).%s Explicitly falling back to %r — "
            "rerun with --device %s to silence this warning, or install "
            "the missing driver to use the requested device.",
            requested, available, hint, fallback, fallback,
        )
        return fallback

    if is_compound:
        raise RuntimeError(
            f"Device request {requested!r} names component(s) unavailable "
            f"to OpenVINO on this host (enumerates: {available}). Drop the "
            f"unavailable components or use --device CPU."
        )
    raise RuntimeError(
        f"Device {requested!r} is not available to OpenVINO on this host "
        f"(enumerated devices: {available}).{hint} Rerun with "
        f"--device CPU or --device AUTO."
    )


def _class_names_from_data_yaml(data_yaml):
    """Parse the ``names`` field of a YOLO ``data.yaml`` -> ``{int: str}``.

    OpenVINO IR conversion (``ov.convert_model``) drops the ultralytics
    ``names`` metadata the ONNX carried, so an IR can't self-report its
    class names the way an ONNX session can (``_class_names_from_session``).
    The ``data.yaml`` is the project's canonical class list; this mirrors
    ``CalibrationSampler._load_dataset``'s names normalization (list ->
    dict). Returns ``None`` if no ``names`` field is present.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover — pyyaml is an ultralytics dep
        raise ImportError(
            "pyyaml is required to read class names from data.yaml: "
            f"{e}. pip install pyyaml"
        ) from e
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class OpenVINOEngine:
    """OpenVINO inference engine for YOLOv8.

    Parameters
    ----------
    model_path
        Path to ``.xml`` (with matching ``.bin`` next to it) or ``.onnx``.
    device
        ``CPU`` / ``GPU`` / ``AUTO`` / ``MULTI:CPU,GPU`` /
        ``HETERO:CPU,GPU``. Case-insensitive; we uppercase it before
        passing to OpenVINO.
    imgsz
        Square network input size (default 640). Must match the IR's
        conversion ``--imgsz`` for a statically-shaped IR — enforced at
        construction (see :attr:`spatial_size`); a dynamically-shaped IR
        accepts any size.
    num_streams
        Throughput hint — number of CPU inference streams. ``"AUTO"``
        lets OpenVINO pick based on core count.
    class_names
        Optional override. Otherwise resolved in order: ONNX metadata
        (``names``) when loading from ``.onnx``, then ``data_yaml`` when
        given (IRs carry no ultralytics names metadata), then the generic
        ``class_0..class_11`` fallback.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        imgsz: int = 640,
        num_streams: Union[int, str] = "AUTO",
        class_names: Optional[dict] = None,
        data_yaml: Optional[Union[str, Path]] = None,
    ) -> None:
        if not _OPENVINO_AVAILABLE:
            raise ImportError(
                "openvino is not installed. Run: pip install -r requirements-openvino.txt"
            )

        self.model_path = str(model_path)
        # OpenVINO's read_model takes the .xml and auto-loads the sibling .bin;
        # the .bin alone is not a valid input. If the user pointed at the .bin
        # (it's the big, obvious file), redirect to the .xml so the call works.
        if self.model_path.lower().endswith(".bin"):
            xml = self.model_path[: -len(".bin")] + ".xml"
            if Path(xml).exists():
                logger.info(
                    "OpenVINO reads the .xml (auto-loads .bin); "
                    "redirecting %s -> %s", self.model_path, xml,
                )
                self.model_path = xml
            else:
                raise FileNotFoundError(
                    f"OpenVINO IR .xml not found next to {self.model_path} "
                    f"(expected {xml}). Pass the .xml, not the .bin."
                )
        self.device = device.upper()
        self.imgsz = imgsz
        self.num_streams = num_streams
        self.class_names = class_names

        self.core = Core()
        logger.info(
            "OpenVINO devices available: %s",
            self.core.available_devices,
        )
        # Preflight the device request against the enumerated devices so
        # an unservable request fails or falls back here, at engine init,
        # with an actionable message (see validate_device_request). An
        # unservable request (e.g. GPU with no Intel GPU driver) resolves
        # to an explicit, warned CPU fallback; self.device records what
        # actually runs, so logs record the device that executed.
        self.device = validate_device_request(
            self.device, list(self.core.available_devices)
        )

        # Read model (IR or ONNX).  ``read_model`` accepts both.
        self.model = self.core.read_model(self.model_path)
        logger.info(
            "Loaded OpenVINO model | inputs=%d | outputs=%d",
            len(self.model.inputs), len(self.model.outputs),
        )

        # Compile for the requested device. NUM_STREAMS is stringly-typed in
        # OpenVINO's property API — any int is normalized to str here, so
        # programmatic callers can pass num_streams=4 directly.
        self.compiled = self.core.compile_model(
            self.model, device_name=self.device,
            config={"NUM_STREAMS": str(self.num_streams)}
            if self.num_streams else {},
        )

        # Discover I/O names + shapes
        self.input_tensor = self.compiled.input(0)
        self.input_name = self.input_tensor.any_name
        # Resolve the model's per-forward batch from ``.partial_shape``:
        # ``.partial_shape[0]`` is a Dimension whose get_length() returns the
        # int for a static dim and raises for a dynamic one, so the try maps
        # static → int (e.g. a static-batch IR) and dynamic
        # → None ('?' — the FP16 IR, DFL [-1,4,16,8400], runs batch>1
        # natively; _effective_batch then honors --batch-size).
        # ``partial_shape`` is the one accessor that works on both static and
        # dynamic IRs.
        try:
            bdim = self.input_tensor.partial_shape[0]
            self._model_batch = int(bdim.get_length())  # static dim -> int
        except Exception:
            self._model_batch = None  # dynamic ('?') — IR accepts any batch
        self._batch_mismatch_warned = False
        logger.info(
            "Model input batch: %s",
            "dynamic" if self._model_batch is None else f"static {self._model_batch}",
        )
        # The IR's spatial dims are pinned at conversion (see
        # :attr:`spatial_size`) — a statically-shaped IR serves exactly one
        # image size. Reject a mismatched imgsz here, with both sizes named.
        if self.spatial_size is not None and self.imgsz != self.spatial_size:
            raise ValueError(
                f"OpenVINO IR {self.model_path} was converted for "
                f"imgsz={self.spatial_size} (spatial dims are pinned at "
                f"conversion — see docs/OPENVINO.md) and cannot serve "
                f"imgsz={self.imgsz}. Re-convert the IR at the requested "
                f"size, or pass imgsz={self.spatial_size}."
            )
        self.output_names = [out.any_name for out in self.compiled.outputs]
        # Output port for sync infer_request.get_tensor()
        self._output_ports = [out for out in self.compiled.outputs]

        # Class names: ONNX metadata (if loaded from .onnx) -> data.yaml ->
        # generic class_i fallback. The data.yaml path is needed for an IR
        # (.xml): ov.convert_model drops the ultralytics ``names`` metadata
        # the ONNX carried, so an IR can't self-report its class names the
        # way an ONNX session can (_class_names_from_session). data.yaml is
        # the project's canonical class list.
        if self.class_names is None and self.model_path.endswith(".onnx"):
            try:
                import onnx
                m = onnx.load(self.model_path)
                meta = {p.key: p.value for p in m.metadata_props}
                if "names" in meta:
                    # literal_eval, not eval — the metadata string is untrusted
                    # model content. Mirrors YOLOv8Engine's
                    # _class_names_from_session.
                    parsed = literal_eval(meta["names"])
                    if isinstance(parsed, dict):
                        self.class_names = {int(k): v for k, v in parsed.items()}
                    elif isinstance(parsed, list):
                        self.class_names = {i: n for i, n in enumerate(parsed)}
            except Exception:
                pass
        if self.class_names is None and data_yaml is not None:
            try:
                parsed = _class_names_from_data_yaml(data_yaml)
                if parsed:
                    self.class_names = parsed
                    logger.info("Class names from data.yaml: %s", data_yaml)
            except Exception as e:
                logger.warning(
                    "Could not read class names from %s: %s", data_yaml, e,
                )
        if self.class_names is None:
            self.class_names = {i: f"class_{i}" for i in range(12)}

        # Warm-up once — OpenVINO JIT-compiles on first call.
        self._warmup()
        logger.info("OpenVINO engine initialized | device=%s", self.device)

    # ------------------------------------------------------------- spatial size
    @property
    def spatial_size(self) -> Optional[int]:
        """Static square input size (H == W) the compiled model accepts.

        The converter pins the IR's spatial dims to its conversion
        ``--imgsz`` (docs/OPENVINO.md), so a static answer here is the
        *only* image size this IR can serve. ``None`` means at least one
        spatial dim is dynamic (any size accepted) or non-square (the
        toolchain never produces one). Read from ``partial_shape`` for
        the same reason as ``_model_batch`` in ``__init__``: the plain
        ``.shape`` accessor raises on a dynamic shape, while
        ``partial_shape`` reads static and dynamic IRs alike.
        """
        try:
            ps = self.input_tensor.partial_shape
            h, w = ps[2].get_length(), ps[3].get_length()
            return int(h) if h == w else None
        except Exception:
            return None

    # ----------------------------------------------------------------- warmup
    def _warmup(self, n_iters: int = 3) -> None:
        logger.info("Warmup (OpenVINO JIT-compiles on first call)...")
        # Use the model's own per-forward batch so a static-batch IR (e.g.
        # batch=1) is exercised with the shape it actually accepts.
        b = self._model_batch if self._model_batch else 1
        dummy = np.random.rand(b, 3, self.imgsz, self.imgsz).astype(np.float32)
        # The first call triggers JIT compilation; fail init on it so a bad
        # config (wrong --device, --imgsz mismatch, IR-vs-device incompat)
        # surfaces at construction with the full context.
        try:
            self.compiled([dummy])
        except Exception as e:
            raise RuntimeError(
                f"OpenVINO warmup failed for device={self.device} with "
                f"input shape {dummy.shape}: {e}. Check --device, --imgsz, "
                f"and that the IR was converted for this device."
            ) from e
        # Subsequent iters just amortize the compiled kernel (cache priming /
        # thread pool spin-up). A failure here is odd but non-fatal — the
        # engine is usable; don't poison init over a warmup-only hiccup.
        for _ in range(n_iters - 1):
            try:
                self.compiled([dummy])
            except Exception as e:  # pragma: no cover — non-fatal warmup hiccup
                logger.warning("Non-first warmup call failed (ignored): %s", e)
                break

    # ------------------------------------------------------------- run batch
    def _pad_to_static_batch(self, batch_np: np.ndarray):
        """Zero-pad ``batch_np`` up to ``_model_batch``; return ``(batch, real_n)``.

        A **static-batch** IR's DFL reshape constant is baked to exactly
        ``_model_batch`` — a partial batch (the *tail* of a sub-batched loop,
        see :func:`_effective_batch`) crashes mid-graph ("shape of input data
        ... conflicts with reshape pattern [1,4,16,8400]"). Shared by the
        sync :func:`_forward` and the async :meth:`OpenVINOAsyncEngine.start_batch`
        so the engine-level invariant — *no partial batch ever reaches the
        graph, no image is ever dropped* — holds on BOTH execution paths.

        For dynamic IRs (``_model_batch is None``) and full batches this is a
        pass-through, so callers can unconditionally slice outputs back with
        ``[:real_n]``.
        """
        real_n = batch_np.shape[0]
        if self._model_batch is not None and real_n < self._model_batch:
            pad = self._model_batch - real_n
            batch_np = np.concatenate(
                [
                    batch_np,
                    np.zeros(
                        (pad,) + batch_np.shape[1:],
                        dtype=batch_np.dtype,
                    ),
                ],
                axis=0,
            )
        return batch_np, real_n

    # ----------------------------------------------------------------- forward
    def _forward(self, batch_np: np.ndarray) -> np.ndarray:
        """Run inference and return numpy output of the (single) head.

        Static-batch tail handling (zero-pad + slice back to ``real_n``)
        lives in :func:`_pad_to_static_batch`; full batches and dynamic IRs
        take the pass-through path (no pad, no slice beyond ``[:real_n]``).
        """
        batch_np, real_n = self._pad_to_static_batch(batch_np)
        result = self.compiled([batch_np])
        if len(self._output_ports) != 1:
            # Same posture as YOLOv8Engine._run_batch's output-type guard: a
            # YOLOv8s export is expected to expose exactly one output.
            raise RuntimeError(
                f"Expected exactly one OpenVINO output, got "
                f"{len(self._output_ports)}"
            )
        out = np.asarray(result[self._output_ports[0]])
        return out[:real_n]

    # ------------------------------------------------------------- run batch
    def _run_batch(
        self,
        data: Dict,
        conf: float,
        iou: float,
        max_det: int,
        save: bool,
        output_dir: Path,
    ) -> List[List[tuple]]:
        """Forward + post_process on one preprocessed batch.

        Shared by ``infer`` (file-backed) and ``infer_frames`` (array-backed)
        so both entry points run the same forward/post-process
        body — mirrors
        ``YOLOv8Engine._run_batch``. ``data`` must already be sized to the
        model's per-forward batch (see ``_effective_batch``) — a static-batch
        IR rejects any other count at the DFL reshape.
        """
        outputs = self._forward(data["images"].cpu().numpy())
        # Explicit check (not ``assert``) so it survives ``python -O``; matches
        # YOLOv8Engine._run_batch's TypeError guard on the output shape.
        if not isinstance(outputs, np.ndarray) or outputs.ndim != 3:
            raise TypeError(
                f"Expected a 3-D OpenVINO output [bs, 4+nc, N], got "
                f"{type(outputs).__name__} ndim="
                f"{getattr(outputs, 'ndim', '?')}"
            )
        batch_dets = post_process(
            outputs=outputs,
            orig_shapes=data["orig_shapes"],
            conf_thres=conf,
            iou_thres=iou,
            imgsz=self.imgsz,
            max_det=max_det,
            ratios=data["ratios"],
            pads=data["pads"],
            # Explicit nc so NMS splits box/cls channels authoritatively;
            # the deployed path does not depend on the magnitude layout
            # heuristic (see src/postprocess._ensure_4nc_first). Mirrors
            # YOLOv8Engine.
            nc=len(self.class_names) if self.class_names else None,
        )

        if save:
            for j, dets in enumerate(batch_dets):
                if not dets:
                    continue
                save_path = output_dir / f"result_{Path(data['paths'][j]).name}"
                if not save_path.suffix:
                    # Synthetic frame paths carry no extension; cv2.imwrite
                    # picks the encoder from the file extension.
                    save_path = save_path.with_suffix(".jpg")
                save_annotated_image(
                    orig_img=data["orig_imgs"][j],
                    detections=dets,
                    save_path=str(save_path),
                    class_names=self.class_names,
                )
        return batch_dets

    # -------------------------------------------------------- effective batch
    def _effective_batch(self, requested: int) -> int:
        """Per-forward image count used to **step** the batch loop.

        A static-batch IR's DFL reshape constant is baked to exactly
        ``_model_batch`` — every forward MUST feed that many images or the
        graph crashes mid-reshape ("shape ... conflicts with reshape pattern
        [1,4,16,8400]"). So:

        * **dynamic IR** (``_model_batch is None``): honor the requested count.
        * **static IR**: always step at ``_model_batch``. When the user asked
          for a different ``--batch-size``, warn and sub-loop. The **tail** of
          the image list (fewer than ``_model_batch`` remaining) is handled
          inside :func:`_forward` (zero-pad + slice), so no image is dropped
          and no partial batch reaches the graph.

        A static IR therefore always returns ``_model_batch`` regardless of
        ``requested``: stepping at any other count would under-feed the
        graph (crash at the DFL reshape) or leave the final partial batch
        un-padded (same crash).
        """
        if self._model_batch is None:
            return max(1, requested)
        if requested != self._model_batch:
            # Caller-agnostic wording: ``requested`` comes from --batch-size
            # on the CLI path but from len(frames) on the infer_frames path,
            # so the message must not name a CLI flag the caller can't set.
            # Warn once per engine instance: the server path calls this per
            # request, and the mismatch is a property of the IR, not of the
            # request.
            if not self._batch_mismatch_warned:
                self._batch_mismatch_warned = True
                logger.warning(
                    "Requested per-forward batch %d but the model's static "
                    "input batch is %d; stepping at %d per forward (the tail "
                    "batch is zero-padded inside _forward). Re-export the ONNX "
                    "with a dynamic batch dim for true batched throughput.",
                    requested, self._model_batch, self._model_batch,
                )
        return self._model_batch

    # ------------------------------------------------------------------- infer
    def infer(
        self,
        imgs_input: Union[str, Path, List[Union[str, Path]]],
        conf: float = 0.25,
        iou: float = 0.45,
        max_imgs: int = 32,
        batch_size: int = 8,
        max_det: int = 300,
        save: bool = True,
        output_dir: Union[str, Path] = "results/predictions",
    ) -> List[List[tuple]]:
        """Run end-to-end inference with the OpenVINO backend.

        Failed batches are logged and omitted from the results (the same
        contract as ``YOLOv8Engine.infer``).
        """
        image_paths = load_images(imgs_input, max_images=max_imgs)
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        eff = self._effective_batch(batch_size)
        total_batches = (len(image_paths) + eff - 1) // eff
        all_results: List[List[tuple]] = []
        failed_batches: List[tuple] = []
        for i in range(0, len(image_paths), eff):
            batch_paths = image_paths[i : i + eff]
            try:
                data = preprocess_imgs(
                    batch_paths,
                    imgsz=self.imgsz,
                    device="cpu",  # OpenVINO runs on CPU/iGPU separately
                    original=save,
                )
                batch_dets = self._run_batch(
                    data, conf=conf, iou=iou, max_det=max_det,
                    save=save, output_dir=output_dir,
                )
                all_results.extend(batch_dets)
                logger.info(
                    "Batch %d | Detections: %d",
                    i // eff + 1,
                    sum(len(x) for x in batch_dets),
                )
            except Exception as e:
                failed_batches.append((i // eff + 1, batch_paths, e))
                logger.exception("Batch inference failed (paths=%s)", batch_paths)
        if failed_batches:
            logger.warning(
                "%d/%d batches failed and were skipped from the results: %s",
                len(failed_batches), total_batches,
                [msg for _, _, msg in failed_batches],
            )
        return all_results

    # ----------------------------------------- infer from in-memory BGR frames
    def infer_frames(
        self,
        frames: List[np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        save: bool = False,
        output_dir: Union[str, Path] = "results/predictions",
    ) -> List[List[tuple]]:
        """Inference from already-decoded BGR frames.

        Mirrors ``YOLOv8Engine.infer_frames`` — the call the FastAPI server
        hot path (``docker/server.py``) makes on its engine: a decoded frame
        (e.g. from ``cv2.imdecode`` of an upload) is preprocessed via
        ``preprocess_frames`` and run through the same ``_run_batch`` pipeline
        as the CLI's ``infer``. Skipping the ``imwrite`` -> ``load_images``
        round-trip keeps request latency off disk.

        Frames are sub-batched to the model's per-forward batch
        (:func:`_effective_batch`), so a static-batch IR still serves a
        multi-frame call by looping single forwards.
        """
        # Parity with YOLOv8Engine.infer_frames: an empty frame list is a
        # caller error, not an empty-but-successful result — the server hot
        # path must see the same behavior from either engine.
        if not frames:
            raise ValueError("Frame list cannot be empty")
        output_dir = Path(output_dir)
        if save:
            # No filesystem side effect on the server hot path (save=False);
            # mirrors YOLOv8Engine.infer_frames.
            output_dir.mkdir(parents=True, exist_ok=True)
        eff = self._effective_batch(len(frames))
        all_results: List[List[tuple]] = []
        for i in range(0, len(frames), eff):
            chunk = frames[i : i + eff]
            # Per-chunk isolation mirrors ``infer``: one bad frame (a corrupt
            # decode, a degenerate shape) must not abort the whole server
            # request — log and skip the chunk, keep the rest.
            try:
                data = preprocess_frames(
                    chunk,
                    imgsz=self.imgsz,
                    device="cpu",
                    original=save,
                )
                batch_dets = self._run_batch(
                    data, conf=conf, iou=iou, max_det=max_det,
                    save=save, output_dir=output_dir,
                )
                all_results.extend(batch_dets)
                logger.info(
                    "Frame batch %d | Detections: %d",
                    i // eff + 1,
                    sum(len(x) for x in batch_dets),
                )
            except Exception:
                logger.exception(
                    "Frame batch inference failed (offset=%d, n=%d)", i, len(chunk),
                )
        return all_results


# ---------------------------------------------------------------------------
# Async helper for pipelined video-stream inference (sync engine stays default)
# ---------------------------------------------------------------------------
class OpenVINOAsyncEngine(OpenVINOEngine):
    """Same as ``OpenVINOEngine`` but uses ``AsyncInferQueue`` for pipelining.

    Useful when you have a video stream: while the current frame is being
    post-processed, the next frame is already being inferred on CPU.

    .. note:: Results come back in **completion order, not submission order**
       — ``wait_and_get`` returns ``(frame_id, output)`` tuples precisely so
       the caller can re-order by ``frame_id``. The callback stores a
       **copy** (``np.array``) of each output: the buffers behind
       ``Tensor.data`` are owned by the queue's request pool and are recycled
       by subsequent ``start_async`` rounds, so the callback must not keep a
       view. The shared ``_pending`` list relies on the GIL for append
       safety under OpenVINO's callback thread; don't port this to a
       free-threaded build without a lock.

    The static-batch contract is shared with the sync engine: ``start_batch``
    routes through :func:`_pad_to_static_batch`, so *no partial batch ever
    reaches the graph* on the async path either, and the callback slices the
    output back to the real image count.
    """

    def __init__(self, *args, n_requests: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Single-output contract, enforced at construction: the callback
        # below indexes _output_ports[0] (the same expectation sync _forward
        # guards per call), and an exception inside an AsyncInferQueue
        # callback is not a usable error path — fail init instead.
        if len(self._output_ports) != 1:
            raise RuntimeError(
                f"Expected exactly one OpenVINO output, got "
                f"{len(self._output_ports)}"
            )
        if AsyncInferQueue is None:
            raise ImportError(
                "AsyncInferQueue is not available in this OpenVINO build; "
                "the sync OpenVINOEngine still works — use OpenVINOEngine."
            )
        self.queue = AsyncInferQueue(self.compiled, n_requests)
        # set_callback is the version-stable registration API; start_async's
        # callback argument only exists on newer builds.
        self.queue.set_callback(self._callback)
        self._pending: List = []

    def _callback(self, infer_request, userdata) -> None:
        """Store ``(frame_id, output copy sliced to real_n)``.

        ``userdata`` is the ``(frame_id, real_n)`` pair ``start_batch``
        submitted; ``real_n`` undoes the static-batch zero-padding so the
        async path returns exactly the same per-image outputs as the sync
        ``_forward``. ``np.array(...)`` COPIES out of the request buffer —
        see the class note on request-pool recycling.
        """
        frame_id, real_n = userdata
        output = infer_request.get_tensor(self._output_ports[0]).data
        self._pending.append((frame_id, np.array(output)[:real_n]))

    def start_batch(self, frame_id: int, batch_np: np.ndarray) -> None:
        """Submit one batch for async inference.

        Static-batch IRs are zero-padded via :func:`_pad_to_static_batch`
        before submission (the callback slices the output back), so a
        partial batch never reaches the graph here either.
        """
        padded, real_n = self._pad_to_static_batch(batch_np)
        self.queue.start_async(padded, (frame_id, real_n))

    def wait_and_get(self) -> List[tuple]:
        self.queue.wait_all()
        out = self._pending
        self._pending = []
        return out
