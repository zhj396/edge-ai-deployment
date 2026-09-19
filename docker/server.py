"""FastAPI inference server for YOLOv8s ONNX Runtime.

Endpoints
---------
* ``GET  /healthz``   — liveness probe (``{"status": "ok"}`` or 503 while loading).
* ``GET  /metrics``   — in-process counters (requests, latency).
* ``POST /detect``    — multipart ``file`` upload OR JSON ``{"image_b64": "..."}``.
                        Optional ``conf``, ``iou``, ``annotated`` query params.
* ``POST /detect_json`` — JSON-body variant of ``/detect``.
* ``POST /reload``    — hot-swap the model behind the lock. **Gated by
                        ``RELOAD_TOKEN``: disabled unless the token is set, and
                        then requires matching ``X-Reload-Token`` header.**

The server is **stateless** (one engine loaded at startup), serializes
inference with a lock (ORT sessions are not reentrant), and graceful on
SIGTERM (uvicorn handles shutdown after the current request finishes via
``tini``).
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Project imports — server.py lives in docker/ but the package layout
# keeps src/, utils/, cli/ at the repo root. PYTHONPATH=/app (set in the
# image) covers the container; the sys.path tweak below covers `uvicorn
# docker.server:app` run from the repo root in dev.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import YOLOv8Engine  # noqa: E402
from utils import get_logger, setup_logging, suppress_third_party_logs  # noqa: E402
from utils.visualization import draw_detections  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration (env vars so ops can tune without rebuilding the image)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read an integer env var; fall back to ``default`` with a warning on a
    bad value instead of crashing the process at import time."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: invalid {name}={raw!r}, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: invalid {name}={raw!r}, using default {default}")
        return default


MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/yolov8s_fp32.onnx")
BACKEND = os.environ.get("BACKEND", "onnx_fp32")
IMGSZ = _env_int("IMGSZ", 640)
DEFAULT_CONF = _env_float("CONF_THRESHOLD", 0.25)
DEFAULT_IOU = _env_float("IOU_THRESHOLD", 0.45)
MAX_DET = _env_int("MAX_DET", 300)
# Reject request bodies whose decoded image exceeds this size (413). Uploads
# are read fully into memory before decoding, so the cap bounds worst-case
# request memory. Set 0 to disable.
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 10)
# If unset, /reload is disabled (403). This is a blunt but effective gate on an
# otherwise unauthenticated remote model-swap — pair it with a network policy
# in production. Token compared in constant-ish time (hmac.compare_digest).
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN")

# Log file via env so the same module works in-container (LOG_FILE=/app/logs/
# server.log, set in the image) and in a repo-root dev run (relative logs/
# inside the worktree, which setup_logging creates).
setup_logging(log_file=os.environ.get("LOG_FILE", "logs/server.log"),
              level=_env_int("LOG_LEVEL", 20))
suppress_third_party_logs()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class DetectResponse(BaseModel):
    backend: str
    imgsz: int
    num_detections: int
    inference_ms: float
    detections: List[Detection]
    image_b64: Optional[str] = Field(
        default=None, description="Annotated image (PNG), only if annotated=true"
    )


class JSONDetectRequest(BaseModel):
    image_b64: str = Field(..., description="base64-encoded image bytes")


# ---------------------------------------------------------------------------
# App state (single engine, mutex around inference + model swap)
# ---------------------------------------------------------------------------
_engine_lock = Lock()
_engine: Optional[YOLOv8Engine] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the engine at startup; a load failure leaves ``_engine=None`` so
    ``/healthz`` reports 503 instead of crashing the container into a restart
    loop (ops can then inspect logs / fix the mount and POST /reload)."""
    global _engine
    try:
        _engine = _load_engine(MODEL_PATH, BACKEND)
    except Exception as e:
        logger.exception("Engine load failed at startup: %s", e)
        _engine = None
    yield
    # Shutdown: drop the engine so the ORT session releases deterministically.
    with _engine_lock:
        _engine = None


app = FastAPI(
    title="YOLOv8s ONNX Runtime Inference Server",
    version="1.2.0",
    description="CPU/GPU inference server for a 12-class YOLOv8s model.",
    lifespan=_lifespan,
)

# Lightweight in-process metrics — production should scrape Prometheus. Updated
# under _engine_lock so concurrent threadpool workers can't lose increments.
_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "inference_ms_total": 0.0,
    "inference_ms_max": 0.0,
}


def _load_engine(model_path: str, backend: str) -> YOLOv8Engine:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    logger.info("Loading engine: backend=%s path=%s", backend, p)
    return YOLOv8Engine(model_path=str(p), backend=backend, imgsz=IMGSZ, device="cpu")


# ---------------------------------------------------------------------------
# Image decode helpers
# ---------------------------------------------------------------------------
def _reject_oversize(n_bytes: int) -> None:
    """413 when the decoded image exceeds MAX_UPLOAD_MB (0 disables the cap)."""
    if MAX_UPLOAD_MB and n_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds MAX_UPLOAD_MB={MAX_UPLOAD_MB}",
        )


def _decode_image_b64(b64_str: str) -> np.ndarray:
    # b64 inflates by ~4/3 — bound the decoded size before allocating it.
    _reject_oversize(len(b64_str) * 3 // 4)
    try:
        raw = base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}") from e
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


async def _read_upload(file: UploadFile) -> np.ndarray:
    # Read at most cap+1 bytes so an oversized body fails fast instead of
    # buffering an arbitrary upload in memory.
    cap_bytes = MAX_UPLOAD_MB * 1024 * 1024 + 1 if MAX_UPLOAD_MB else None
    raw = await file.read(cap_bytes)
    _reject_oversize(len(raw))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to PIL for formats cv2 doesn't handle well
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e
    return img


def _encode_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Inference helper (whole body under the lock — engine isn't reentrant)
# ---------------------------------------------------------------------------
def _infer_single(
    img: np.ndarray,
    conf: float,
    iou: float,
    annotated: bool,
) -> DetectResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    with _engine_lock:
        # Bind the engine reference inside the lock so a concurrent /reload
        # swap can't make response metadata (backend) describe a different
        # engine than the one that produced the detections.
        engine = _engine
        t0 = time.perf_counter()
        # In-memory path: feed the decoded frame straight into the engine via
        # infer_frames, which preprocesses arrays (no tempfile round-trip) and
        # reuses the same _run_batch the CLI's `infer` uses.
        results = engine.infer_frames(
            [img], conf=conf, iou=iou, max_det=MAX_DET, save=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        raw_dets = results[0] if results else []

        out_dets: List[Detection] = []
        for x1, y1, x2, y2, c, cls in raw_dets:
            out_dets.append(
                Detection(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=float(c),
                    class_id=int(cls),
                    class_name=str(_engine.class_names.get(int(cls), int(cls))),
                )
            )

        image_b64 = None
        if annotated and raw_dets:
            drawn = draw_detections(
                img.copy(), raw_dets, class_names=engine.class_names
            )
            image_b64 = _encode_png_b64(drawn)

        # metrics updated under the lock — non-atomic += otherwise loses
        # increments under concurrent threadpool workers.
        _metrics["requests_total"] += 1
        _metrics["inference_ms_total"] += elapsed_ms
        _metrics["inference_ms_max"] = max(_metrics["inference_ms_max"], elapsed_ms)

    return DetectResponse(
        backend=engine.backend,
        imgsz=IMGSZ,
        num_detections=len(out_dets),
        inference_ms=round(elapsed_ms, 2),
        detections=out_dets,
        image_b64=image_b64,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> Dict:
    if _engine is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok", "backend": _engine.backend, "imgsz": IMGSZ}


@app.get("/metrics")
def metrics() -> Dict:
    n = max(_metrics["requests_total"], 1)
    return {
        "requests_total": _metrics["requests_total"],
        "errors_total": _metrics["errors_total"],
        "inference_ms_mean": round(_metrics["inference_ms_total"] / n, 2),
        "inference_ms_max": round(_metrics["inference_ms_max"], 2),
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: Optional[UploadFile] = File(default=None),
    image_b64: Optional[str] = Form(default=None),
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    annotated: bool = False,
):
    """Run detection.

    Send the image as multipart ``file=@image.jpg`` OR as form field
    ``image_b64=...``. Optional query params: ``conf``, ``iou``,
    ``annotated`` (return base64 PNG of detections).
    """
    try:
        if file is not None:
            img = await _read_upload(file)
        elif image_b64:
            img = _decode_image_b64(image_b64)
        else:
            raise HTTPException(status_code=400, detail="Provide file or image_b64")
        return _infer_single(img, conf=conf, iou=iou, annotated=annotated)
    except HTTPException as e:
        # 4xx are client errors (bad/oversize upload) — only server-side
        # failures (5xx) count as server errors in /metrics.
        if e.status_code >= 500:
            with _engine_lock:
                _metrics["errors_total"] += 1
        raise
    except Exception as e:
        with _engine_lock:
            _metrics["errors_total"] += 1
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/detect_json", response_model=DetectResponse)
async def detect_json(req: JSONDetectRequest, conf: float = DEFAULT_CONF,
                      iou: float = DEFAULT_IOU, annotated: bool = False):
    """JSON body variant of /detect — accepts ``{"image_b64": "..."}``."""
    try:
        img = _decode_image_b64(req.image_b64)
        return _infer_single(img, conf=conf, iou=iou, annotated=annotated)
    except HTTPException as e:
        if e.status_code >= 500:
            with _engine_lock:
                _metrics["errors_total"] += 1
        raise
    except Exception as e:
        with _engine_lock:
            _metrics["errors_total"] += 1
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reload")
def reload_model(model_path: Optional[str] = None,
                 backend: Optional[str] = None,
                 x_reload_token: str = Header(default="")) -> Dict:
    """Hot-reload the model — useful after an OTA model update.

    Disabled unless ``RELOAD_TOKEN`` is set in the environment; when set, the
    request must carry a matching ``X-Reload-Token`` header. This stops anyone
    who can reach the port from pointing the engine at an arbitrary container
    path. Pair with network policy / mTLS for real hardening.
    """
    import hmac
    if not RELOAD_TOKEN or not hmac.compare_digest(x_reload_token, RELOAD_TOKEN):
        raise HTTPException(status_code=403, detail="reload disabled or bad token")
    global _engine
    target_path = model_path or MODEL_PATH
    target_backend = backend or BACKEND
    try:
        new_engine = _load_engine(target_path, target_backend)
        with _engine_lock:
            _engine = new_engine
        return {"status": "ok", "backend": target_backend, "model": target_path}
    except Exception as e:
        logger.exception("Reload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
