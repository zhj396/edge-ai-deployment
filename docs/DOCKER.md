# Docker Deployment

Two images ship with this repo:

| Image | Purpose | Size (measured) |
|---|---|---|
| `yolov8s-ort:toolchain` | Runs `main.py` — export, quantize, consistency, benchmark | 2.99 GB |
| `yolov8s-ort:server`    | FastAPI inference server (CPU) | 2.92 GB |

> Image-size provenance: images built 2026-10-16 on this host (WSL2 Ubuntu
> 24.04, Docker 29.1.3); sizes read via `docker images`. Both smoke-tested
> the same day: the toolchain image ran `main.py infer` (onnx_int8, 4
> images, batch 2) and the server image answered `GET /healthz` →
> `{"status":"ok","backend":"onnx_int8","imgsz":640}`, `POST /detect`
> (multipart and base64, `annotated=true` included) and an untokened
> `POST /reload` (403), running as the non-root `app` user (uid 1000).

Both are CPU-only and reuse the project's pinned `requirements-cpu.txt` stack, so
the container runs the exact wheels the CLI is tested against — the same pin
set, the same numpy/onnxruntime ABI. Validated on
Linux (WSL2 Ubuntu 24.04, Docker 29.x) — the toolchain image's in-container wheel set
is exactly the pin set: torch 2.6.0+cpu / onnxruntime 1.26.0 / numpy 2.2.6 /
ultralytics 8.4.114.

> The server image carries the full CPU stack plus a thin FastAPI web layer: its
> import graph transitively needs `ultralytics` (`src/postprocess.py` uses
> `ultralytics.utils.nms` for NMS) and `onnx` (`utils.model_utils`). The
> multi-stage build only strips the **compiler** (`gcc` / `build-essential`)
> from the runtime image.

---

## Prerequisites

Any Docker with daemon access (Docker Desktop handles this itself). On a
stock **Linux server install** the daemon socket is root-only — add your
user to the `docker` group once, then re-login:

```bash
sudo usermod -aG docker $USER
```

> On networks where Docker Hub is unreachable, pull the base image from a
> mirror and retag it: `docker pull <mirror>/library/python:3.12-slim &&
> docker tag <mirror>/library/python:3.12-slim python:3.12-slim`.

---

## Build

```bash
# Toolchain (full CLI in a box)
docker build -f docker/Dockerfile.toolchain -t yolov8s-ort:toolchain .

# Inference server (CPU)
docker build -f docker/Dockerfile.server -t yolov8s-ort:server .
```

`.dockerignore` excludes the whole `models/` and `data/` directories from the
build context, so models and data are never baked in — they are bind-mounted
at runtime. The ResNet-50 weights used by the calibration sampler are likewise
not in the image; the sampler downloads them on first use.

---

## Run the Toolchain Image

Mount your local `models/`, `data/`, and `results/` so artifacts persist across
container runs.

### Export + Quantize

```bash
# Export PT → ONNX FP32
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    yolov8s-ort:toolchain \
    python main.py export \
        --model /app/models/yolov8s.pt \
        --output /app/models/yolov8s_fp32.onnx

# Quantize FP32 → INT8
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    yolov8s-ort:toolchain \
    python main.py quantize \
        --model /app/models/yolov8s_fp32.onnx \
        --output /app/models/yolov8s_int8.onnx \
        --imgs-input /app/data \
        --max-cal-samples 300
```

### Benchmark (writes CSV back to host)

```bash
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    yolov8s-ort:toolchain \
    python main.py benchmark \
        --model onnx_fp32:/app/models/yolov8s_fp32.onnx \
        --model onnx_int8:/app/models/yolov8s_int8.onnx \
        --imgs-input /app/data \
        --max-images 32 \
        --warmup 10 \
        --runs 25
```

The resulting `results/benchmark_summary_<TS>.csv` shows up on your host because
of the `-v` bind. Each CLI run is timestamped; take the newest summary with
`ls -t results/benchmark_summary_*.csv | head -1`.

### Via docker compose

`docker compose run toolchain <args>` replaces the default `--help` command, so:

```bash
docker compose run toolchain python main.py benchmark \
    --model onnx_fp32:models/yolov8s_fp32.onnx --imgs-input data --max-images 16
```

---

## Run the Inference Server

### Quick start

```bash
docker run --rm -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    -e MODEL_PATH=/app/models/yolov8s_int8.onnx \
    -e BACKEND=onnx_int8 \
    yolov8s-ort:server
```

Tunable env vars:

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/app/models/yolov8s_fp32.onnx` | Absolute path inside container |
| `BACKEND` | `onnx_fp32` | `onnx_fp32` / `onnx_int8` |
| `IMGSZ` | `640` | Network input size |
| `CONF_THRESHOLD` | `0.25` | Default confidence threshold |
| `IOU_THRESHOLD` | `0.45` | Default NMS IoU threshold |
| `MAX_DET` | `300` | Max detections per image |
| `MAX_UPLOAD_MB` | `10` | Reject larger images with 413 (`0` disables). Also caps the whole request body (2x this value + 64 KB, covering base64 inflation) via `Content-Length` before it is buffered |
| `MAX_IMAGE_PIXELS` | `16777216` (4096×4096) | Decompression-bomb guard: reject images whose *decoded* pixel count exceeds this (413, `0` disables). `MAX_UPLOAD_MB` caps compressed bytes only — a <1 MB PNG can decode to gigabytes of pixels, so dimensions are sniffed from the image header before decoding, and a second check on the decoded array covers formats whose header PIL cannot parse (cv2's own cap is 2³⁰ px, far above this default) |
| `LOG_LEVEL` | `20` (INFO) | Python logging level |
| `LOG_FILE` | `/app/logs/server.log` (set in image) | Log path; a repo-root dev run defaults to `logs/server.log` |
| `RELOAD_TOKEN` | *(empty → disabled)* | Set to enable authenticated `/reload` |

Invalid numeric values in these vars fall back to the documented default;
a warning is printed at boot.

### In-memory hot path

`POST /detect` feeds the decoded frame straight into `YOLOv8Engine.infer_frames`,
which preprocesses arrays (via `preprocess_frames`) and reuses the same
`_run_batch` the CLI's `infer` uses. No tempfile `imwrite` → `imread` round-trip
per request — important for p99 latency on a CPU-bound server.

### Smoke-test the server

```bash
# Health
curl -s http://localhost:8000/healthz
# {"status":"ok","backend":"onnx_int8","imgsz":640}

# Metrics
curl -s http://localhost:8000/metrics

# Detect via file upload — conf/iou/annotated are QUERY params (they are
# FastAPI scalar params, not form fields; -F silently ignores them)
curl -s -X POST "http://localhost:8000/detect?conf=0.3&annotated=true" \
    -F "file=@./data/sample.jpg" \
    -o result.json
jq '.num_detections, .inference_ms, .detections[:3]' result.json

# Detect via base64 JSON
curl -s -X POST http://localhost:8000/detect_json \
    -H "Content-Type: application/json" \
    -d "{\"image_b64\":\"$(base64 -w0 ./data/sample.jpg)\"}"
```

### Hot-reload the model (gated)

`/reload` is **disabled by default** (`RELOAD_TOKEN` empty → 403). To enable an
authenticated hot-swap after an OTA model update:

```bash
docker run --rm -p 8000:8000 \
    -e MODEL_PATH=/app/models/yolov8s_int8.onnx \
    -e RELOAD_TOKEN="$(openssl rand -hex 24)" \
    -v $(pwd)/models:/app/models:ro \
    yolov8s-ort:server

# then, with the same token (model_path/backend are QUERY params — a -d body
# is ignored and the default MODEL_PATH is silently reloaded):
curl -X POST \
    "http://localhost:8000/reload?model_path=/app/models/yolov8s_int8_v2.onnx&backend=onnx_int8" \
    -H "X-Reload-Token: $RELOAD_TOKEN"
```

Pair with network policy / mTLS for real hardening — the token is the only
authentication on the remote model-swap endpoint.

---

## Resource limits

Production deployments should cap memory and CPU:

```bash
docker run --rm -p 8000:8000 \
    --memory=2g --memory-swap=2g \
    --cpus=4.0 \
    ...
```

ORT thread counts are tuned for single-stream YOLOv8s in `YOLOv8Engine`:
intra-op = `min(4, ncpus)`, inter-op = 2 when intra-op > 1 (host-dependent —
verify on your machine before citing numbers); set `--cpus` to match the host
you benchmarked on.

---

## Why multi-stage for the server image?

The `Dockerfile.server` uses two stages:

1. **builder** — installs `gcc` and `build-essential`, installs the full CPU
   stack + web layer.
2. **runtime** — only carries the compiled wheels; **no compiler, no build
   headers**. Runs as non-root `app` (uid 1000) under `tini` for clean SIGTERM.

The split shrinks the runtime image by the compiler/toolchain weight and removes
`apt-get install gcc` (no trivial privilege escalation via compiler). The
*dependency* set is identical between stages: the runtime carries the tested
CPU stack, including `ultralytics` (`src/postprocess.py` imports
`ultralytics.utils.nms` for NMS) and `onnx` (`utils.model_utils`), which
`from src import YOLOv8Engine` requires at startup.

---

## Troubleshooting

### Server returns 503 on `/healthz`
The engine didn't load — usually because `MODEL_PATH` is wrong inside the
container. Check `docker logs <container-id>` for the traceback, then re-mount
your models directory.

### `cv2` import fails with "libGL.so.1: cannot open"
The full `opencv-python` wheel links against the system `libGL`, so this
error means the importing environment lacks that library. The provided
image installs `opencv-python` (via `requirements-cpu.txt`) plus the
`libgl1` / `libglib2.0-0` runtime libs in the Dockerfile — keep both.
If the error appears inside the container, `libgl1` was dropped from the
build; in a local venv, install the distro's `libgl1` system package.

### `torch` install is huge
`requirements-cpu.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu`
to pull the CPU-only torch wheel (~200 MB compressed); the default
PyPI wheel bundles CUDA (~2.5 GB). Don't remove this line.

### Permission errors on `/app/logs`
The runtime image runs as the non-root `app` user (uid 1000) and creates
`/app/logs` at build time. If you bind-mount `logs/` from the host, ensure the
host directory is writable by uid 1000 (`chown -R 1000:1000 logs/`).

### Want GPU support?
The current images are CPU-only, matching this repo's validated deployment
scope (the GPU work in this project runs on Kaggle T4 via the
`requirements-kaggle.txt` overlay, not Docker). A GPU-serving image would need
an NVIDIA Container Toolkit base image plus an `onnxruntime-gpu` overlay — and
the CUDA EP must go through the IO-Binding/DLPack path described in
CLAUDE.md invariant #4, not a plain `session.run` on host numpy.
