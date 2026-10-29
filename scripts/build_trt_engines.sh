#!/bin/bash
# =============================================================================
# YOLOv8s TensorRT Engine Builder — trtexec-first + Python API fallback
# =============================================================================
# Two build paths (the third, Ultralytics native, is `python main.py tensorrt
# export`):
#   * trtexec   — preferred for local Linux builds (FP16/FP32 only)
#   * Python API fallback — `python main.py tensorrt build` (the programmatic
#     path used on Kaggle/Colab where trtexec is unavailable). This script
#     tries trtexec first, falls back on failure.
#
# INT8 always uses the Python path: trtexec cannot drive the head-exclusion
# pipeline (the ORT QDQ quantization that keeps the /model.22/ detect head
# FP32), so the INT8 engine is built through the Python path, which runs that
# quantization first and hands TRT the QDQ ONNX.
#
# Usage:
#   bash scripts/build_trt_engines.sh            # build fp16 + int8
#   bash scripts/build_trt_engines.sh fp16       # fp16 only
#   bash scripts/build_trt_engines.sh int8       # int8 only
# =============================================================================

# pipefail: the trtexec run is piped through tee for logging, so the pipeline
# exit status is trtexec's; a failed build then reports failure and the
# Python-API fallback triggers.
set -eo pipefail

MODEL="yolov8s"
PT_PATH="models/${MODEL}.pt"
OUTPUT_DIR="models"
LOG_DIR="logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# STATIC=1 builds a static-batch=1 engine (Turing/sm_75 route). The author's
# validated-T4 session hit a tactic-selection failure on the default
# dynamic-batch DFL reshape (observed signature: nbDims > Dims::MAX_DIMS
# around a Shape+Slice node; the cause is inferred; pinned-environment verdict
# _TBD_ — docs/TENSORRT.md); a static-batch ONNX has no symbolic dim, so that
# subgraph does not appear. On Ampere+ (sm_80+) leave STATIC unset for
# dynamic. Each variant caches its own ONNX so a static build never consumes
# a cached dynamic export.
STATIC=${STATIC:-0}
DYN_FLAG=""
SHAPES_FLAG="--minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640"
ONNX_PATH="models/${MODEL}_trt.onnx"   # TRT-friendly export (opset 13, no simplify)
if [ "${STATIC}" = "1" ]; then
    DYN_FLAG="--no-dynamic"
    SHAPES_FLAG="--shapes=images:1x3x640x640"
    ONNX_PATH="models/${MODEL}_trt_static.onnx"
    echo "STATIC=1: static-batch=1 build (Turing route)"
fi

echo "======================================================"
echo "YOLOv8s TensorRT Engine Builder (TensorRT 10.x)"
echo "======================================================"
echo "ONNX Path: ${ONNX_PATH}"
echo ""

# Export a TRT-friendly ONNX (opset 13, no-simplify; static if STATIC=1). The
# canonical opset-17+simplify ONNX carries the subgraph behind the Turing
# build-failure signature (docs/TENSORRT.md).
if [ ! -f "${ONNX_PATH}" ]; then
    echo "Exporting TRT-friendly ONNX (${ONNX_PATH}) from ${PT_PATH}..."
    python main.py export --model "${PT_PATH}" --output "${ONNX_PATH}" \
        --opset 13 --no-simplify ${DYN_FLAG} --device cuda --no-validate || {
        echo "Error: TRT-friendly ONNX export failed (need ${PT_PATH})."
        exit 1
    }
fi

precisions=${1:-"fp16 int8"}

build_with_trtexec() {
    local prec=$1
    local engine_path="${OUTPUT_DIR}/${MODEL}_${prec}.engine"
    local log_file="${LOG_DIR}/trtexec_${prec}.log"

    echo "Building ${prec^^} with trtexec..."
    if [ "${prec}" = "int8" ]; then
        echo "  INT8 via trtexec cannot carry the head-exclusion QDQ pipeline;"
        echo "  routing to the Python path"
        echo "  ('python main.py tensorrt build --precision int8')."
        return 1
    fi
    local cmd="trtexec --onnx=${ONNX_PATH} --saveEngine=${engine_path} --memPoolSize=workspace:8192"
    cmd+=" ${SHAPES_FLAG}"
    if [ "${prec}" = "fp16" ]; then
        cmd+=" --fp16"
    fi

    echo "  Command: ${cmd}" | tee "${log_file}"
    if eval "${cmd}" 2>&1 | tee -a "${log_file}"; then
        echo "${prec^^} engine built with trtexec: ${engine_path}"
        ls -lh "${engine_path}"
        return 0
    else
        echo "trtexec failed for ${prec}, falling back to Python API..."
        return 1
    fi
}

build_with_python() {
    local prec=$1
    local engine_path="${OUTPUT_DIR}/${MODEL}_${prec}.engine"
    local static_flag=""
    [ "${STATIC}" = "1" ] && static_flag="--static"
    echo "Building ${prec^^} with Python API fallback..."
    if python main.py tensorrt build \
        --model "${PT_PATH}" \
        --output "${engine_path}" \
        --precision "${prec}" ${static_flag}; then
        echo "${prec^^} completed with Python fallback"
        return 0
    else
        echo "Python fallback also failed for ${prec}"
        return 1
    fi
}

failed=0
for prec in ${precisions}; do
    echo ""
    echo "=================================================="
    echo "Building ${prec^^} engine..."
    echo "=================================================="
    if build_with_trtexec "${prec}"; then
        echo "${prec^^} SUCCESS (trtexec)"
    elif build_with_python "${prec}"; then
        echo "${prec^^} SUCCESS (Python fallback)"
    else
        echo "Failed to build ${prec^^} engine"
        failed=1
    fi
done

if [ "${failed}" = "1" ]; then
    echo ""
    echo "One or more engine builds failed."
    exit 1
fi

echo ""
echo "======================================================"
echo "Engine files:"
ls -lh "${OUTPUT_DIR}"/*.engine 2>/dev/null || echo "No engines found."
echo ""
echo "Next: python main.py benchmark \\"
echo "  --model tensorrt_fp16:models/yolov8s_fp16.engine \\"
echo "  --model tensorrt_int8:models/yolov8s_int8.engine"
echo "======================================================"
