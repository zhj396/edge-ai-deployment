#!/bin/bash
# =============================================================================
# YOLOv8s TensorRT 10.4.0 Engine Builder — trtexec-first + Python API fallback
# =============================================================================
# Two build paths (the third, Ultralytics native, is `python main.py tensorrt
# export`):
#   * trtexec   — preferred for local Linux (more deterministic layer fusion)
#   * Python API fallback — `python main.py tensorrt build` (the programmatic
#     path used on Kaggle/Colab where trtexec is unavailable; the interview
#     story). This script tries trtexec first, falls back on failure.
#
# Usage:
#   bash scripts/build_trt_engines.sh            # build fp16 + int8
#   bash scripts/build_trt_engines.sh fp16       # fp16 only
#   bash scripts/build_trt_engines.sh int8       # int8 only
# =============================================================================

set -e

MODEL="yolov8s"
PT_PATH="models/${MODEL}.pt"
ONNX_PATH="models/${MODEL}_trt.onnx"   # TRT-friendly export (opset 13, no simplify)
OUTPUT_DIR="models"
LOG_DIR="logs"
CALIB_CACHE="models/calibration.cache"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "======================================================"
echo "YOLOv8s TensorRT Engine Builder (TRT 10.4.0)"
echo "======================================================"
echo "ONNX Path: ${ONNX_PATH}"
echo ""

# STATIC=1 builds a static-batch=1 engine (Turing/sm_75 route). The default
# dynamic-batch DFL reshape forces a Shape+Slice subgraph TRT 10.4 cannot lower
# on Turing (nbDims > Dims::MAX_DIMS); a static-batch ONNX has no symbolic dim
# so no Shape subgraph. On Ampere+ (sm_80+) leave STATIC unset for dynamic.
STATIC=${STATIC:-0}
DYN_FLAG=""
SHAPES_FLAG="--minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640"
if [ "${STATIC}" = "1" ]; then
    DYN_FLAG="--no-dynamic"
    SHAPES_FLAG="--shapes=images:1x3x640x640"
    echo "STATIC=1: static-batch=1 build (Turing route)"
fi

# Export a TRT-friendly ONNX (opset 13, no-simplify; static if STATIC=1). The
# canonical opset-17+simplify ONNX trips a TRT 10.4 build error on Turing.
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
    local cmd="trtexec --onnx=${ONNX_PATH} --saveEngine=${engine_path} --workspace=8192"
    cmd+=" ${SHAPES_FLAG}"
    if [ "${prec}" = "fp16" ]; then
        cmd+=" --fp16"
    elif [ "${prec}" = "int8" ]; then
        cmd+=" --int8"
        if [ -f "${CALIB_CACHE}" ]; then
            cmd+=" --calib=${CALIB_CACHE}"
            echo "  Using calibration cache: ${CALIB_CACHE}"
        else
            echo "  Warning: no calibration cache; INT8 via trtexec needs one."
            echo "  Run 'python main.py tensorrt build --precision int8' instead"
            echo "  (the Python path calibrates from the shared CalibrationSampler)."
        fi
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
    fi
done

echo ""
echo "======================================================"
echo "Engine files:"
ls -lh "${OUTPUT_DIR}"/*.engine 2>/dev/null || echo "No engines found."
echo ""
echo "Next: python main.py benchmark \\"
echo "  --model tensorrt_fp16:models/yolov8s_fp16.engine \\"
echo "  --model tensorrt_int8:models/yolov8s_int8.engine"
echo "======================================================"
