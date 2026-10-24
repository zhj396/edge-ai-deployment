#!/usr/bin/env bash
#
# trt_cpp_setup.sh — set up the C++ build deps for the trt_cpp backend:
# OpenCV (C++ dev) + the TensorRT 10.16.1 SDK.
#
# Scope (DECOUPLED by design — see docs/TENSORRT.md §trt_cpp):
#   This script handles the NON-PIP C++ build deps: the OpenCV C++ dev package
#   (apt — the pip opencv-python wheel doesn't ship OpenCVConfig.cmake/headers)
#   and the TensorRT C++ SDK (download + extract the tarball + export env). It
#   does NOT touch Python deps. The `tensorrt` wheel + `cuda-python` (needed to
#   BUILD the .engine via `python main.py tensorrt build`) come from
#   `pip install -r requirements-tensorrt.txt` — the documented, PyPI-reproducible
#   path. This separation eliminates the overlap/conflict between the tarball
#   wheel and the requirements pin: pip owns Python, this script owns the C++ deps.
#
#   The script VERIFIES (does not install) that the installed `tensorrt` wheel
#   matches the tarball version, so the .engine built by Python cross-loads in
#   the C++ module (TRT engines are version-locked across major versions).
#
# Why a tarball, not apt: apt's default TensorRT is now 11.x (CUDA 13), but the
# project pins TRT 10.16.1 / CUDA 12.9 (matching the Kaggle CUDA-12.x runtime + the
# tensorrt==10.16.1.11 wheel). An 11.x system install won't deserialize a 10.x-built
# engine. The tarball gives the exact pinned version + the C++ headers/dev symlink
# (libnvinfer.so) that the pip wheel omits.
#
# MUST be sourced, not executed, so the env exports persist into your shell:
#   source scripts/trt_cpp_setup.sh
# (A `bash scripts/trt_cpp_setup.sh` run sets the exports in a subshell that dies
#  on exit — cmake then can't find TensorRT. The script warns if run this way.)
#
# Portable: defaults to /kaggle/working on Kaggle (the documented T4 target),
# else $HOME. Override with TENSORRT_INSTALL_DIR=/your/path.
#
# DISK LAYOUT (Kaggle): /kaggle/working has a ~20G quota. The TRT tarball
# (~8G) plus its extracted tree (~12G) cannot coexist there, so the script
# stages the tarball in a scratch dir on the container overlay (/tmp, ~1T
# free), and extracts ONLY the C++ build subset (include/ + lib/, ~2-3G) into
# the install dir — bin/, data/, and the Python wheels inside the tarball are
# not needed by the _trt_cpp build (pip owns the Python side). The tarball is
# deleted before extraction, so the scratch dir never holds both copies.
# (Sizes observed on the Kaggle T4 image; treat them as guidance,
# not guarantees — NVIDIA tarball contents vary by release.)

# Sourcing runs this in the caller's shell, so shell options are SAVED here
# and RESTORED at the end — a bare `set -Eeuo pipefail` would leak errexit /
# pipefail / nounset into the user's interactive shell (a grep that finds
# nothing would later kill one of their pipelines).
_TRT_SETUP_OPTS=$(set +o)
_TRT_SETUP_PIPEFAIL=$(set -o | awk '$1=="pipefail"{print $2}')
_TRT_SETUP_NOUNSET=$(set -o | awk '$1=="nounset"{print $2}')
set -Eeuo pipefail

_trt_cpp_restore_opts() {
    case "${_TRT_SETUP_PIPEFAIL}" in on) set -o pipefail ;; off) set +o pipefail ;; esac
    case "${_TRT_SETUP_NOUNSET}" in on) set -o nounset ;; off) set +o nounset ;; esac
    eval "${_TRT_SETUP_OPTS}"
}

TENSORRT_VERSION="10.16.1.11"
CUDA_VERSION="12.9"

# --- Install dir: Kaggle-aware default, overridable ------------------------
# /kaggle/working is writable + persistent on Kaggle notebooks; elsewhere use
# $HOME. Override: TENSORRT_INSTALL_DIR=/opt source scripts/trt_cpp_setup.sh
if [[ -z "${TENSORRT_INSTALL_DIR:-}" ]]; then
    if [[ -d /kaggle/working ]]; then
        TENSORRT_INSTALL_DIR="/kaggle/working"
    else
        TENSORRT_INSTALL_DIR="${HOME}"
    fi
fi

# --- Scratch dir for the tarball: container overlay, NOT the quota'd dir ----
# On Kaggle the overlay (/tmp) has ~1T free vs /kaggle/working's ~20G quota;
# the tarball + extraction never fit together in the quota'd dir. Elsewhere
# the tarball just lives next to the install dir.
if [[ -d /kaggle/working && -d /tmp ]]; then
    TARBALL_DIR="/tmp"
else
    TARBALL_DIR="${TENSORRT_INSTALL_DIR}"
fi

TENSORRT_ROOT="${TENSORRT_INSTALL_DIR}/TensorRT-${TENSORRT_VERSION}"
TARBALL="${TARBALL_DIR}/TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz"
URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.1/tars/$(basename "${TARBALL}")"

echo "========================================"
echo "trt_cpp TensorRT ${TENSORRT_VERSION} (cuda-${CUDA_VERSION}) setup"
echo "========================================"
echo "Install dir : ${TENSORRT_INSTALL_DIR}"
echo "TensorRT root: ${TENSORRT_ROOT}"

# --- Sourced-vs-executed check ---------------------------------------------
# The env exports at the end only persist if this script is SOURCED. Detect a
# `bash script.sh` run and warn (the download/extract still succeed, but cmake
# in the parent shell won't see TENSORRT_ROOT).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo
    echo "WARNING: this script is being EXECUTED, not SOURCED. The TENSORRT_ROOT /"
    echo "         LD_LIBRARY_PATH exports below will die with this subshell, and"
    echo "         cmake won't find TensorRT. Re-run with:"
    echo "             source scripts/trt_cpp_setup.sh"
    echo "         (The download + extract still complete; only the env is lost.)"
fi

echo
echo "[0] OpenCV C++ dev — required by cpp/common (ort_core_common)"
# The cpp/ tree links OpenCV's C++ headers + libs (OpenCVConfig.cmake). The pip
# opencv-python wheel does NOT ship these — you need the system dev package.
# Check first; auto-install on Debian/Ubuntu (the Kaggle target) if missing.
# Skip the auto-install with SKIP_OPENCV_INSTALL=1 if you manage OpenCV yourself.
# NOTE: this script is SOURCED, so a fatal error uses `return` (not `exit`,
# which would kill the user's interactive shell).
_opencv_found() { find /usr /opt -name OpenCVConfig.cmake 2>/dev/null | grep -q .; }
if _opencv_found; then
    echo "  OpenCV C++ config found:"
    find /usr /opt -name OpenCVConfig.cmake 2>/dev/null | head -3 | sed 's/^/    /'
else
    echo "  OpenCV C++ dev NOT found (pip's opencv-python doesn't ship OpenCVConfig.cmake)."
    if [[ "${SKIP_OPENCV_INSTALL:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
        echo "  Installing libopencv-dev via apt (Debian/Ubuntu)..."
        _sudo=""; [[ "$(id -u)" -ne 0 ]] && _sudo="sudo"
        if ! ${_sudo} apt-get update -qq || ! ${_sudo} apt-get install -y -qq libopencv-dev; then
            echo "  ERROR: apt install of libopencv-dev failed."
            echo "  Install OpenCV C++ dev manually (>= 4.5, components core imgproc imgcodecs dnn):"
            echo "    Debian/Ubuntu: sudo apt-get install -y libopencv-dev"
            echo "  Or point CMake at it: -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4"
            return 1 2>/dev/null || exit 1
        fi
        echo "  Installed. OpenCVConfig.cmake:"
        find /usr -name OpenCVConfig.cmake 2>/dev/null | head -3 | sed 's/^/    /'
    else
        echo "  ERROR: OpenCV C++ dev missing and auto-install unavailable (no apt-get, or SKIP_OPENCV_INSTALL=1)."
        echo "  Install it manually (>= 4.5, components core imgproc imgcodecs dnn):"
        echo "    Debian/Ubuntu: sudo apt-get install -y libopencv-dev"
        echo "  Or point CMake at it: -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4"
        return 1 2>/dev/null || exit 1
    fi
fi

echo
echo "[1] GPU"
nvidia-smi || echo "  (nvidia-smi unavailable — fine for download/extract; needed at build/run time)"

echo
echo "[2] CUDA"
nvcc --version || echo "  (nvcc not on PATH — the cmake build finds CUDA via find_package(CUDAToolkit))"

echo
echo "[3] Download TensorRT tarball"
if [[ -f "${TARBALL}" ]]; then
    echo "Already exists: ${TARBALL}"
else
    # NVIDIA's developer download may require auth/cookies on some networks
    # (it works unauthenticated on Kaggle). On wget failure the script
    # prints manual download steps so the error stays actionable.
    if ! wget --progress=bar:force "${URL}" -O "${TARBALL}"; then
        rm -f "${TARBALL}"  # don't leave a truncated/HTML-error file behind
        echo
        echo "ERROR: automatic download failed (NVIDIA may require login on this network)."
        echo "Manual fallback:"
        echo "  1. Go to https://developer.nvidia.com/tensorrt/download/10x"
        echo "  2. Download TensorRT ${TENSORRT_VERSION} GA for Linux x86_64, CUDA ${CUDA_VERSION}:"
        echo "       $(basename "${TARBALL}")"
        echo "  3. Place it at: ${TARBALL}"
        echo "  4. Re-run: source scripts/trt_cpp_setup.sh"
        return 1 2>/dev/null || exit 1
    fi
fi

echo
echo "[4] Verify archive"
ls -lh "${TARBALL}"
# Don't pipe tar to head (pipefail + SIGPIPE); list to a temp file instead.
tar -tzf "${TARBALL}" > /tmp/trt_cpp_files.txt
echo "Archive OK. First files:"
sed -n '1,10p' /tmp/trt_cpp_files.txt

echo
echo "[5] Extract (C++ build subset only: include/ + lib/)"
# Extract include/ + lib/ directly from the tarball into TENSORRT_ROOT — the
# _trt_cpp build needs headers + shared objects only (pip owns the Python
# wheels inside the tarball). Extracting the subset keeps the install dir at
# ~2-3G, which fits the /kaggle/working quota alongside the repo. On Kaggle
# the tarball itself was staged in /tmp (overlay), so it never competes with
# the quota'd dir.
if [[ -d "${TENSORRT_ROOT}/include" && -d "${TENSORRT_ROOT}/lib" ]]; then
    echo "Already extracted: ${TENSORRT_ROOT}"
else
    mkdir -p "${TENSORRT_ROOT}"
    tar -xzf "${TARBALL}" -C "${TENSORRT_ROOT}" \
        --wildcards "TensorRT-${TENSORRT_VERSION}/include/*" "TensorRT-${TENSORRT_VERSION}/lib/*" \
        --strip-components=1
fi
# The tarball has served its purpose once the subset is extracted — remove it
# so the scratch dir (/tmp on Kaggle, shared with the overlay quota) doesn't
# keep an 8G file for the rest of the session. Verify before deleting.
test -f "${TENSORRT_ROOT}/include/NvInfer.h"
test -f "${TENSORRT_ROOT}/lib/libnvinfer.so.10" || test -f "${TENSORRT_ROOT}/lib/libnvinfer.so"
rm -f "${TARBALL}"
echo "Tarball removed after subset extraction: ${TARBALL}"

echo
echo "[6] Verify SDK layout"
test -d "${TENSORRT_ROOT}/lib"
test -d "${TENSORRT_ROOT}/include"
test -f "${TENSORRT_ROOT}/include/NvInfer.h"
test -f "${TENSORRT_ROOT}/include/NvInferRuntime.h"
# The dev symlink libnvinfer.so (not just libnvinfer.so.10) is what CMake's
# find_library needs to link the _trt_cpp module — find_library matches
# lib<nvinfer>.so, NOT the versioned libnvinfer.so.10. NVIDIA dev tarballs
# usually ship the symlink; some ship only the versioned one, so create the
# symlink if absent.
if [[ ! -e "${TENSORRT_ROOT}/lib/libnvinfer.so" ]]; then
    _nvinfer_target="$(ls -1 "${TENSORRT_ROOT}"/lib/libnvinfer.so.* 2>/dev/null | head -1 || true)"
    if [[ -n "${_nvinfer_target}" ]]; then
        ln -sf "$(basename "${_nvinfer_target}")" "${TENSORRT_ROOT}/lib/libnvinfer.so"
        echo "  Created dev symlink: libnvinfer.so -> $(basename "${_nvinfer_target}")"
    else
        echo "  WARNING: no libnvinfer.so* in ${TENSORRT_ROOT}/lib — CMake find_library will fail."
        echo "           The tarball may be incomplete; re-download it."
    fi
fi
echo "SDK layout OK: ${TENSORRT_ROOT}"
find "${TENSORRT_ROOT}/lib" -maxdepth 1 -name 'libnvinfer.so*' -print

echo
echo "[7] Environment (these exports persist ONLY if the script was SOURCED)"
export TENSORRT_ROOT
export PATH="${TENSORRT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${TENSORRT_ROOT}/include:${CPATH:-}"
export LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LIBRARY_PATH:-}"
echo "TENSORRT_ROOT   = ${TENSORRT_ROOT}"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"

echo
echo "[8] trtexec (sanity — the SDK's own CLI)"
if [[ -x "${TENSORRT_ROOT}/bin/trtexec" ]]; then
    "${TENSORRT_ROOT}/bin/trtexec" --version 2>/dev/null || echo "  (trtexec --version unavailable; non-fatal)"
else
    echo "  (bin/ was not extracted — the subset install carries include/ + lib/ only;"
    echo "   trtexec is not needed by the _trt_cpp build. Non-fatal.)"
fi

echo
echo "[9] VERIFY Python wheel matches the tarball (no install — decoupled)"
# The .engine is built by the Python `tensorrt` wheel; the C++ module links the
# tarball libs. They MUST be the same TRT version or deserializeCudaEngine fails.
# This script does NOT install the wheel (pip owns Python deps via
# requirements-tensorrt.txt) — it only verifies the match.
if python3 - "${TENSORRT_VERSION}" <<'PY' 2>/dev/null
import sys
import tensorrt as trt
v = trt.__version__
pin = sys.argv[1]
assert v.startswith(".".join(pin.split(".")[:3])), f"tensorrt wheel {v} != tarball {pin}"
print(f"  tensorrt wheel {v} == tarball {pin}  OK")
PY
then
    :
else
    echo "  WARNING: the installed 'tensorrt' wheel is absent or != 10.16.1.x."
    echo "           The .engine you build won't cross-load in the C++ module."
    echo "           Fix: pip install --extra-index-url https://pypi.nvidia.com \\"
    echo "                -r requirements-tensorrt.txt"
    echo "           (installs tensorrt==10.16.1.11 + cuda-python>=12.3.0)"
fi

echo
echo "========================================"
echo "TensorRT ${TENSORRT_VERSION} C++ SDK ready."

# Restore the caller's shell options (this script is SOURCED — a leaked
# errexit/pipefail/nounset would change the interactive shell's behavior).
_trt_cpp_restore_opts
unset -f _trt_cpp_restore_opts
echo
echo
echo "Next steps:"
echo "  1. Build the engine (Python, needs GPU + cuda-python):"
echo "       python main.py tensorrt build --model models/yolov8s.pt --precision fp16 --static"
echo "  2. Build the C++ pybind11 module (pass -DTENSORRT_ROOT explicitly — on"
echo "     Kaggle/Jupyter each ! cell is a fresh subshell, so the exported env"
echo "     var above does NOT persist to the next cell; the -D flag does):"
echo "       cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON -DTENSORRT_ROOT=${TENSORRT_ROOT}"
echo "       cmake --build cpp/build --target _trt_cpp"
echo "     (prints 'TensorRT found: 10.16.1.11'; if a prior configure failed, clear"
echo "      the cache first: rm -rf cpp/build)"
echo "  3. Verify + run:"
echo "       python -c 'from src import trt_cpp_available; print(trt_cpp_available())'  # True"
echo "       python main.py benchmark --model trt_cpp_fp16:models/yolov8s_fp16.engine \\"
echo "            --imgs-input data --device cpu"
echo "========================================"
