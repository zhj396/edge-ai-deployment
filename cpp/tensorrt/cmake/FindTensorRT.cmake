# FindTensorRT.cmake — locate TensorRT 8.6+ / 10.x across common install locations.
#
# Copied from the standalone reference at yolov8s_trt_cpp/cmake/FindTensorRT.cmake
# (proven across Kaggle NVCR / Debian multiarch / tarball / Windows), with one
# addition: $ENV{TENSORRT_DIR} alongside $ENV{TENSORRT_ROOT} to match the repo's
# own SDK-discovery convention (ORT_DIR / ONNXRUNTIME_DIR in cpp/CMakeLists.txt).
#
# Why custom? The CMake-shipped FindTensorRT module (CUDA 12) does not search
# the multi-arch Debian paths used by NVCR / Kaggle base images:
#   /usr/lib/x86_64-linux-gnu/libnvinfer.so
# We add explicit fall-backs so `cmake ..` "just works" on:
#   * Kaggle nvidia/tensorrt:24.xx-py3   (NVCR base)
#   * Local tarball install               (/usr/local/TensorRT-*)
#   * apt (nvidia-tensorrt)               (/usr/lib/x86_64-linux-gnu)
#   * Windows dev box                     (C:/TensorRT or $TENSORRT_ROOT)
#
# Output variables (cache):
#   TENSORRT_FOUND         — TRUE if a usable install was found
#   TENSORRT_INCLUDE_DIRS  — list of header dirs (NvInfer.h, NvOnnxParser.h)
#   TENSORRT_LIBRARIES      — list of full paths to .so/.dll/.a (nvinfer only)
#   TENSORRT_VERSION       — major.minor.patch as string

include(FindPackageHandleStandardArgs)

set(_trt_search_paths
    # Explicit override FIRST — a CMake var (-DTENSORRT_ROOT= / -DTENSORRT_DIR=)
    # or an env var must win over auto-discovered system paths (matches the
    # repo's ONNXRUNTIME_DIR convention in cpp/CMakeLists.txt). The CMake-var
    # form is load-bearing on Kaggle/Jupyter, where each `!` cell is a separate
    # subshell so a `source`d script's exported env vars die before the next
    # cell's cmake runs — pass -DTENSORRT_ROOT=<path> instead (it persists in
    # CMakeCache.txt). This is also load-bearing when the system has a DIFFERENT
    # TRT version than the project pin: e.g. apt installs TRT 11.1 to
    # /usr/lib/x86_64-linux-gnu, but the project pins TRT 10.4.0 (the tensorrt
    # wheel that builds the .engine) — point TENSORRT_ROOT at a 10.4.0 tarball
    # to override the system 11.1. Without override-first ordering the system
    # version always wins and the engine fails to deserialize. Empty/un set
    # entries expand to "" and are skipped by find_path/find_library.
    ${TENSORRT_DIR}
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_DIR}
    $ENV{TENSORRT_ROOT}
    # Kaggle / Debian multiarch
    /usr/lib/x86_64-linux-gnu
    /usr/include/x86_64-linux-gnu
    # Generic Linux
    /usr/local/lib
    /usr/local/include
    /usr/lib
    /usr/include
    # Tarball install
    /usr/local/tensorrt
    /opt/tensorrt
    /opt/TensorRT-10.4.0.26
    /kaggle/working/TensorRT-10.4.0.26
    $ENV{HOME}/TensorRT
    # Windows
    C:/TensorRT
)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${_trt_search_paths}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer libnvinfer
    PATHS ${_trt_search_paths}
    NO_DEFAULT_PATH
)

# Fallback: find_library matches lib<nvinfer>.so but NOT the versioned SONAME
# libnvinfer.so.10. NVIDIA *dev* tarballs ship the libnvinfer.so dev symlink,
# but a runtime-only extract (or a manual unzip) may have only libnvinfer.so.10
# — which made find_library return NOTFOUND ("TensorRT not found") on a real
# Kaggle build. Glob for any libnvinfer.so* and use the first; linking against
# the versioned lib is fine (the linker records its SONAME, the runtime loader
# resolves libnvinfer.so.10). scripts/trt_cpp_setup.sh also creates the symlink.
if(NOT TENSORRT_NVINFER_LIB)
    foreach(_trt_p ${_trt_search_paths})
        if(_trt_p)
            file(GLOB _nvinfer_glob
                "${_trt_p}/libnvinfer.so*" "${_trt_p}/lib/libnvinfer.so*")
            if(_nvinfer_glob)
                list(SORT _nvinfer_glob)
                list(GET _nvinfer_glob 0 TENSORRT_NVINFER_LIB)
                message(STATUS
                    "TensorRT nvinfer: find_library missed the dev symlink; "
                    "globbed ${TENSORRT_NVINFER_LIB}")
                break()
            endif()
        endif()
    endforeach()
endif()

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser libnvonnxparser
    PATHS ${_trt_search_paths}
    NO_DEFAULT_PATH
)

find_library(TENSORRT_NVPARSERS_LIB
    NAMES nvparsers libnvparsers
    PATHS ${_trt_search_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TENSORRT_INCLUDE_DIR TENSORRT_NVINFER_LIB
    VERSION_VAR TENSORRT_VERSION
)

if(TensorRT_FOUND)
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})

    # nvonnxparser / nvparsers are optional for pure inference (deserialise-only)
    # path but required for the EngineBuilder. The trt_cpp backend is deserialize-
    # only, so it links nvinfer alone. Caller can check TENSORRT_NVONNXPARSER_LIB.
    set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIB})
    if(TARGET CUDA::cudart)
        # cudart is a CUDA dependency — keep it out of TENSORRT_LIBRARIES so the
        # find_library in CMakeLists can layer CUDA + TensorRT cleanly.
    endif()

    # Extract major version from NvInfer.h
    if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
        file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_v)
        string(REGEX MATCH "NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _ "${_trt_v}")
        set(TENSORRT_MAJOR "${CMAKE_MATCH_1}")
        string(REGEX MATCH "NV_TENSORRT_MINOR[ \t]+([0-9]+)" _ "${_trt_v}")
        set(TENSORRT_MINOR "${CMAKE_MATCH_1}")
        string(REGEX MATCH "NV_TENSORRT_PATCH[ \t]+([0-9]+)" _ "${_trt_v}")
        set(TENSORRT_PATCH "${CMAKE_MATCH_1}")
        if(TENSORRT_MAJOR)
            set(TENSORRT_VERSION "${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCH}")
        endif()
    endif()

    message(STATUS "TensorRT found: ${TENSORRT_VERSION} (headers: ${TENSORRT_INCLUDE_DIR})")
    if(TENSORRT_NVONNXPARSER_LIB)
        message(STATUS "  nvonnxparser: ${TENSORRT_NVONNXPARSER_LIB}")
    else()
        message(STATUS "  nvonnxparser not found (optional — trt_cpp is deserialize-only)")
    endif()
endif()

# Internal helper: print all candidate paths that were searched. Used by
# the user-friendly error message in the top-level CMakeLists.txt.
function(trt_cpp_print_search_paths)
    message(STATUS "Searched paths for TensorRT:")
    foreach(p ${_trt_search_paths})
        message(STATUS "  ${p}")
    endforeach()
endfunction()

# Mark internal variables as advanced to keep cache GUI clean.
mark_as_advanced(TENSORRT_INCLUDE_DIR TENSORRT_NVINFER_LIB
                 TENSORRT_NVONNXPARSER_LIB TENSORRT_NVPARSERS_LIB)
