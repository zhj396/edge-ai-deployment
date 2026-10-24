# In-process TensorRT C++ pybind11 backend (_trt_cpp). Forward-only accelerator:
# deserialises a .engine and runs enqueueV3 over the CUDA Runtime API,
# returning the raw [bs, 4+nc, 8400] head tensor to Python. Same metric surface
# (native mAP + kernel_latency_ms) as the Python tensorrt: path — see
# docs/TENSORRT.md (the trt_cpp backend).
#
# Optional + OFF by default (gated in the parent cpp/CMakeLists.txt by
# BUILD_TRT_CPP). A box without TensorRT + CUDA + pybind11 must NOT fail the
# ort_cpp build. The build only enters this subdir when the user passes
# -DBUILD_TRT_CPP=ON, mirroring the Python side's optional-dep posture
# (tensorrt / cuda-python are NOT in the base pin set).
#
# Build:
#   cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON
#   cmake --build cpp/build --target _trt_cpp
#
# Run from the venv so pybind11 finds the CLI's interpreter (avoids Python-ABI
# mismatch): the .so/.pyd is loaded lazily by src.benchmark.resolve_trt_cpp_module.
#
# The engine is LOAD-ONLY: it deserialises a .engine built by the Python
# build_tensorrt_engine (Path A) / export_tensorrt_ultralytics (Path B). It
# therefore links nvinfer ONLY — NO nvonnxparser (the parser is for building,
# which stays in Python). See FindTensorRT.cmake (+$ENV{TENSORRT_DIR}).

# --- pybind11: try a system install first, else FetchContent v2.13.6.      ---
# pybind11's config mode resolves the interpreter + Development.Module.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found system-wide; FetchContent'ing v2.13.6")
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(pybind11)
endif()

# --- CUDA: enable the language + CUDAToolkit (CMake 3.17+ for CUDAToolkit). ---
# The default must be set BEFORE enable_language: enabling CUDA initializes
# CMAKE_CUDA_ARCHITECTURES to the CMake default (or the environment's value),
# so a `NOT DEFINED` check after that point never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default sm_75 = Tesla T4 (the project's documented Turing target). Overridable
    # via -DCMAKE_CUDA_ARCHITECTURES=... (e.g. 80 for Ampere, "75;86" for both).
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# --- TensorRT: reuse the reference finder (copied into ./cmake).            ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindTensorRT)
if(NOT TensorRT_FOUND)
    message(FATAL_ERROR
        "TensorRT not found. Set TENSORRT_DIR / TENSORRT_ROOT, or install to a "
        "standard path (/usr/include/x86_64-linux-gnu + /usr/lib/x86_64-linux-gnu "
        "on Debian/Kaggle, C:/TensorRT on Windows). The trt_cpp backend is "
        "optional — build with -DBUILD_TRT_CPP=OFF (the default) to skip it.")
endif()

# --- The module. pybind11_add_module names it _trt_cpp; emits               ---
# _trt_cpp<SOABI>.so | _trt_cpp<SOABI>.pyd. The Python loader globs the
# SOABI-suffixed name (src.benchmark.resolve_trt_cpp_module).
pybind11_add_module(_trt_cpp
    src/trt_engine.cpp
    src/bindings.cpp
)
target_include_directories(_trt_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIRS}
)
target_link_libraries(_trt_cpp
    PRIVATE
        ort_core_common          # documented contract (cpp/common/CMakeLists.txt:5-7)
        ${TENSORRT_LIBRARIES}    # nvinfer only (no nvonnxparser — deserialize-only)
        CUDA::cudart             # Runtime API: cudaMalloc/cudaMemcpyAsync/cudaStreamSync
)
# RPATH so libnvinfer.so.10 resolves at runtime from a non-standard prefix (e.g.
# a TRT 10.4.0 tarball in /opt). Essential when TENSORRT_DIR points at a tarball
# whose lib/ is NOT on the default loader path; harmless for an apt install
# (/usr/lib/x86_64-linux-gnu is already standard). Mirrors how ort_cpp sets
# BUILD_RPATH/INSTALL_RPATH for the ORT SDK. Without it the module loads but
# dlopen fails at import with "libnvinfer.so.10: cannot open shared object file".
get_filename_component(_trt_lib_dir "${TENSORRT_NVINFER_LIB}" DIRECTORY)
set_target_properties(_trt_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tensorrt"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tensorrt"
    BUILD_RPATH   "${_trt_lib_dir}"
    INSTALL_RPATH "${_trt_lib_dir}")

# Third-party headers (pybind11 / NvInfer / CUDA) trigger signed/unsigned mismatch
# (/wd4244 /wd4267 /wd4800) and deprecated-declaration warnings (/wd4996) on MSVC
# — silence them.
if(MSVC)
    target_compile_options(_trt_cpp PRIVATE /wd4244 /wd4267 /wd4800 /wd4996)
else()
    # TRT 10.x's own headers emit -Wdeprecated-declarations noise (IPluginV2,
    # IInt8EntropyCalibrator, IGpuAllocator::allocate, ... — all deprecated but
    # still present). These are NVIDIA's deprecations in THEIR headers, not our
    # code; silence them so the build output stays readable.
    target_compile_options(_trt_cpp PRIVATE -Wno-deprecated-declarations)
endif()

if(YOLOV8_SANITIZE AND NOT WIN32)
    target_compile_options(_trt_cpp PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_libraries(_trt_cpp PRIVATE -fsanitize=address,undefined)
endif()

message(STATUS "  _trt_cpp      : pybind11 module -> ${CMAKE_BINARY_DIR}/tensorrt")
message(STATUS "  TensorRT     : ${TENSORRT_VERSION} (nvinfer only)")
message(STATUS "  CUDA arch    : ${CMAKE_CUDA_ARCHITECTURES}")
