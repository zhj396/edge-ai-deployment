cmake_minimum_required(VERSION 3.18)

project(ORT_CPP
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "YOLOv8s inference driver built on the ONNX Runtime C++ API"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ----------------------------------------------------------------------------
# OpenCV — the ddepth overload of cv::dnn::blobFromImage (the CV_32F arg) needs
# OpenCV 4.0+; the 5-arg cv::dnn::NMSBoxes exists since 3.x. So the TECHNICAL
# floor is 4.0; 4.5 is the lowest version actually exercised — the Kaggle T4
# base image (this project's documented deployment target) ships OpenCV 4.5.4,
# and Debian/Ubuntu 22.04 ships 4.6.0; 4.6/4.7+ are fine too. Bump this floor
# only if a newer op is introduced; lower it only down to 4.0 (the blobFromImage
# ddepth overload). Both ort_cpp and the optional trt_cpp backend link
# ort_core_common, which uses only these ops — so the floor is shared.
# ----------------------------------------------------------------------------
find_package(OpenCV 4.5 REQUIRED
    COMPONENTS core imgproc imgcodecs dnn
)

# ----------------------------------------------------------------------------
# ONNX Runtime SDK
#   - ONNXRUNTIME_DIR env var (preferred on Windows / non-default installs)
#   - ./third_party/onnxruntime/ (fallback — the vendored SDK, auto-extracted
#     from the committed release tarball; see the auto-extract block below)
#
# NOTE: the SDK lives under third_party/ so that the cpp/onnxruntime/ path is
# free for the ORT C++ BACKEND source (each backend lives in cpp/<backend>/). Do not put
# backend source in third_party/onnxruntime/.
# ----------------------------------------------------------------------------
if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ORT_DIR "$ENV{ONNXRUNTIME_DIR}")
    set(ORT_VENDORED FALSE)
else()
    set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
    set(ORT_VENDORED TRUE)
endif()

# ----------------------------------------------------------------------------
# Auto-extract the vendored ORT SDK tarball on first configure.
#   The release archive (onnxruntime-<platform>-<ver>.tgz) extracts to a
#   versioned top folder (onnxruntime-linux-x64-1.20.0/); we rename it to
#   onnxruntime/ — the layout the rest of this file and onnxruntime/CMakeLists
#   expect. Only runs for the vendored fallback path (not when ONNXRUNTIME_DIR
#   points elsewhere) and only when the extracted tree is absent. Re-running
#   cmake is a no-op once include/onnxruntime_cxx_api.h exists. The extracted
#   tree is gitignored; the tarball is the committed source of truth.
#
# drvfs/NTFS: the .tgz stores libonnxruntime.so + libonnxruntime.so.1 as
#   *symlinks* to the versioned lib. WSL-on-Windows-drive mounts (/mnt/...) and
#   some NTFS mounts refuse to create them, so tar exits non-zero and leaves
#   the .so missing. We (a) tolerate tar's non-zero exit — the real files still
#   extract — and (b) materialize the missing .so* as real copies of the
#   versioned lib so the link step (needs libonnxruntime.so) and the runtime
#   RPATH load (needs libonnxruntime.so.1, the embedded SONAME) both resolve.
# ----------------------------------------------------------------------------
if(ORT_VENDORED AND NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    file(GLOB ORT_TGZ_FILES "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-*.tgz")
    if(ORT_TGZ_FILES)
        list(GET ORT_TGZ_FILES 0 ORT_TGZ)
        message(STATUS "Auto-extracting ONNX Runtime SDK: ${ORT_TGZ}")
        set(ORT_STAGE "${CMAKE_BINARY_DIR}/ort-extract-stage")
        file(REMOVE_RECURSE "${ORT_STAGE}")
        file(MAKE_DIRECTORY "${ORT_STAGE}")
        # Tolerate non-zero: drvfs may refuse to create the .so symlinks.
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ORT_TGZ}"
            WORKING_DIRECTORY "${ORT_STAGE}"
            RESULT_VARIABLE _ort_rc
            OUTPUT_QUIET ERROR_QUIET)
        file(GLOB _ort_extracted_dirs "${ORT_STAGE}/onnxruntime-*/")
        list(LENGTH _ort_extracted_dirs _ort_extracted_count)
        if(_ort_extracted_count EQUAL 0)
            message(FATAL_ERROR
                "Extraction of ${ORT_TGZ} produced no onnxruntime-*/ folder "
                "in ${ORT_STAGE}. Extract manually: "
                "bash cpp/third_party/setup-onnxruntime.sh")
        endif()
        list(GET _ort_extracted_dirs 0 _ort_src_dir)
        file(REMOVE_RECURSE "${ORT_DIR}")
        file(RENAME "${_ort_src_dir}" "${ORT_DIR}")
        file(REMOVE_RECURSE "${ORT_STAGE}")
        message(STATUS "  -> extracted to ${ORT_DIR}")
    endif()
endif()

# drvfs/NTFS materialization: if the .so symlinks didn't survive extraction,
# copy the versioned lib into libonnxruntime.so(.1) as real files. configure_file
# COPYONLY is a binary-safe single-file copy to an explicit destination name.
if(ORT_VENDORED AND NOT WIN32 AND NOT APPLE)
    file(GLOB _ort_real_so "${ORT_DIR}/lib/libonnxruntime.so.*.*")
    if(_ort_real_so)
        list(GET _ort_real_so 0 _ort_real_so)
        if(NOT EXISTS "${ORT_DIR}/lib/libonnxruntime.so.1")
            configure_file("${_ort_real_so}" "${ORT_DIR}/lib/libonnxruntime.so.1" COPYONLY)
        endif()
        if(NOT EXISTS "${ORT_DIR}/lib/libonnxruntime.so")
            configure_file("${_ort_real_so}" "${ORT_DIR}/lib/libonnxruntime.so" COPYONLY)
        endif()
    endif()
endif()

if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime SDK not found at: ${ORT_DIR}\n"
        "  Set ONNXRUNTIME_DIR to a path containing include/onnxruntime_cxx_api.h\n"
        "  and lib/{libonnxruntime.so,onnxruntime.lib}, or drop a release tarball\n"
        "  (onnxruntime-*-1.20.0.tgz) into cpp/third_party/ and re-run cmake (it\n"
        "  auto-extracts). Download: https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0")
endif()

if(WIN32)
    if(NOT EXISTS "${ORT_DIR}/lib/onnxruntime.lib")
        message(FATAL_ERROR
            "Windows ORT SDK requires ${ORT_DIR}/lib/onnxruntime.lib.\n"
            "  Download onnxruntime-win-x64-1.20.0.zip from GitHub releases.")
    endif()
else()
    if(NOT EXISTS "${ORT_DIR}/lib/libonnxruntime.so")
        message(FATAL_ERROR
            "Linux ORT SDK requires ${ORT_DIR}/lib/libonnxruntime.so.\n"
            "  Download onnxruntime-linux-x64-1.20.0.tgz from GitHub releases.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Third-party deps via FetchContent (shared across backends — declared at top so
# both common/ and onnxruntime/ see the *_SOURCE_DIR variables)
#   - cxxopts: header-only CLI parser (MIT, ~2k LOC)
#   - yaml-cpp: 0.8.0 (MIT, header+small static lib)
# ----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.1.1
    GIT_SHALLOW    TRUE
)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)

# Reduce yaml-cpp surface — we only need the parser
set(YAML_CPP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB  OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL        OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cxxopts yaml-cpp)

# ----------------------------------------------------------------------------
# Optional: AddressSanitizer + UBSan (Linux dev only) — applied to every target
# ----------------------------------------------------------------------------
option(YOLOV8_SANITIZE "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

# ----------------------------------------------------------------------------
# Build the shared core, then the ORT backend. Each backend gets its own
# cpp/<backend>/ + CMakeLists.txt.
# ----------------------------------------------------------------------------
add_subdirectory(common)
add_subdirectory(onnxruntime)

# ----------------------------------------------------------------------------
# Optional: in-process TensorRT C++ pybind11 backend (_trt_cpp module). OFF by
# default — requires TensorRT + CUDA + pybind11 (none vendored; ~1.5GB SDK);
# a box without them must NOT fail the ort_cpp build (mirrors the Python side's
# optional-dep posture: tensorrt / cuda-python are NOT in the base pin set).
# The module is a forward-only accelerator (deserialize a .engine + run
# execute_async_v3 over the CUDA Runtime API); preprocess/postprocess stay
# Python. See docs/TENSORRT.md (trt_cpp section).
#   cmake -S cpp -B cpp/build -DBUILD_TRT_CPP=ON
#   cmake --build cpp/build --target _trt_cpp
# ----------------------------------------------------------------------------
option(BUILD_TRT_CPP "Build the in-process TensorRT C++ pybind11 backend (_trt_cpp)" OFF)
if(BUILD_TRT_CPP)
    add_subdirectory(tensorrt)
endif()

# ----------------------------------------------------------------------------
# Status banner
# ----------------------------------------------------------------------------
message(STATUS "===================================================")
message(STATUS "ORT_CPP ${PROJECT_VERSION}")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV        : ${OpenCV_VERSION}")
message(STATUS "  ORT SDK       : ${ORT_DIR}")
message(STATUS "  yaml-cpp      : ${yaml-cpp_VERSION}")
message(STATUS "  cxxopts       : ${cxxopts_VERSION}")
message(STATUS "===================================================")
