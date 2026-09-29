# Backend-agnostic shared core — the C++ mirror of src/preprocess.py +
# src/postprocess.py (letterbox + output-shape parse + decode/scale-back/NMS),
# plus shared infra (types / logger / timer / yaml loader / config / draw_utils).
#
# Deliberately has NO ONNX Runtime dependency: common/ links only OpenCV (+ the
# yaml parser), so a future cpp/tensorrt/ backend can reuse this core unchanged
# by linking ort_core_common and handing it a raw float* + shape.
add_library(ort_core_common STATIC
    src/letterbox.cpp
    src/nms_decode.cpp
    src/yaml_classes.cpp
    src/draw_utils.cpp
)

# common/include is PUBLIC so backend targets that link ort_core_common inherit it
# (the backend Preprocessor/Postprocessor #include "letterbox.h"/"nms_decode.h").
target_include_directories(ort_core_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# OpenCV is PUBLIC (every backend uses cv types via the core). yaml-cpp is PRIVATE —
# only yaml_classes.cpp needs it, and the header never exposes yaml types, so a static
# ort_core_common consumer never needs yaml headers (the symbols resolve at final link).
target_link_libraries(ort_core_common
    PUBLIC  ${OpenCV_LIBS}
    PRIVATE yaml-cpp::yaml-cpp
)

# cxxopts triggers signed/unsigned mismatch on MSVC /W4 — silence on the core too.
if(MSVC)
    target_compile_options(ort_core_common PRIVATE /wd4244 /wd4267 /wd4800)
endif()

if(YOLOV8_SANITIZE AND NOT WIN32)
    target_compile_options(ort_core_common PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_libraries(ort_core_common PRIVATE -fsanitize=address,undefined)
endif()
