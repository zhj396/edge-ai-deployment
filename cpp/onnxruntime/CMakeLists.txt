# ONNX Runtime C++ backend — the `ort_cpp` executable. Owns the ORT session
# (yolov8.cpp) + the ORT-specific tensor wrappers (preprocessor.cpp /
# postprocessor.cpp) that delegate the letterbox/decode numerics to
# ort_core_common. Plus the CLI (cli.cpp), the demo driver (main.cpp), and the
# benchmark loop (benchmark.cpp).
add_executable(ort_cpp
    src/main.cpp
    src/yolov8.cpp
    src/preprocessor.cpp
    src/postprocessor.cpp
    src/cli.cpp
    src/benchmark.cpp
)

# Backend-private headers (yolov8.h / preprocessor.h / postprocessor.h / cli.h /
# benchmark.h) + the ORT SDK headers + cxxopts. common/include is inherited
# transitively via the PUBLIC include on ort_core_common (linked below).
target_include_directories(ort_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ORT_DIR}/include
        ${cxxopts_SOURCE_DIR}/include
)

# ort_core_common: PUBLIC-propagates OpenCV + common/include. yaml-cpp resolves
# from the static core's PRIVATE link deps at final link time. cxxopts is a
# header-only interface lib — link it here for cli.cpp's includes.
target_link_libraries(ort_cpp
    PRIVATE
        ort_core_common
        cxxopts::cxxopts
        yaml-cpp::yaml-cpp
)

# Link the ORT SDK library — artifact name differs by platform.
#   Linux:   libonnxruntime.so
#   macOS:   libonnxruntime.dylib
#   Windows: onnxruntime.lib  (the .dll is loaded at runtime via PATH)
if(WIN32)
    target_link_libraries(ort_cpp PRIVATE ${ORT_DIR}/lib/onnxruntime.lib)
    target_link_directories  (ort_cpp PRIVATE ${ORT_DIR}/lib)
elseif(APPLE)
    target_link_libraries(ort_cpp PRIVATE ${ORT_DIR}/lib/libonnxruntime.dylib)
else()
    target_link_libraries(ort_cpp PRIVATE ${ORT_DIR}/lib/libonnxruntime.so)
endif()

# RPATH so the ORT .so resolves next to the binary on Linux
if(NOT WIN32 AND NOT APPLE)
    set_target_properties(ort_cpp PROPERTIES
        BUILD_RPATH   "${ORT_DIR}/lib"
        INSTALL_RPATH "${ORT_DIR}/lib"
    )
endif()

# cxxopts triggers signed/unsigned mismatch on MSVC /W4 — silence them.
if(MSVC)
    target_compile_options(ort_cpp PRIVATE /wd4244 /wd4267 /wd4800)
endif()

if(YOLOV8_SANITIZE AND NOT WIN32)
    message(STATUS "Building ort_cpp with ASan + UBSan")
    target_compile_options(ort_cpp PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_libraries(ort_cpp PRIVATE -fsanitize=address,undefined)
endif()

# ----------------------------------------------------------------------------
# Install
# ----------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS ort_cpp
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
