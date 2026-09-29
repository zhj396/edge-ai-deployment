#pragma once

#include <string>

// Default runtime configuration. The CLI parser in cli.h produces a more
// explicit struct (CLIArgs); AppConfig holds the default paths and
// thresholds used by the CLI driver (main.cpp) and the default sample
// image for --benchmark.
struct AppConfig {
    // Model: the canonical 12-class YOLOv8s ONNX (FP32) exported by the Python
    // project's `export` subcommand. Mirrors cli/__init__.py::DEFAULT_MODEL_FP32
    // so the C++ deployment points at the same artifact the Python backends
    // consume. NOTE: `models/yolov8s.onnx` does NOT exist in this repo — the
    // export produces `yolov8s_fp32.onnx` (and `_int8.onnx`).
    std::string model_path = "models/yolov8s_fp32.onnx";

    // Single image mode — used when neither --image nor --dir is provided, and as
    // the default image for --benchmark when no --image is given. Resolved relative
    // to the executable's working directory. Points at an image shipped under
    // data/images/val (the Release val split).
    std::string image_path = "data/images/val/000000001532.jpg";

    // Class names file. Mirrors the Python project's data.yaml so the
    // C++ deployment uses a single source of truth.
    std::string yaml_path = "data/data.yaml";

    // Where annotated images and benchmark artifacts are written.
    // For --image mode, this is the output file path.
    // For --dir mode, this is the output directory (created if missing).
    std::string output_path = "results";

    // Letterbox input edge. YOLOv8s was trained at 640x640; smaller values
    // trade accuracy for latency.
    int input_size = 640;

    // Confidence and IoU thresholds. Defaults match Ultralytics.
    float conf_threshold = 0.25f;
    float iou_threshold  = 0.45f;

    // Benchmark: 0 = disabled (single infer), >0 = N timed iterations.
    int benchmark_runs = 0;
};
