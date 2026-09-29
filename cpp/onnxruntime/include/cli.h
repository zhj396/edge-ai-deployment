#pragma once

#include <string>

// Parsed command-line arguments. `valid == false` means `error` holds
// a human-readable reason; `main()` should print it and exit non-zero.
// `help == true` means the user passed --help; `main()` should print
// usage and exit zero.
struct CLIArgs {
    // Input source. --image and --dir are mutually exclusive image sources;
    // --benchmark is a *modifier* that may combine with --image (single-image
    // benchmark) or --dir (multi-image benchmark, capped by --max-images).
    // With no --image/--dir, --benchmark falls back to a default sample image.
    std::string image;     // --image    single image
    std::string dir;       // --dir      directory batch (jpg/jpeg/png/bmp)
    int         benchmark = 0;  // --benchmark N   N timed iters (0 = disabled)
    int         max_images = 50; // --max-images  cap on images loaded for --dir benchmark

    // Model & thresholds
    std::string model;     // --model    ONNX path
    std::string yaml_path; // --yaml     data.yaml for class names
    int         imgsz = 640;
    float       conf  = 0.25f;
    float       iou   = 0.45f;

    // Output
    std::string output;     // --output   file (--image) or dir (--dir/--benchmark)
    std::string save_json;  // --save-json

    // Performance
    int intra_op_threads = 0;  // --intra-op-threads   0 = ORT default (all cores)
    int inter_op_threads = 0;  // --inter-op-threads   0 = ORT default
    int workers         = 0;  // --workers             # of parallel --dir workers
                              //                       <= 0 = hardware_concurrency()

    // Status flags (set by parser)
    bool help   = false;
    bool valid  = true;
    std::string error;
};

// Parse argv into a CLIArgs. Never throws; sets `valid = false` on error.
CLIArgs parse_cli(int argc, char** argv);

// Pretty-print usage to stdout.
void print_help(const char* prog_name);
