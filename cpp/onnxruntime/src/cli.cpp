#include "cli.h"

#include <iostream>
#include <string>

#include <cxxopts.hpp>

void print_help(const char* prog) {
    std::cout <<
        "YOLOv8s ONNX Runtime C++ Inference\n"
        "Usage: " << prog << " [options]\n"
        "\n"
        "Input source:\n"
        "  -i, --image <path>           single image inference\n"
        "  -d, --dir <path>             directory batch mode (jpg/jpeg/png/bmp)\n"
        "      --benchmark <int>        run N timed iterations (modifier: combine with\n"
        "                               --image <one> or --dir <many, capped>; alone =\n"
        "                               default sample image)\n"
        "      --max-images <int>       cap on images loaded for --dir benchmark (default: 50)\n"
        "\n"
        "Model & thresholds:\n"
        "  -m, --model <path>           ONNX model path (default: models/yolov8s_fp32.onnx)\n"
        "  -y, --yaml <path>            data.yaml for class names (default: data/data.yaml)\n"
        "      --imgsz <int>            input edge size (default: 640)\n"
        "      --conf  <float>          confidence threshold (default: 0.25)\n"
        "      --iou   <float>          IoU threshold for NMS (default: 0.45)\n"
        "\n"
        "Output:\n"
        "  -o, --output <path>          output file (--image) or dir (--dir/--benchmark)\n"
        "                               default: results/\n"
        "      --save-json <path>       dump benchmark summary to JSON\n"
        "      --dump-raw-dir <dir>     dump raw pre-NMS forward tensors (.npy per image)\n"
        "                               for the Python consistency harness; requires --dir\n"
        "\n"
        "Performance:\n"
        "      --intra-op-threads <int> ORT intra-op threads (default: 0 = ORT default)\n"
        "      --inter-op-threads <int> ORT inter-op threads (default: 0 = ORT default)\n"
        "      --workers <int>          parallel --dir workers (default: 0 = hardware_concurrency)\n"
        "\n"
        "Other:\n"
        "  -h, --help                   show this help and exit\n";
}

CLIArgs parse_cli(int argc, char** argv) {
    CLIArgs args;

    cxxopts::Options options(argv[0], "YOLOv8s ONNX Runtime C++ Inference");

    options.add_options()
        ("m,model",            "ONNX model path",
            cxxopts::value<std::string>())
        ("i,image",            "single image path",
            cxxopts::value<std::string>())
        ("d,dir",              "directory of images",
            cxxopts::value<std::string>())
        ("y,yaml",             "data.yaml for class names",
            cxxopts::value<std::string>())
        ("o,output",           "output path",
            cxxopts::value<std::string>())
        ("imgsz",              "input edge size",
            cxxopts::value<int>()->default_value("640"))
        ("conf",               "confidence threshold",
            cxxopts::value<float>()->default_value("0.25"))
        ("iou",                "IoU threshold for NMS",
            cxxopts::value<float>()->default_value("0.45"))
        ("benchmark",          "N benchmark iterations (0 = disabled)",
            cxxopts::value<int>()->default_value("0"))
        ("max-images",         "cap on images loaded for --dir benchmark",
            cxxopts::value<int>()->default_value("50"))
        ("intra-op-threads",   "ORT intra-op threads (0 = ORT default)",
            cxxopts::value<int>()->default_value("0"))
        ("inter-op-threads",   "ORT inter-op threads (0 = ORT default)",
            cxxopts::value<int>()->default_value("0"))
        ("workers",            "parallel --dir workers (0 = hardware_concurrency)",
            cxxopts::value<int>()->default_value("0"))
        ("save-json",          "benchmark JSON output",
            cxxopts::value<std::string>())
        ("dump-raw-dir",       "dump raw pre-NMS forward tensors (.npy per image); requires --dir",
            cxxopts::value<std::string>())
        ("h,help",             "show help");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            args.help = true;
            return args;
        }

        if (result.count("model"))     args.model      = result["model"].as<std::string>();
        if (result.count("image"))     args.image      = result["image"].as<std::string>();
        if (result.count("dir"))       args.dir        = result["dir"].as<std::string>();
        if (result.count("yaml"))      args.yaml_path  = result["yaml"].as<std::string>();
        if (result.count("output"))    args.output     = result["output"].as<std::string>();
        if (result.count("save-json")) args.save_json  = result["save-json"].as<std::string>();
        if (result.count("dump-raw-dir")) args.dump_raw_dir = result["dump-raw-dir"].as<std::string>();

        args.imgsz            = result["imgsz"].as<int>();
        args.conf             = result["conf"].as<float>();
        args.iou              = result["iou"].as<float>();
        args.benchmark        = result["benchmark"].as<int>();
        args.max_images       = result["max-images"].as<int>();
        args.intra_op_threads = result["intra-op-threads"].as<int>();
        args.inter_op_threads = result["inter-op-threads"].as<int>();
        args.workers          = result["workers"].as<int>();

        // --image and --dir are mutually exclusive image sources. --benchmark is a
        // modifier that may combine with either (or stand alone with the default
        // sample image). So: reject --image+--dir; require at least one of
        // --image/--dir/--benchmark.
        const bool has_img = !args.image.empty();
        const bool has_dir = !args.dir.empty();
        const bool bench   = args.benchmark > 0;
        if (has_img && has_dir) {
            args.valid = false;
            args.error = "Conflicting image sources: use --image OR --dir, not both.";
        } else if (!bench && !has_img && !has_dir) {
            args.valid = false;
            args.error = "No input source: provide --image, --dir, or --benchmark N.";
        }
        if (!args.dump_raw_dir.empty() && args.dir.empty()) {
            args.valid = false;
            args.error = "--dump-raw-dir requires --dir <in_dir>.";
        }

        if (args.imgsz <= 0) {
            args.valid = false;
            args.error = "--imgsz must be positive.";
        }
        if (args.conf < 0.0f || args.conf > 1.0f) {
            args.valid = false;
            args.error = "--conf must be in [0, 1].";
        }
        if (args.iou < 0.0f || args.iou > 1.0f) {
            args.valid = false;
            args.error = "--iou must be in [0, 1].";
        }
        if (args.intra_op_threads < 0 || args.inter_op_threads < 0) {
            args.valid = false;
            args.error = "--intra-op-threads and --inter-op-threads must be >= 0.";
        }
        if (args.workers < 0) {
            args.valid = false;
            args.error = "--workers must be >= 0.";
        }
        if (args.max_images < 1) {
            args.valid = false;
            args.error = "--max-images must be >= 1.";
        }
    }
    catch (const cxxopts::exceptions::exception& e) {
        args.valid = false;
        args.error = std::string("Argument parse error: ") + e.what();
    }

    return args;
}
