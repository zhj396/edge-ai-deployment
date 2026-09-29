#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// Console API (SetConsoleOutputCP) lives here; don't rely on OpenCV/ORT headers
// transitively pulling in <windows.h> — that breaks builds without that chain.
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "cli.h"
#include "config.h"
#include "draw_utils.h"
#include "benchmark.h"
#include "logger.h"
#include "timer.h"
#include "yaml_classes.h"
#include "yolov8.h"

namespace fs = std::filesystem;

namespace {

// Hardcoded fallback used only when the yaml file cannot be loaded
// (missing file, parse error). Mirrors data/data.yaml so the demo
// still runs out of the box without --yaml pointing to anything.
const std::vector<std::string> kFallbackClassNames = {
    "person", "bicycle", "car", "bus", "truck", "motorcycle",
    "dog",    "cat",     "chair", "bottle", "backpack", "traffic light"
};

// Try to load class names from yaml. On any failure, log a warning and
// fall back to the hardcoded list — the demo should still run.
std::vector<std::string> load_or_fallback(const std::string& yaml_path) {
    try {
        auto names = load_class_names(yaml_path);
        LOG_INFO("Loaded " + std::to_string(names.size())
                 + " class names from " + yaml_path);
        return names;
    }
    catch (const std::exception& e) {
        LOG_WARN(std::string("Failed to load class names from ") + yaml_path
                 + ": " + e.what());
        LOG_WARN("Falling back to hardcoded 12-class list.");
        return kFallbackClassNames;
    }
}

bool ensure_parent_dir(const std::string& path) {
    fs::path p(path);
    fs::path parent = p.has_parent_path() ? p.parent_path() : fs::current_path();
    std::error_code ec;
    if (!fs::exists(parent, ec)) {
        if (!fs::create_directories(parent, ec)) {
            LOG_ERROR("Failed to create output directory: " + parent.string());
            return false;
        }
    }
    return true;
}

// Forward decl — defined below; run_benchmark/run_directory both test it.
bool is_image_extension(const fs::path& p);

bool run_single_image(
    YOLOv8& detector,
    const CLIArgs& args,
    const std::vector<std::string>& class_names
) {
    if (!fs::exists(args.image)) {
        LOG_ERROR("Image not found: " + args.image);
        return false;
    }
    cv::Mat image = cv::imread(args.image);
    if (image.empty()) {
        LOG_ERROR("Failed to load image: " + args.image);
        return false;
    }

    // Warm-up — first Run() pays for ORT graph optimizations and any
    // thread-pool spin-up; we don't want this in the timed number.
    for (int i = 0; i < 3; ++i) {
        detector.detect(image, args.conf, args.iou);
    }

    // Wall-clock the detect() call. Timer is a start->elapsed stopwatch (no stop());
    // elapsedMs() reads from start(). The per-stage breakdown in profile() is finer.
    Timer total_timer;
    total_timer.start();
    auto detections = detector.detect(image, args.conf, args.iou);
    const double wall_ms = total_timer.elapsedMs();

    const auto& p = detector.profile();
    LOG_INFO("=== Single image ===");
    LOG_INFO("  file        : " + args.image);
    LOG_INFO("  detections  : " + std::to_string(detections.size()));
    LOG_INFO("  preprocess  : " + std::to_string(p.preprocess_ms)  + " ms");
    LOG_INFO("  infer       : " + std::to_string(p.infer_ms)       + " ms");
    LOG_INFO("  postprocess : " + std::to_string(p.postprocess_ms) + " ms");
    LOG_INFO("  total       : " + std::to_string(p.total_ms)       + " ms");
    LOG_INFO("  wall        : " + std::to_string(wall_ms) + " ms");

    if (!args.output.empty()) {
        if (!ensure_parent_dir(args.output)) return false;
        drawDetections(image, detections);
        if (!cv::imwrite(args.output, image)) {
            LOG_ERROR("Failed to write output image: " + args.output);
            return false;
        }
        LOG_INFO("  saved to    : " + args.output);
    }
    return true;
}

bool run_benchmark(
    YOLOv8& detector,
    const CLIArgs& args
) {
    // --benchmark is a modifier: it times N iterations over an image set that comes
    // from --dir (multi, capped by --max-images, cycled for N>set) or --image
    // (single), or the default sample image if neither is given. A single repeated
    // image keeps the input buffer cache-resident across iterations; a multi-image
    // dir run adds per-image decode and cold-cache input traffic to the timings.
    std::vector<cv::Mat> images;
    std::vector<std::string> paths;

    if (!args.dir.empty()) {
        fs::path dir(args.dir);
        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
            LOG_ERROR("Directory not found: " + args.dir);
            return false;
        }
        std::vector<fs::path> image_paths;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (entry.is_regular_file(ec) && is_image_extension(entry.path())) {
                image_paths.push_back(entry.path());
            }
        }
        std::sort(image_paths.begin(), image_paths.end());
        if (image_paths.empty()) {
            LOG_ERROR("No images (.jpg/.jpeg/.png/.bmp) found in: " + args.dir);
            return false;
        }
        const size_t cap = static_cast<size_t>(std::max(1, args.max_images));
        const size_t load_n = std::min(cap, image_paths.size());
        for (size_t i = 0; i < load_n; ++i) {
            cv::Mat img = cv::imread(image_paths[i].string());
            if (!img.empty()) {
                images.push_back(img);
                paths.push_back(image_paths[i].filename().string());
            }
        }
        if (images.empty()) {
            LOG_ERROR("Failed to decode any image in: " + args.dir);
            return false;
        }
        if (load_n < image_paths.size()) {
            LOG_INFO("  loaded " + std::to_string(images.size()) + "/"
                     + std::to_string(image_paths.size())
                     + " images (capped by --max-images=" + std::to_string(cap) + ")");
        } else {
            LOG_INFO("  loaded " + std::to_string(images.size()) + " images");
        }
    } else {
        std::string image_path = args.image;
        if (image_path.empty()) {
            AppConfig defaults;
            image_path = defaults.image_path;
            LOG_INFO("  no --image/--dir given; benchmarking default: " + image_path);
        }
        if (!fs::exists(image_path)) {
            LOG_ERROR("Image not found: " + image_path
                      + " (pass --image <path> or --dir <path>)");
            return false;
        }
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            LOG_ERROR("Failed to load image: " + image_path);
            return false;
        }
        images = { image };
        paths = { image_path };
    }

    const int warmup = 3;
    const int iters  = std::max(1, args.benchmark);

    std::vector<BenchmarkRow> rows;
    BenchmarkSummary summary = Benchmark::run(
        detector,
        images, paths,
        warmup, iters,
        args.conf, args.iou,
        &rows
    );

    Benchmark::print_summary(summary, std::cout);

    if (!args.output.empty()) {
        // args.output is the CSV directory here — create it itself, not just
        // its parent (a fresh checkout has no results/; it is gitignored).
        std::error_code mk_ec;
        fs::create_directories(args.output, mk_ec);
        Benchmark::write_csv(args.output + "/benchmark.csv", rows);
    }
    if (!args.save_json.empty()) {
        if (!ensure_parent_dir(args.save_json)) return false;
        Benchmark::write_json(args.save_json, summary, args.model);
    }
    return true;
}

bool is_image_extension(const fs::path& p) {
    static const std::array<std::string, 4> exts = {
        ".jpg", ".jpeg", ".png", ".bmp"
    };
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const auto& e : exts) {
        if (ext == e) return true;
    }
    return false;
}

// Minimal fixed-size worker pool. Each worker pulls std::function<void(int)>
// tasks from a shared queue and runs them with its OWN thread index (0..N-1).
// wait_idle() blocks until the queue is empty and no tasks are in flight.
//
// The thread index passed to each task is what makes per-thread resources safe:
// run_directory pins one YOLOv8 instance per worker_id, so each detector is
// only ever touched by its owning thread. (A YOLOv8 instance is NOT reentrant —
// the Preprocessor reuses one NCHW buffer/letterbox canvas/tensor_ per detect()
// and detect() mutates profile_result_ — so two threads on the same instance is
// a data race. Tagging tasks with `i % n_workers` instead of the thread's own
// index would NOT serialize same-wid tasks: the shared queue lets any free
// thread grab any task, so two images tagged wid 0 could run concurrently on
// two threads against worker_detectors[0]. The thread-pinned worker_id is the
// THREAD's identity, unique per thread, and each thread runs one task at a
// time.)
//
// Synchronization: a single mutex guards the queue + in_flight counter; two
// condition variables signal "new work" and "queue drained".
class WorkerPool {
public:
    explicit WorkerPool(int n) {
        if (n < 1) n = 1;
        workers_.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Enqueue a task that will be invoked with the running thread's worker_id.
    void enqueue(std::function<void(int)> task) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push(std::move(task));
        }
        cv_work_.notify_one();
    }

    // Block until queue is empty AND no tasks are in flight.
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_idle_.wait(lk, [this] {
            return tasks_.empty() && in_flight_ == 0;
        });
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    void worker_loop(int worker_id) {
        for (;;) {
            std::function<void(int)> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
                ++in_flight_;
            }
            try {
                task(worker_id);
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("Task failed: ") + e.what());
            } catch (...) {
                LOG_ERROR("Task failed with an unknown exception");
            }
            bool drained = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                --in_flight_;
                if (tasks_.empty() && in_flight_ == 0) drained = true;
            }
            if (drained) cv_idle_.notify_all();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void(int)>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_idle_;
    bool stop_ = false;
    int  in_flight_ = 0;
};

bool run_directory(
    YOLOv8& /*detector_unused*/,   // kept for API parity; --workers uses its own pool
    const CLIArgs& args,
    const std::vector<std::string>& class_names
) {
    fs::path dir(args.dir);
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        LOG_ERROR("Directory not found: " + args.dir);
        return false;
    }

    // Collect + sort image paths for deterministic processing order.
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        if (is_image_extension(entry.path())) {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        LOG_ERROR("No images (.jpg/.jpeg/.png/.bmp) found in: " + args.dir);
        return false;
    }

    fs::path out_dir = args.output.empty() ? fs::path("results") : fs::path(args.output);
    if (!fs::exists(out_dir, ec)) {
        fs::create_directories(out_dir, ec);
    }

    // Resolve worker count. CLI 0 = default = hardware_concurrency().
    int n_workers = args.workers;
    if (n_workers <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        n_workers = hw > 0 ? static_cast<int>(hw) : 1;
    }
    n_workers = std::max(1, n_workers);

    LOG_INFO("=== Directory batch ===");
    LOG_INFO("  source dir  : " + args.dir);
    LOG_INFO("  images      : " + std::to_string(image_paths.size()));
    LOG_INFO("  output dir  : " + out_dir.string());
    LOG_INFO("  workers     : " + std::to_string(n_workers));

    // Each worker needs its OWN YOLOv8 instance because the
    // Preprocessor owns a pre-allocated NCHW buffer that would race
    // if shared. Cost: N copies of the ORT session in RSS.
    // The pool pins one detector per worker THREAD (worker_id == thread
    // index 0..N-1 == worker_detectors index) — see WorkerPool's comment.
    std::vector<std::unique_ptr<YOLOv8>> worker_detectors;
    worker_detectors.reserve(static_cast<size_t>(n_workers));
    for (int i = 0; i < n_workers; ++i) {
        worker_detectors.push_back(
            std::make_unique<YOLOv8>(args.model, class_names, args.imgsz)
        );
    }

    // Warmup using worker 0 — first Run() pays for graph opt + thread
    // pool spin-up. The other workers also benefit (ORT pages in the
    // model weights, which are shared via the OS page cache).
    {
        cv::Mat warmup_img = cv::imread(image_paths.front().string());
        if (!warmup_img.empty()) {
            for (int i = 0; i < 2; ++i) {
                worker_detectors[0]->detect(warmup_img, args.conf, args.iou);
            }
        }
    }

    WorkerPool pool(n_workers);
    std::mutex rows_mu;
    std::vector<BenchmarkRow> rows;
    rows.reserve(image_paths.size());
    std::atomic<int> total_detections{0};

    // Enqueue one task per image. The task receives the owning thread's worker_id
    // (NOT i % n_workers) so each detector is only touched by one thread.
    for (size_t i = 0; i < image_paths.size(); ++i) {
        pool.enqueue([
            i, &image_paths, &worker_detectors,
            &out_dir, &args, &rows_mu, &rows, &total_detections
        ](int worker_id) {
            const fs::path& path = image_paths[i];
            cv::Mat img = cv::imread(path.string());
            if (img.empty()) {
                LOG_WARN("  skip (unreadable) : " + path.filename().string());
                return;
            }

            YOLOv8& d = *worker_detectors[worker_id];
            auto t0 = std::chrono::high_resolution_clock::now();
            auto detections = d.detect(img, args.conf, args.iou);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double total_ms = std::chrono::duration<double, std::milli>(
                t1 - t0
            ).count();
            const auto& p = d.profile();

            // Annotated output: <out_dir>/<basename>.jpg
            fs::path out_path = out_dir / path.filename();
            cv::Mat annotated = img.clone();
            drawDetections(annotated, detections);
            if (!cv::imwrite(out_path.string(), annotated)) {
                LOG_WARN("  failed to write  : " + out_path.string());
            }

            BenchmarkRow row;
            row.index          = static_cast<int>(i);
            row.image          = path.filename().string();
            row.preprocess_ms  = p.preprocess_ms;
            row.infer_ms       = p.infer_ms;
            row.postprocess_ms = p.postprocess_ms;
            row.total_ms       = total_ms;
            row.rss_mb         = Benchmark::rss_mb();

            {
                std::lock_guard<std::mutex> lk(rows_mu);
                rows.push_back(row);
            }
            total_detections.fetch_add(static_cast<int>(detections.size()));

            LOG_INFO("  [" + std::to_string(i + 1) + "/"
                     + std::to_string(image_paths.size()) + "] "
                     + path.filename().string()
                     + "  det=" + std::to_string(detections.size())
                     + "  t=" + std::to_string(total_ms) + " ms");
        });
    }

    pool.wait_idle();

    BenchmarkSummary summary = Benchmark::summarize(rows);
    LOG_INFO("=== Summary ===");
    LOG_INFO("  processed   : " + std::to_string(summary.count));
    LOG_INFO("  total dets  : " + std::to_string(total_detections.load()));
    LOG_INFO("  mean        : " + std::to_string(summary.mean_ms) + " ms");
    LOG_INFO("  p50         : " + std::to_string(summary.p50_ms) + " ms");
    LOG_INFO("  p95         : " + std::to_string(summary.p95_ms)  + " ms");
    LOG_INFO("  throughput  : " + std::to_string(summary.throughput_fps) + " FPS");
    LOG_INFO("  peak RSS    : " + std::to_string(summary.peak_rss_mb) + " MB");

    Benchmark::write_csv(out_dir.string() + "/benchmark.csv", rows);
    if (!args.save_json.empty()) {
        if (!ensure_parent_dir(args.save_json)) return false;
        Benchmark::write_json(args.save_json, summary, args.model);
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Force UTF-8 on Windows so non-ASCII paths don't get mangled by
        // std::filesystem::path::string() round-tripping through ACP.
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif

        CLIArgs args = parse_cli(argc, argv);

        if (args.help) {
            print_help(argv[0]);
            return 0;
        }
        if (!args.valid) {
            std::cerr << "ERROR: " << args.error << "\n\n";
            print_help(argv[0]);
            return 1;
        }

        // Fill in defaults from AppConfig for fields the user didn't set.
        AppConfig defaults;
        if (args.model.empty())   args.model   = defaults.model_path;
        if (args.yaml_path.empty()) args.yaml_path = defaults.yaml_path;
        if (args.output.empty())  args.output  = defaults.output_path;

        if (!fs::exists(args.model)) {
            LOG_ERROR("Model not found: " + args.model);
            return 1;
        }

        // Load class names from --yaml (with hardcoded fallback on failure).
        std::vector<std::string> class_names = load_or_fallback(args.yaml_path);

        LOG_INFO("========== YOLOv8s ONNX Runtime C++ ==========");
        LOG_INFO("  model    : " + args.model);
        LOG_INFO("  imgsz    : " + std::to_string(args.imgsz));
        LOG_INFO("  conf     : " + std::to_string(args.conf));
        LOG_INFO("  iou      : " + std::to_string(args.iou));
        LOG_INFO("  threads  : intra=" + std::to_string(args.intra_op_threads)
                 + " inter=" + std::to_string(args.inter_op_threads));

        YOLOv8 detector(args.model, class_names, args.imgsz,
                        args.intra_op_threads, args.inter_op_threads);

        bool ok = true;
        // --benchmark is a modifier checked FIRST: it can combine with --image or
        // --dir (or fall back to the default sample), so don't let run_single_image
        // grab it when an image source is also present.
        if (args.benchmark > 0) {
            ok = run_benchmark(detector, args);
        } else if (!args.dir.empty()) {
            ok = run_directory(detector, args, class_names);
        } else if (!args.image.empty()) {
            ok = run_single_image(detector, args, class_names);
        } else {
            LOG_ERROR("No input source: provide --image, --dir, or --benchmark N.");
            ok = false;
        }

        return ok ? 0 : 1;
    }
    catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime Error: " + std::string(e.what()));
    }
    catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV Error: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        LOG_ERROR("Exception: " + std::string(e.what()));
    }
    catch (...) {
        LOG_ERROR("Unknown fatal error");
    }
    return 1;
}
