#pragma once

#include <chrono>

// Monotonic wall-clock stopwatch. Time units are explicit (the Ms suffix of
// elapsedMs) — there is no implicit conversion.
class Timer {
public:
    void start() {
        t0_ = Clock::now();
    }

    double elapsedMs() const {
        return elapsedSince(t0_);
    }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsedSince(Clock::time_point t0) {
        const auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }

    Clock::time_point t0_;
};
