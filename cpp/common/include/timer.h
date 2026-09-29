#pragma once

#include <chrono>

// Monotonic wall-clock stopwatch with split-lap support. All time units
// are explicit (Ms / Us suffix) — there is no implicit conversion.
class Timer {
public:
    void start() {
        t0_ = Clock::now();
        last_ = t0_;
    }

    double elapsedMs() const {
        return elapsedSince(t0_);
    }

    double elapsedUs() const {
        return elapsedSinceUs(t0_);
    }

    // Time since start(); updates the split reference.
    double lapMs() {
        const auto now = Clock::now();
        last_ = now;
        return elapsedSince(t0_);
    }

    double lapUs() {
        const auto now = Clock::now();
        last_ = now;
        return elapsedSinceUs(t0_);
    }

    // Time since the previous lap() (or start() if no lap yet).
    double splitMs() {
        const auto now = Clock::now();
        const double dt = elapsedSince(last_);
        last_ = now;
        return dt;
    }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsedSince(Clock::time_point t0) {
        const auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }

    static double elapsedSinceUs(Clock::time_point t0) {
        const auto now = Clock::now();
        return std::chrono::duration<double, std::micro>(now - t0).count();
    }

    Clock::time_point t0_;
    Clock::time_point last_;
};
