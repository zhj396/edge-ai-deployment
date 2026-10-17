#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

class Logger {
public:
    enum class Level {
        INFO,
        WARNING,
        ERROR
    };

    static void log(Level level, const std::string& msg)
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;

#ifdef _WIN32
        localtime_s(&tm_buf, &time);
#else
        localtime_r(&time, &tm_buf);
#endif

        std::string prefix;
        switch (level) {
            case Level::INFO:
                prefix = "[INFO]";
                break;
            case Level::WARNING:
                prefix = "[WARN]";
                break;
            case Level::ERROR:
                prefix = "[ERROR]";
                break;
        }

        // Build the full line first, then emit it under a mutex: --workers runs
        // log calls concurrently on std::cout, and a single locked write keeps
        // each line intact instead of interleaving mid-line.
        std::ostringstream line;
        line << prefix
             << " "
             << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
             << " "
             << msg
             << "\n";

        static std::mutex out_mutex;
        {
            std::lock_guard<std::mutex> lock(out_mutex);
            std::cout << line.str() << std::flush;
        }
    }
};

#define LOG_INFO(msg) Logger::log(Logger::Level::INFO, msg)
#define LOG_WARN(msg) Logger::log(Logger::Level::WARNING, msg)
#define LOG_ERROR(msg) Logger::log(Logger::Level::ERROR, msg)
