/*!
 * @file logger.hpp
 *
 * @brief Logger header file
 */

#pragma once

#include <cstdint>
#include <chrono>
#include <string>

namespace racon {

static const std::string version = "v1.0.0";

class Logger {
public:
    Logger(): time_(0.), bar_(0), time_point_() {
}

    Logger(const Logger&) = default;
    Logger& operator=(const Logger&) = default;

    Logger(Logger&&) = default;
    Logger& operator=(Logger&&) = default;

    ~Logger() {}

    /*!
     * @brief Resets the time point
     */
    void log() {
        auto now = std::chrono::steady_clock::now();
    if (time_point_ != std::chrono::time_point<std::chrono::steady_clock>()) {
        time_ += std::chrono::duration_cast<std::chrono::duration<double>>(now - time_point_).count();
    }
    time_point_ = now;
    }

    /*!
     * @brief Prints the elapsed time from last time point to stderr
     */
    void log(const std::string& msg) const {
    std::cerr << msg << " " << std::fixed
        << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time_point_).count()
        << " s" << std::endl;
}

    /*!
     * @brief Prints a progress bar and the elapsed time from last time to
     * stderr (the progress bar resets after 20 calls)
     */
    void bar(const std::string& msg) {
    ++bar_;
    std::string progress_bar = "[" + std::string(bar_, '=') + (bar_ == 20 ? "" : ">" + std::string(19 - bar_, ' ')) + "]";

    std::cerr << msg << " " << progress_bar << " " << std::fixed
        << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time_point_).count()
        << " s";

    bar_ %= 20;
    if (bar_ == 0) {
        std::cerr << std::endl;
    } else {
        std::cerr << "\r";
    }
}

    /*!
     * @brief Prints the total elapsed time from the first log() call
     */
    void total(const std::string& msg) const {
    std::cerr << msg << " " << std::fixed
        << time_ + std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time_point_).count()
        << " s" << std::endl;
}

private:
    double time_;
    std::uint32_t bar_;
    std::chrono::time_point<std::chrono::steady_clock> time_point_;
};

}
