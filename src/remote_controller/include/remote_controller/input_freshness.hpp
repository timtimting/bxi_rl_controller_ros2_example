#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace remote_controller {

inline std::int64_t freshness_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct InputFreshness {
    std::string source = "none", device, mode = "event", io_kind = "event", stop_status;
    bool open = false, active = false;
    int timeout_ms = 300;
    std::uint64_t reads = 0, bytes = 0, samples = 0, crc_errors = 0;
    std::int64_t reader_ns = 0, io_ns = 0, sample_ns = 0;
};

inline double freshness_age_ms(std::int64_t timestamp, std::int64_t now)
{
    return timestamp > 0 ? static_cast<double>(now - timestamp) / 1e6 : -1.0;
}

inline std::string freshness_status(const InputFreshness &s, std::int64_t now)
{
    if (s.source == "none") return "no_source";
    if (!s.open) return "device_closed";
    if (s.reader_ns && freshness_age_ms(s.reader_ns, now) > s.timeout_ms) return "reader_stalled";
    if (s.mode != "periodic") return "event_source_unverified";
    if (!s.sample_ns) return s.io_ns ? "waiting_valid_frame" : "waiting_usb";
    if (freshness_age_ms(s.sample_ns, now) <= s.timeout_ms) return "fresh";
    return s.io_ns && freshness_age_ms(s.io_ns, now) <= s.timeout_ms ? "protocol_stale" : "usb_stale";
}

inline bool freshness_requires_clear(const InputFreshness &s, std::int64_t now)
{
    const auto status = freshness_status(s, now);
    return status == "reader_stalled" || status == "protocol_stale" ||
        status == "usb_stale" || status == "device_closed" ||
        (s.mode == "periodic" &&
         (status == "waiting_usb" || status == "waiting_valid_frame"));
}

class InputFreshnessTracker {
public:
    void reset() {
        reader_ns_ = 0; io_ns_ = 0; sample_ns_ = 0;
        reads_ = 0; bytes_ = 0; samples_ = 0; crc_errors_ = 0;
    }
    void reader_tick() { reader_ns_ = freshness_now_ns(); }
    void read_bytes(std::size_t size) {
        io_ns_ = freshness_now_ns(); ++reads_; bytes_ += size;
    }
    void valid_sample() { sample_ns_ = freshness_now_ns(); ++samples_; }
    void crc_errors(std::uint64_t errors) { crc_errors_ += errors; }
    InputFreshness snapshot() const {
        InputFreshness s;
        s.reader_ns = reader_ns_; s.io_ns = io_ns_; s.sample_ns = sample_ns_;
        s.reads = reads_; s.bytes = bytes_; s.samples = samples_; s.crc_errors = crc_errors_;
        return s;
    }
private:
    std::atomic<std::int64_t> reader_ns_{0}, io_ns_{0}, sample_ns_{0};
    std::atomic<std::uint64_t> reads_{0}, bytes_{0}, samples_{0}, crc_errors_{0};
};

inline std::string freshness_quote(const std::string &text)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : text) {
        if (ch == '"' || ch == '\\') out << '\\' << static_cast<char>(ch);
        else if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(ch);
        else out << static_cast<char>(ch);
    }
    out << '"';
    return out.str();
}

inline std::string freshness_json(const InputFreshness &s, std::int64_t now, std::int64_t stamp,
    std::uint64_t published, std::int64_t last_publish, bool on_change, std::int64_t instance,
    double sample_hz = -1, double read_hz = -1, std::uint64_t timer_gaps = 0,
    std::size_t subscribers = 0)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\"schema\":1,\"stamp_ns\":" << stamp << ",\"instance\":" << instance
        << ",\"source\":" << freshness_quote(s.source) << ",\"device\":" << freshness_quote(s.device)
        << ",\"mode\":" << freshness_quote(s.mode) << ",\"status\":" << freshness_quote(freshness_status(s, now))
        << ",\"io_kind\":" << freshness_quote(s.io_kind)
        << ",\"last_stop_status\":" << freshness_quote(s.stop_status)
        << ",\"observability\":\"host_arrival_only\""
        << ",\"active\":" << (s.active ? "true" : "false") << ",\"open\":" << (s.open ? "true" : "false")
        << ",\"timeout_ms\":" << s.timeout_ms << ",\"reader_age_ms\":" << freshness_age_ms(s.reader_ns, now)
        << ",\"usb_read_age_ms\":" << freshness_age_ms(s.io_ns, now)
        << ",\"valid_frame_age_ms\":" << freshness_age_ms(s.sample_ns, now)
        << ",\"reads\":" << s.reads << ",\"bytes\":" << s.bytes << ",\"sample_seq\":" << s.samples
        << ",\"sample_hz\":" << sample_hz << ",\"read_hz\":" << read_hz
        << ",\"crc_errors\":" << s.crc_errors << ",\"published\":" << published
        << ",\"publish_age_ms\":" << freshness_age_ms(last_publish, now)
        << ",\"timer_gaps\":" << timer_gaps << ",\"matched_subscribers\":" << subscribers
        << ",\"publish_on_change\":" << (on_change ? "true" : "false") << '}';
    return out.str();
}

}  // namespace remote_controller
