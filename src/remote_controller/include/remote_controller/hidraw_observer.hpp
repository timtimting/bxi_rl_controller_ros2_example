#pragma once

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <glob.h>
#include <limits.h>
#include <string>
#include <unistd.h>

#include "remote_controller/input_freshness.hpp"

namespace remote_controller {

// A separate hidraw open has its own read queue. This observer neither grabs
// the device nor consumes the joystick driver's event queue.
class HidrawObserver {
public:
    ~HidrawObserver() { close_device(); }
    bool open_device(const std::string &joystick, const std::string &configured = "") {
        close_device();
        path_ = configured;
        if (path_.empty()) {
            char resolved[PATH_MAX];
            if (!realpath(joystick.c_str(), resolved)) return false;
            const std::string canonical(resolved);
            const auto name = canonical.substr(canonical.find_last_of('/') + 1);
            if (!realpath(("/sys/class/input/" + name + "/device").c_str(), resolved)) return false;
            std::string parent(resolved);
            for (int level = 0; level < 4 && path_.empty(); ++level) {
                glob_t matches{};
                if (glob((parent + "/hidraw/hidraw*").c_str(), 0, nullptr, &matches) == 0 && matches.gl_pathc == 1) {
                    const std::string item(matches.gl_pathv[0]);
                    path_ = "/dev/" + item.substr(item.find_last_of('/') + 1);
                }
                globfree(&matches);
                parent = parent.substr(0, parent.find_last_of('/'));
            }
        }
        if (path_.empty()) return false;
        fd_ = open(path_.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd_ >= 0) tracker_.reset();
        return fd_ >= 0;
    }
    void poll() {
        if (fd_ < 0) return;
        unsigned char bytes[4096];
        // Bound diagnostic work even if a broken device floods reports.
        for (int i = 0; i < 64; ++i) {
            const auto count = read(fd_, bytes, sizeof(bytes));
            if (count > 0) { tracker_.read_bytes(static_cast<std::size_t>(count)); tracker_.valid_sample(); }
            else { if (count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) close_device(); break; }
        }
    }
    void close_device() { if (fd_ >= 0) close(fd_); fd_ = -1; }
    bool is_open() const { return fd_ >= 0; }
    const std::string &path() const { return path_; }
    InputFreshness snapshot() const { return tracker_.snapshot(); }
private:
    int fd_ = -1;
    std::string path_;
    InputFreshnessTracker tracker_;
};

}  // namespace remote_controller
