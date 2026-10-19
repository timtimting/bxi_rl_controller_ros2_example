#include "remote_controller/drivers/crsf_frame_parser.hpp"
#include "remote_controller/input_device_manager.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <pty.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

int main()
{
    using namespace remote_controller;
    int master = -1, slave = -1;
    char path[128] = {};
    assert(openpty(&master, &slave, path, nullptr, nullptr) == 0);

    auto config = load_remote_config(REMOTE_CONTROLLER_TEST_CONFIG_PATH);
    config.input_selection.scan_interval_ms = 10;
    config.input_selection.promote_stable_ms = 0;
    for (auto &device : config.input_devices) {
        if (device.type == "crsf") device.device = path;
    }
    InputMapper mapper(config);
    std::mutex mapper_lock;
    std::vector<std::string> logs;
    InputDeviceManager manager(config, mapper, mapper_lock, [](const std::vector<std::string> &) {},
        [&logs](const std::string &message) { logs.push_back(message); }, false, "crsf");

    std::vector<std::uint8_t> frame(26, 0);
    frame[0] = 0xc8; frame[1] = 24; frame[2] = 0x16;
    frame.back() = CrsfFrameParser::crc8_dvb_s2(frame.data() + 2, 23);
    bool active = false;
    for (int i = 0; i < 150 && !active; ++i) {
        assert(write(master, frame.data(), frame.size()) == static_cast<ssize_t>(frame.size()));
        if (manager.tick()) manager.notify_safe_output_published();
        active = manager.input_freshness().active;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!active) {
        for (const auto &line : logs) std::cerr << line << '\n';
    }
    assert(active);
    {
        const std::lock_guard<std::mutex> guard(mapper_lock);
        mapper.set_signal("crsf.xbox.right_y", 0.8);
        communication::msg::MotionCommands command;
        mapper.fill_message(command);
        if (std::fabs(command.vel_des.x) <= 0.001) {
            std::cerr << "vx after source set=" << command.vel_des.x << '\n';
            for (const auto &line : logs) std::cerr << line << '\n';
        }
        assert(std::fabs(command.vel_des.x) > 0.001);
    }

    frame.back() ^= 1;
    bool safe_barrier = false;
    for (int i = 0; i < 100 && !safe_barrier; ++i) {
        assert(write(master, frame.data(), frame.size()) == static_cast<ssize_t>(frame.size()));
        safe_barrier = manager.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    assert(safe_barrier);
    {
        const std::lock_guard<std::mutex> guard(mapper_lock);
        communication::msg::MotionCommands command;
        mapper.fill_message(command);
        assert(command.vel_des.x == 0.0);
        assert(command.vel_des.y == 0.0);
        assert(command.yawdot_des == 0.0);
    }
    bool found_reason = false;
    for (const auto &line : logs) {
        if (line.find("input freshness protocol_stale") != std::string::npos) found_reason = true;
    }
    assert(found_reason);
    manager.notify_safe_output_published();
    manager.stop();
    close(master); close(slave);
}
