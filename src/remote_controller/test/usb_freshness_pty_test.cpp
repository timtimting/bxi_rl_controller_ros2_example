#include "remote_controller/drivers/driver_registry.hpp"
#include "remote_controller/drivers/crsf_frame_parser.hpp"
#include <cassert>
#include <chrono>
#include <mutex>
#include <pty.h>
#include <thread>
#include <unistd.h>
#include <vector>

int main()
{
    using namespace remote_controller;
    int master = -1, slave = -1; char path[128] = {};
    const int opened = openpty(&master, &slave, path, nullptr, nullptr);
    assert(opened == 0);
    RemoteConfig config;
    InputMapper mapper(config); std::mutex lock;
    InputDeviceConfig device;
    device.name="test_usb"; device.type="crsf"; device.device=path; device.loss_timeout_ms=300;
    auto driver=create_input_driver(device, mapper, lock, [](const std::vector<std::string>&){}, [](const std::string&){});
    assert(driver); driver->start();
    std::vector<std::uint8_t> frame(26,0); frame[0]=0xc8; frame[1]=24; frame[2]=0x16;
    frame.back()=CrsfFrameParser::crc8_dvb_s2(frame.data()+2,23);
    for (int i=0;i<20;++i) {
        assert(write(master,frame.data(),frame.size())==static_cast<ssize_t>(frame.size()));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto before=driver->freshness();
    assert(before.samples>5 && before.bytes>0);
    assert(freshness_status(before,freshness_now_ns())=="fresh");
    frame.back()^=1;
    for (int i=0;i<25;++i) {
        assert(write(master,frame.data(),frame.size())==static_cast<ssize_t>(frame.size()));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    auto garbage=driver->freshness();
    assert(garbage.samples==before.samples && garbage.crc_errors>0 && garbage.bytes>before.bytes);
    assert(freshness_status(garbage,freshness_now_ns())=="protocol_stale");
    std::this_thread::sleep_for(std::chrono::milliseconds(450));
    assert(freshness_status(driver->freshness(),freshness_now_ns())=="usb_stale");
    driver->stop(); close(master); close(slave);
}
