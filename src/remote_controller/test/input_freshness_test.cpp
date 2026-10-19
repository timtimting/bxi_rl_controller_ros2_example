#include "remote_controller/input_freshness.hpp"
#include "remote_controller/hidraw_observer.hpp"
#include <cassert>
#include <iostream>
#include <sys/stat.h>

int main()
{
    using namespace remote_controller;
    constexpr std::int64_t now = 2000000000LL;
    InputFreshness s;
    assert(freshness_status(s, now) == "no_source");
    s.source = "usb"; s.mode = "periodic"; s.open = true;
    s.reader_ns = now; s.io_ns = now; s.sample_ns = now;
    assert(freshness_status(s, now) == "fresh");
    assert(!freshness_requires_clear(s, now));
    s.sample_ns = now - 301000000LL;
    assert(freshness_status(s, now) == "protocol_stale");
    assert(freshness_requires_clear(s, now));
    s.io_ns = s.sample_ns;
    assert(freshness_status(s, now) == "usb_stale");
    s.reader_ns = s.sample_ns;
    assert(freshness_status(s, now) == "reader_stalled");
    s.mode = "event"; s.reader_ns = now;
    assert(freshness_status(s, now) == "event_source_unverified");
    assert(!freshness_requires_clear(s, now));
    s.open = false;
    assert(freshness_status(s, now) == "device_closed");
    assert(freshness_requires_clear(s, now));
    s.stop_status = "fresh";
    assert(freshness_status(s, now) == "device_closed");
    s.stop_status = "protocol_stale";
    assert(freshness_status(s, now) == "device_closed");
    assert(freshness_quote("a\nb\t\"")=="\"a\\u000ab\\u0009\\\"\"");
    InputFreshnessTracker tracker;
    tracker.reader_tick(); tracker.read_bytes(26); tracker.valid_sample();
    tracker.read_bytes(26); tracker.valid_sample(); // Identical values still mean new frames.
    const auto snapshot = tracker.snapshot();
    assert(snapshot.samples == 2 && snapshot.bytes == 52 && snapshot.reads == 2);
    tracker.read_bytes(10); tracker.crc_errors(1); // Garbage must not refresh valid-frame time/count.
    const auto garbage = tracker.snapshot();
    assert(garbage.samples == 2 && garbage.sample_ns == snapshot.sample_ns && garbage.crc_errors == 1);
    assert(freshness_json(s, now, 1, 2, now, false, 3).find("\"sample_seq\":") != std::string::npos);
    char fifo[]="/tmp/remote-usb-freshness-XXXXXX";
    const int temporary=mkstemp(fifo); assert(temporary>=0); close(temporary); unlink(fifo);
    assert(mkfifo(fifo,0600)==0);
    HidrawObserver observer; assert(observer.open_device("",fifo));
    const int writer=open(fifo,O_WRONLY|O_NONBLOCK); assert(writer>=0);
    assert(write(writer,"report",6)==6); observer.poll();
    const auto report=observer.snapshot(); assert(report.samples==1 && report.bytes==6);
    observer.poll(); assert(observer.snapshot().samples==1); // Idle cannot create a fresh report.
    close(writer); observer.poll(); assert(!observer.is_open()); unlink(fifo);
    std::cout << "PASS: USB arrival, valid-frame freshness, reader stall, event-only idle and counters\n";
}
