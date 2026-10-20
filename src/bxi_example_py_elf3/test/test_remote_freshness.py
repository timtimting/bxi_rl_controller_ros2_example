import json

from bxi_example_py_elf3.framework.platform.input_safety.remote_freshness import RemotePathMonitor


def frame(now=1.0, **overrides):
    value = dict(schema=1, stamp_ns=int(now * 1e9), instance=1, source="crsf", device="/dev/ttyCRSF",
                 mode="periodic", status="fresh", usb_read_age_ms=1, valid_frame_age_ms=1,
                 reader_age_ms=1, publish_age_ms=1, timeout_ms=300, sample_seq=1, published=1,
                 publish_on_change=False)
    value.update(overrides)
    return json.dumps(value)


def observe(m, now=1.0, **overrides):
    return m.observe(frame(now, **overrides), now, int(now * 1e9))


def test_usb_protocol_publish_and_demo_failures_are_distinct():
    m = RemotePathMonitor(0)
    assert observe(m)
    assert m.status(1, 1) == "fresh"
    assert m.status(1, 600) == "demo_receive_stale"
    assert observe(m, 1.1, publish_age_ms=600, sample_seq=2)
    assert m.status(1.1, 1) == "publisher_stale"
    assert observe(m, 1.2, status="protocol_stale", sample_seq=3)
    assert m.status(1.2, 1) == "protocol_stale"
    assert observe(m, 1.3, status="usb_stale", sample_seq=4)
    assert m.status(1.3, 1) == "usb_stale"
    assert m.status(2, 1) == "status_link_stale"


def test_republished_snapshot_does_not_count_as_new_source_data():
    m = RemotePathMonitor(0)
    assert observe(m)
    assert observe(m, 1.1)
    assert observe(m, 1.4)
    assert m.status(1.4, 1) == "valid_frames_not_advancing"
    assert observe(m, 1.5, sample_seq=2)
    assert m.status(1.5, 1) == "fresh"


def test_invalid_and_old_status_do_not_refresh_receive_age():
    m = RemotePathMonitor(0)
    assert observe(m)
    assert not m.observe(frame(), 2, 2_000_000_000)
    assert m.received_at == 1
    assert m.status(2, 1) == "invalid_status"
    assert not observe(m, 2, usb_read_age_ms=float("nan"))
    assert not m.observe("{}", 2, 2_000_000_000)
    assert not observe(m, 2, status="unknown_future_status")
    assert m.input_unsafe(2, 1)


def test_logs_are_sparse_and_recovery_is_immediate():
    m = RemotePathMonitor(0)
    assert observe(m)
    assert m.report(1, 1)[0] == "info"
    assert m.report(1.1, 1) is None
    assert m.report(2, 1)[0] == "warning"
    assert m.report(2.1, 1) is None
    assert m.report(7, 1)[0] == "warning"
    assert observe(m, 7.1, sample_seq=2)
    assert m.report(7.1, 1)[0] == "info"


def test_idle_event_sources_and_on_change_publishers_are_not_usb_faults():
    m = RemotePathMonitor(0)
    assert observe(m, mode="event", status="event_source_unverified", sample_seq=0)
    assert m.status(1, 1) == "event_source_unverified"
    assert observe(m, 2, publish_on_change=True, publish_age_ms=5000, sample_seq=2)
    assert m.status(2, None) == "on_change_idle_or_fresh"
    empty = RemotePathMonitor(0)
    assert empty.status(1, None) == "waiting_status"
    assert empty.status(4, None) == "status_unavailable"


def test_missing_freshness_status_warns_once_without_placeholder_fields():
    monitor = RemotePathMonitor(0)
    severity, text = monitor.report(4, 400)
    assert severity == "warning"
    assert text == (
        "remote_path status=status_unavailable; "
        "no remote_controller/freshness data received"
    )
    assert monitor.report(10, 10000) is None
