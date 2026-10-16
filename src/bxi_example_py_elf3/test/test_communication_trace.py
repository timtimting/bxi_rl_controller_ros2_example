"""Fault injection for communication recording, without hardware or DDS sockets."""

import asyncio
import json
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from bxi_example_py_elf3.framework.platform.communication_trace import (
    CommunicationExecutor, CommunicationTrace,
)


@pytest.fixture
def recorder():
    trace = CommunicationTrace()
    trace._log = Mock()
    clock = [10_000_000_000, 110_000_000_000]
    with patch("time.monotonic_ns", side_effect=lambda: clock[0]), patch(
        "time.time_ns", side_effect=lambda: clock[1]
    ):
        yield trace, clock


def report(trace):
    return reports(trace)[-1]


def reports(trace):
    return [json.loads(call.args[1].split(" ", 1)[1])
            for call in trace._log.call_args_list
            if call.args[1].startswith("communication_trace ")]


def summaries(trace):
    return [call.args for call in trace._log.call_args_list
            if not call.args[1].startswith("communication_trace ")]


def advance(clock, delta):
    clock[0] += delta
    clock[1] += delta


def test_healthy_periodic_traffic_does_not_log(recorder):
    trace, clock = recorder
    for _ in range(100):
        trace.record("joint_rx")
        trace.record("command_tx")
        trace.enter("waiting_next_cycle")
        trace.poll()
        advance(clock, 20_000_000)
    trace._log.assert_not_called()


def test_inflight_stall_is_visible_before_control_returns_and_then_recovers(recorder):
    trace, clock = recorder
    trace.record("command_tx", stamp_ns=42)
    trace.enter("framework_lock")
    trace.poll()
    advance(clock, 250_000_000)
    trace.poll()
    data = report(trace)
    assert data["event"] == "stall"
    assert data["active"]["control"]["stage"] == "framework_lock"
    assert data["streams"]["command_tx"]["stamp_ns"] == 42
    assert data["stacks"]["control"]
    trace.record("command_tx", stamp_ns=43)
    trace.enter("waiting_next_cycle")
    advance(clock, 50_000_000)
    trace.record("command_tx", stamp_ns=44)
    trace.enter("waiting_next_cycle")
    trace.poll()
    assert len(reports(trace)) == 1
    severity, text = summaries(trace)[-1]
    assert severity == "info"
    assert "恢复" in text
    assert "stamp_ns=42→43" in text


def test_transient_gap_retains_old_and_new_command_stamps(recorder):
    trace, clock = recorder
    trace.record("command_tx", stamp_ns=123)
    advance(clock, 1_200_000_000)
    trace.record("command_tx", stamp_ns=456)
    trace.poll()
    data = report(trace)
    assert data["event"] == "incident"
    assert data["incidents"][0] == {
        "kind": "stream_gap", "stream": "command_tx", "mono_ns": clock[0],
        "gap_ms": 1200, "previous_stamp_ns": 123, "stamp_ns": 456,
    }


def test_dds_receive_wait_and_transport_estimates_are_separate(recorder):
    trace, clock = recorder
    trace.record("joint_rx", stamp_ns=clock[1] - 800_000_000,
                 ros_now_ns=clock[1], info={
                     "source_timestamp": clock[1] - 800_000_000,
                     "received_timestamp": clock[1] - 600_000_000,
                 })
    trace.poll()
    sample = report(trace)["streams"]["joint_rx"]
    assert sample["source_age_ms"] == 800
    assert sample["dds_to_callback_ms"] == 600
    assert sample["source_to_dds_ms"] == 200


def test_unavailable_dds_timestamps_are_unknown_not_zero_latency(recorder):
    trace, clock = recorder
    trace.record("joint_rx", info={"source_timestamp": 0, "received_timestamp": 0})
    advance(clock, 250_000_000)
    trace.poll()
    sample = report(trace)["streams"]["joint_rx"]
    assert sample["dds_to_callback_ms"] is None
    assert sample["source_to_dds_ms"] is None


def test_wall_clock_jump_is_distinct_from_monotonic_stream_gap(recorder):
    trace, clock = recorder
    trace.record("command_tx")
    trace.poll()
    advance(clock, 20_000_000)
    clock[1] += 2_000_000_000
    trace.record("command_tx")
    trace.poll()
    data = report(trace)
    assert data["stalled_streams"] == []
    assert data["incidents"] == [{"kind": "wall_clock_jump", "delta_ms": 2000}]


def test_recording_and_repeated_fault_logs_are_bounded(recorder):
    trace, clock = recorder
    for _ in range(1000):
        advance(clock, 250_000_000)
        trace.record("joint_rx", stamp_ns=1, ros_now_ns=clock[1])
    assert len(trace._incidents) == 32
    advance(clock, 250_000_000)
    trace.poll()
    for _ in range(10):
        advance(clock, 50_000_000)
        trace.poll()
    assert len(reports(trace)) == 1
    assert len(summaries(trace)) == 1
    assert len(trace._history) == 10
    assert len(trace._pending) <= 32


def test_idle_remote_repeats_short_summary_without_requerying_graph(recorder):
    trace, clock = recorder
    trace._context = Mock(return_value={"state": "com.bxi.basic_actions/zero_torque"})
    trace._summary_context = Mock(return_value={"state": "com.bxi.basic_actions/zero_torque",
                                               "remote_input_fresh": False})
    for _ in range(16):
        trace.record("remote_rx")
    for _ in range(326):
        trace.record("command_tx")
        for _ in range(18):
            trace.record("joint_rx")
        for _ in range(10):
            trace.record("imu_rx")
        trace.enter("waiting_next_cycle")
        trace.poll()
        advance(clock, 20_000_000)
    assert len(reports(trace)) == 1
    assert len(summaries(trace)) == 2  # First warning, then five seconds later.
    trace._context.assert_called_once()
    text = summaries(trace)[-1][1]
    assert "遥控输入停更5.22s" in text
    assert "电机命令=50.0Hz" in text
    assert "关节反馈=900.0Hz" in text
    assert "IMU=500.0Hz" in text
    assert "遥控输入=0.0Hz" in text
    assert "状态=zero_torque" in text
    assert "遥控有效=否" in text
    assert report(trace)["stalled_streams"] == ["remote_rx"]


def test_new_control_stall_bypasses_remote_summary_throttle(recorder):
    trace, clock = recorder
    trace.record("remote_rx")
    for _ in range(7):
        trace.record("command_tx")
        trace.enter("waiting_next_cycle")
        trace.poll()
        advance(clock, 50_000_000)
    assert len(reports(trace)) == 1
    trace.record("command_tx")
    trace.enter("framework_lock")
    for _ in range(6):
        trace.poll()
        advance(clock, 50_000_000)
    assert len(reports(trace)) == 2
    assert report(trace)["diagnostic_id"] == 2
    assert report(trace)["blocked_lanes"] == ["control"]
    assert report(trace)["stacks"]["control"]
    trace.record("command_tx")
    trace.enter("waiting_next_cycle")
    trace.poll()
    assert len(reports(trace)) == 2  # Partial recovery needs only a short line.
    assert "遥控输入停更" in summaries(trace)[-1][1]
    assert "卡在" not in summaries(trace)[-1][1]


def test_recovery_rearms_detail_for_next_outage(recorder):
    trace, clock = recorder
    trace.record("remote_rx", stamp_ns=1)
    for _ in range(7):
        trace.poll()
        advance(clock, 50_000_000)
    trace.record("remote_rx", stamp_ns=2)
    trace.poll()
    assert len(reports(trace)) == 1
    severity, text = summaries(trace)[-1]
    assert severity == "info" and "恢复" in text
    assert "stamp_ns=1→2" in text
    for _ in range(7):
        advance(clock, 50_000_000)
        trace.poll()
    assert len(reports(trace)) == 2


def test_continuously_stale_stamps_do_not_dump_json_per_frame(recorder):
    trace, clock = recorder
    trace._context = Mock(return_value={})
    for _ in range(310):
        trace.record("joint_rx", stamp_ns=clock[1] - 1_000_000_000, ros_now_ns=clock[1])
        trace.poll()
        advance(clock, 20_000_000)
    assert len(reports(trace)) == 1
    assert len(summaries(trace)) == 2
    trace._context.assert_called_once()
    assert len(trace._pending) <= 32
    # A new clock anomaly must remain visible while the stale-stamp fault repeats.
    clock[1] += 2_000_000_000
    trace.poll()
    assert len(reports(trace)) == 2
    assert any(i["kind"] == "wall_clock_jump" for i in report(trace)["incidents"])
    # After quiet traffic, a new stale-stamp episode receives a fresh detail.
    for _ in range(520):
        trace.record("joint_rx")
        trace.poll()
        advance(clock, 20_000_000)
    before = len(reports(trace))
    trace.record("joint_rx", stamp_ns=clock[1] - 1_000_000_000, ros_now_ns=clock[1])
    trace.poll()
    assert len(reports(trace)) == before + 1


def test_executor_preserves_metadata_and_business_callback_exception(recorder):
    trace, clock = recorder
    executor = CommunicationExecutor.__new__(CommunicationExecutor)
    executor._sigint_gc = None
    executor._trace = trace
    executor._trace_clock = lambda: SimpleNamespace(nanoseconds=clock[1])
    sub = Mock()
    sub.handle = MagicMock()
    sub.callback.side_effect = ValueError("bad sample")
    executor._traced_subscriptions = {sub: "joint_rx"}
    message = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=109, nanosec=0)))
    info = {"source_timestamp": clock[1] - 1_000_000_000,
            "received_timestamp": clock[1] - 500_000_000}
    sub.handle.take_message.return_value = (message, info)
    taken = executor._take_subscription(sub)
    with pytest.raises(ValueError, match="bad sample"):
        asyncio.run(executor._execute_subscription(sub, taken))
    sub.callback.assert_called_once_with(message)
    assert "joint_rx" not in trace._active
    assert trace._streams["joint_rx"].dds_to_callback_ms == 500


def test_unregistered_subscription_keeps_original_executor_behavior(recorder):
    trace, clock = recorder
    executor = CommunicationExecutor.__new__(CommunicationExecutor)
    executor._sigint_gc = None
    executor._trace = trace
    executor._traced_subscriptions = {}
    sub = Mock()
    sub.handle = MagicMock()
    message = object()
    sub.handle.take_message.return_value = (message, {"source_timestamp": clock[1]})
    taken = executor._take_subscription(sub)
    assert taken is message
    asyncio.run(executor._execute_subscription(sub, taken))
    sub.callback.assert_called_once_with(message)
    assert not trace._streams


def test_empty_take_does_not_create_a_fake_receive_sample(recorder):
    trace, clock = recorder
    executor = CommunicationExecutor.__new__(CommunicationExecutor)
    executor._sigint_gc = None
    executor._trace = trace
    sub = Mock()
    executor._traced_subscriptions = {sub: "joint_rx"}
    asyncio.run(executor._execute_subscription(sub, None))
    sub.callback.assert_not_called()
    assert not trace._streams


def test_independent_watchdog_observes_blocked_control_thread():
    trace = CommunicationTrace(timeout_sec=0.03)
    waiting, release, detected = Event(), Event(), Event()
    records = []

    def log(_severity, text):
        if not text.startswith("communication_trace "):
            return
        data = json.loads(text.split(" ", 1)[1])
        records.append(data)
        if data["event"] == "stall":
            detected.set()

    def control():
        trace.record("command_tx")
        trace.enter("state_update")
        waiting.set()
        release.wait(2)

    worker = Thread(target=control)
    worker.start()
    try:
        assert waiting.wait(1)
        trace.start(log)
        assert detected.wait(1)
        assert worker.is_alive()  # No completed-cycle event was needed.
        stall = next(r for r in records if r["event"] == "stall")
        assert stall["active"]["control"]["stage"] == "state_update"
    finally:
        release.set()
        worker.join(1)
        trace.stop()
