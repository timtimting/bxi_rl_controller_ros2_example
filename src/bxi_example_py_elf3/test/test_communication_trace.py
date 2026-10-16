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
    return json.loads(trace._log.call_args.args[1].split(" ", 1)[1])


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
    advance(clock, 1_000_000_000)
    trace.record("command_tx", stamp_ns=44)
    trace.enter("waiting_next_cycle")
    trace.poll()
    assert report(trace)["event"] == "recovered"


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
    trace.poll()
    for _ in range(10):
        advance(clock, 50_000_000)
        trace.poll()
    assert trace._log.call_count == 1
    assert len(trace._history) == 10
    assert len(trace._pending) <= 32


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
