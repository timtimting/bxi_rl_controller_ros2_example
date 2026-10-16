"""Remote receive, disconnect and lock-contention regressions without robot IO."""

from collections import deque
import importlib
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from bxi_example_py_elf3.remote_input import RemoteInputMonitor


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("field", ["timeout_sec", "max_age_sec"])
def test_invalid_monitor_thresholds(field, value):
    with pytest.raises(ValueError):
        RemoteInputMonitor(**{field: value})


def observe(monitor, *, stamp=1_000_000_000, received=1.0, now=1_000_000_000,
            velocity=(0.5, 0.0, 0.0)):
    return monitor.observe(
        velocity, stamp_ns=stamp, received_at=received, ros_now_ns=now
    )


def test_receive_watchdog_and_recovery():
    monitor = RemoteInputMonitor()
    assert not monitor.is_fresh(1.0)
    assert not monitor.expire(1.0)  # No input yet; initial velocity is zero.
    assert observe(monitor) == "fresh"
    assert monitor.is_fresh(1.299)
    assert monitor.expire(1.301)
    assert not monitor.expire(1.4)
    assert monitor.snapshot(1.4)["timeouts"] == 1
    assert observe(monitor, stamp=1_500_000_000, received=1.5,
                   now=1_500_000_000) == "recovered"
    assert monitor.is_fresh(1.5)


@pytest.mark.parametrize(
    "stamp,now,velocity,reason",
    [
        (1_010_000_000, 1_400_000_000, (0, 0, 0), "stale_stamp"),
        (1_800_000_000, 1_400_000_000, (0, 0, 0), "future_stamp"),
        (1_000_000_000, 1_100_000_000, (0, 0, 0), "nonincreasing_stamp"),
        (990_000_000, 1_100_000_000, (0, 0, 0), "nonincreasing_stamp"),
        (1_400_000_000, 1_400_000_000, (float("nan"), 0, 0), "invalid_velocity"),
        (1_400_000_000, 1_400_000_000, (0, float("inf"), 0), "invalid_velocity"),
    ],
)
def test_rejected_input_does_not_refresh_watchdog(stamp, now, velocity, reason):
    monitor = RemoteInputMonitor()
    observe(monitor)
    assert observe(monitor, stamp=stamp, received=1.4, now=now,
                   velocity=velocity) == reason
    assert monitor.last_received_at == 1.0
    assert monitor.expire(1.4)
    assert monitor.snapshot(1.4)["rejected"] == 1


def test_legacy_unstamped_publisher_still_has_receive_watchdog():
    monitor = RemoteInputMonitor()
    assert observe(monitor, stamp=0) == "fresh"
    assert monitor.snapshot(1.1)["source_stamp_available"] is False
    assert monitor.snapshot(1.1)["source_age_ms_at_receive"] is None
    assert monitor.expire(1.4)


@pytest.fixture
def demo():
    # Import the real controller without pinning this pytest process or starting ROS.
    with patch(
        "bxi_example_py_elf3.framework.platform.cpu_affinity.bootstrap_process_scheduling",
        return_value=None,
    ):
        return importlib.import_module("bxi_example_py_elf3.bxi_example_demo")


def controller(demo):
    from bxi_example_py_elf3.framework.joints import JointLayout, JointStateBuffer
    from bxi_example_py_elf3.framework.platform.runtime import RobotControlRuntime
    from bxi_example_py_elf3.framework.runtime.state_machine import RemoteEventAdapter

    adapter = RemoteEventAdapter({"walk": {"slot": "btn_1", "value": 1}})
    runtime = RobotControlRuntime.__new__(RobotControlRuntime)
    runtime._framework_lock = Lock()
    runtime._remote_event_lock = Lock()
    runtime.framework = SimpleNamespace(
        extract_remote_events=lambda msg, sync_only=False: adapter.extract_events(
            msg, sync_only=sync_only
        )
    )
    joints = JointStateBuffer(JointLayout(("test_joint",)), dtype=np.float64)
    node = SimpleNamespace(
        runtime=runtime, step=2, lock_in=Lock(),
        _remote_input=RemoteInputMonitor(), _next_remote_warning=0.0,
        raw_cmd_vel=np.zeros(3, dtype=np.float32), pending_remote_events=deque(),
        _check_imu_runtime_safety=lambda: None,
        _joint_source=joints, _joint_snapshot=None, _observation=None,
        _quat_xyzw_snapshot=np.zeros(4), _quat_wxyz_snapshot=np.zeros(4),
        _omega_snapshot=np.zeros(3), _linear_acceleration_snapshot=np.zeros(3),
        _cmd_snapshot=np.zeros(3, dtype=np.float32),
        quat_xyzw=np.zeros(4), quat_wxyz=np.zeros(4), omega=np.zeros(3),
        linear_acceleration=np.zeros(3), _imu_protection_latched=False,
        get_logger=Mock(return_value=Mock()),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)
        ),
    )
    return node


def message(ns=1_000_000_000, speed=0.5, button=1):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)),
        vel_des=SimpleNamespace(x=speed, y=0.0), yawdot_des=0.0, btn_1=button,
    )


def receive(demo, node, msg, now):
    with patch.object(demo.time, "monotonic", return_value=now):
        demo.BxiExample.joy_callback(node, msg)


def test_remote_callback_completes_while_control_framework_is_locked(demo):
    node = controller(demo)
    completed = Event()
    errors = []

    def callback():
        try:
            receive(demo, node, message(), 1.0)
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    with node.runtime._framework_lock:
        worker = Thread(target=callback, daemon=True)
        worker.start()
        finished_while_locked = completed.wait(1.0)
    worker.join(1.0)
    assert finished_while_locked, "remote reception waited for control/inference lock"
    assert not errors
    np.testing.assert_allclose(node.raw_cmd_vel, (0.5, 0, 0))
    assert list(node.pending_remote_events) == ["walk"]


@pytest.mark.parametrize(
    "case,log_level",
    [("joint_first", "info"), ("joint_invalid", "error"),
     ("imu_first", "info"), ("override_waiting", "warning")],
)
def test_slow_input_log_does_not_block_remote_reception(demo, case, log_level):
    node = controller(demo)
    node._joint_source = demo.NamedJointStateSource(dtype=np.float64)
    node._joint_received = False
    node._bad_joint_state_warned = False
    node.imu_required = False
    node.topic_prefix = "simulation/"
    node._imu_first_received_logged = False
    node._motor_override_waiting_for_joints_warned = False
    imu = SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    callbacks = {
        "joint_first": lambda: demo.BxiExample._update_joint_state(
            node, ["test_joint"], [0.0], [0.0]
        ),
        "joint_invalid": lambda: demo.BxiExample._update_joint_state(
            node, ["test_joint"], [], []
        ),
        "imu_first": lambda: demo.BxiExample.imu_callback(node, imu),
        "override_waiting": lambda: demo.BxiExample.motor_override_callback(
            node, SimpleNamespace(actuators_name=["test_joint"])
        ),
    }
    entered_log, release_log, remote_completed = Event(), Event(), Event()
    errors = []
    logger = node.get_logger()

    def slow_log(_message):
        entered_log.set()
        if not release_log.wait(3.0):
            raise AssertionError("test did not release the blocked log")

    getattr(logger, log_level).side_effect = slow_log

    def run_input():
        try:
            callbacks[case]()
        except BaseException as exc:
            errors.append(exc)

    def run_remote():
        try:
            receive(demo, node, message(), 1.0)
        except BaseException as exc:
            errors.append(exc)
        finally:
            remote_completed.set()

    input_worker = Thread(target=run_input, daemon=True)
    remote_worker = Thread(target=run_remote, daemon=True)
    try:
        input_worker.start()
        assert entered_log.wait(1.0), errors
        remote_worker.start()
        completed_while_logging = remote_completed.wait(1.0)
    finally:
        release_log.set()
        input_worker.join(1.0)
        if remote_worker.ident is not None:
            remote_worker.join(1.0)
    assert completed_while_logging, "log output held the shared input lock"
    assert not errors
    np.testing.assert_allclose(node.raw_cmd_vel, (0.5, 0, 0))
    assert list(node.pending_remote_events) == ["walk"]
    # First-frame logs and invalid-input suppression still happen only once.
    callbacks[case]()
    assert getattr(logger, log_level).call_count == 1
    if case == "joint_first":
        assert node._joint_received
    elif case == "joint_invalid":
        assert not node._joint_received and node._bad_joint_state_warned
    elif case == "imu_first":
        assert node._imu_received and node._imu_first_received_logged
    else:
        assert node._motor_override_waiting_for_joints_warned


def test_repeated_full_state_has_no_duplicate_button_event_and_recovers_lost_stop(demo):
    node = controller(demo)
    receive(demo, node, message(), 1.0)
    receive(demo, node, message(ns=1_010_000_000), 1.01)
    assert list(node.pending_remote_events) == ["walk"]
    # A zero update was lost; the next identical zero heartbeat still updates input.
    receive(demo, node, message(ns=1_030_000_000, speed=0), 1.03)
    np.testing.assert_array_equal(node.raw_cmd_vel, (0, 0, 0))


def test_timeout_zeros_control_snapshot_and_discards_pending_actions(demo):
    node = controller(demo)
    receive(demo, node, message(), 1.0)
    with patch.object(demo.time, "monotonic", return_value=1.4):
        observation, events = demo.BxiExample.snapshot_control_inputs(node)
    np.testing.assert_array_equal(observation.raw_cmd_vel, (0, 0, 0))
    assert events == ()
    assert not node.pending_remote_events
    assert node._remote_input.timeouts == 1


def test_recovery_does_not_replay_held_action_but_allows_next_press(demo):
    node = controller(demo)
    receive(demo, node, message(button=0), 1.0)
    receive(demo, node, message(ns=1_100_000_000), 1.5)
    assert not node.pending_remote_events  # Held reconnect button only synchronizes.
    receive(demo, node, message(ns=1_110_000_000, button=0), 1.51)
    receive(demo, node, message(ns=1_120_000_000), 1.52)
    assert list(node.pending_remote_events) == ["walk"]


def test_rejected_delayed_packet_cannot_restore_motion_or_trigger_action(demo):
    node = controller(demo)
    receive(demo, node, message(speed=0, button=0), 1.0)
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=2_000_000_000)
    )
    receive(demo, node, message(ns=1_010_000_000, speed=1), 2.0)
    np.testing.assert_array_equal(node.raw_cmd_vel, (0, 0, 0))
    assert not node.pending_remote_events
    assert node._remote_input.last_received_at == 1.0
