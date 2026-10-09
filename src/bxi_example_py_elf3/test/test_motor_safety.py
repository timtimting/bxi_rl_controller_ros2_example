from threading import Event, Lock
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bxi_example_py_elf3.motor_safety import (
    EXPECTED_HARDWARE_JOINTS,
    EXPECTED_MOTOR_DISABLE_MASK,
    HardwareMotorGuard,
    MotorSafetyFault,
)
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    HALF_DEMO_REQUIRED_MOTOR_DISABLE_MASK,
    hardware_node_from_context,
)


NAMES = EXPECTED_HARDWARE_JOINTS


def feedback(guard, now=1.0, names=NAMES, effort=None):
    size = len(names)
    return guard.observe_feedback(
        names,
        [0.0] * size,
        [0.0] * size,
        [] if effort is None else effort,
        now=now,
    )


def ready_guard(now=1.0):
    guard = HardwareMotorGuard()
    guard.verify_mask(EXPECTED_MOTOR_DISABLE_MASK)
    for _ in range(3):
        assert feedback(guard, now)
    assert guard.startup_ready(now)
    return guard


def command(guard, *, now=1.0, track_position=False, **changes):
    fields = {
        "names": NAMES,
        "position": [0.0] * len(NAMES),
        "velocity": [0.0] * len(NAMES),
        "torque": [0.0] * len(NAMES),
        "kp": [20.0] * len(NAMES),
        "kd": [1.0] * len(NAMES),
    }
    fields.update(changes)
    guard.check_command(**fields, now=now, track_position=track_position)


def test_layout_and_mask_must_match_before_startup():
    assert len(NAMES) == 19
    assert EXPECTED_MOTOR_DISABLE_MASK == 0x60007DF0
    guard = HardwareMotorGuard()
    for _ in range(3):
        feedback(guard)
    assert not guard.startup_ready(1.0)
    with pytest.raises(MotorSafetyFault, match="motor_disable"):
        guard.verify_mask(0x60000000)
    assert not guard.startup_ready(1.0)
    guard.verify_mask(EXPECTED_MOTOR_DISABLE_MASK)
    assert guard.startup_ready(1.0)


def test_configured_default_cannot_change_the_verified_wiring(monkeypatch):
    from bxi_example_py_elf3 import motor_safety

    monkeypatch.setattr(motor_safety, "DEFAULT_EXTRA_MOTOR_DISABLE_MASK", 0x60000000)
    with pytest.raises(MotorSafetyFault, match="verified 19-joint configuration"):
        HardwareMotorGuard()


def test_hardware_launch_rejects_wrong_final_mask():
    launch = pytest.importorskip("launch")
    context = launch.LaunchContext()
    context.launch_configurations.update({
        "robot_config_file": "",
        "enable_head": "false",
        "enable_imu": "true",
        "motor_disable_mask": "0x0",
    })
    with pytest.raises(ValueError, match="half-body hardware motor_disable"):
        hardware_node_from_context(
            context,
            required_motor_disable=HALF_DEMO_REQUIRED_MOTOR_DISABLE_MASK,
        )


def test_startup_requires_three_fresh_complete_ordered_frames():
    guard = HardwareMotorGuard()
    guard.verify_mask(EXPECTED_MOTOR_DISABLE_MASK)
    for _ in range(2):
        feedback(guard, 1.0)
    assert not guard.startup_ready(1.0)
    feedback(guard, 1.0, NAMES[:-1])
    assert not guard.startup_ready(1.0)
    for _ in range(3):
        feedback(guard, 1.1)
    assert guard.startup_ready(1.1)
    assert not guard.startup_ready(1.31)
    with pytest.raises(MotorSafetyFault, match="order"):
        feedback(guard, 1.1, NAMES[::-1])
    with pytest.raises(MotorSafetyFault, match="unexpected"):
        feedback(guard, 1.1, NAMES[:-1] + ("missing_motor",))


def test_partial_frames_cannot_mask_stale_full_frame():
    guard = ready_guard()
    for index in range(len(NAMES)):
        feedback(guard, 1.02 + index * 0.011, NAMES[:index] + NAMES[index + 1:])
    with pytest.raises(MotorSafetyFault, match="complete 19-joint"):
        guard.check_feedback(1.22)


def test_bad_feedback_and_persistent_tracking_error():
    guard = ready_guard()
    with pytest.raises(MotorSafetyFault, match="non-finite"):
        guard.observe_feedback(NAMES, [float("nan")] * len(NAMES),
                               [0.0] * len(NAMES), [], now=1.0)
    guard.invalidate_feedback()
    assert not guard.startup_ready(1.0)
    for _ in range(3):
        feedback(guard, 2.0)
    target = [0.0] * len(NAMES)
    target[NAMES.index("l_shoulder_y_joint")] = 1.0
    command(guard, now=2.0, track_position=True, position=target)
    feedback(guard, 2.6)
    command(guard, now=2.6, track_position=True, position=target)
    feedback(guard, 2.76)
    with pytest.raises(MotorSafetyFault, match="l_shoulder_y_joint tracking error"):
        command(guard, now=2.76, track_position=True, position=target)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("names", NAMES[:-1], "expected 19 joints"),
        ("position", [4.0] + [0.0] * 18, "position command"),
        ("velocity", [11.0] + [0.0] * 18, "velocity command"),
        ("torque", [151.0] + [0.0] * 18, "torque command"),
        ("kp", [-1.0] + [0.0] * 18, "gains"),
        ("kd", [21.0] + [0.0] * 18, "gains"),
        ("position", [float("nan")] + [0.0] * 18, "non-finite"),
    ],
)
def test_final_command_rejects_invalid_values(field, value, reason):
    guard = ready_guard()
    with pytest.raises(MotorSafetyFault, match=reason):
        command(guard, **{field: value})


def test_hardware_startup_does_not_reset_without_verified_mask(monkeypatch):
    pytest.importorskip("rclpy")
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3.bxi_example_demo import BxiExample

    resets = []
    platform = SimpleNamespace(
        imu_required=True,
        _imu_received=True,
        _motor_guard=HardwareMotorGuard(),
        _poll_hardware_motor_mask=lambda now: None,
        _motor_mask_last_verified_at=0.0,
        _joint_received=True,
        _next_motor_startup_warning=2.0,
        lock_in=Lock(),
        step=0,
        robot_reset=lambda *args: resets.append(args) or True,
        get_logger=lambda: SimpleNamespace(info=lambda *_: None),
    )
    assert not BxiExample.startup_step(platform, 1.0)
    assert not resets
    platform._motor_guard.verify_mask(EXPECTED_MOTOR_DISABLE_MASK)
    for _ in range(3):
        feedback(platform._motor_guard)
    platform._motor_mask_last_verified_at = 1.0
    assert not BxiExample.startup_step(platform, 1.0)
    assert resets == [(1, False)]


def test_invalid_override_never_reaches_hardware(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("communication.msg")
    from builtin_interfaces.msg import Time
    from bxi_example_py_elf3.framework.joints import JointLayout
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3.bxi_example_demo import BxiExample

    guard = ready_guard(now=0.0)
    now = time.monotonic()
    feedback(guard, now)
    positions = np.zeros(len(NAMES), dtype=np.float32)
    frame = SimpleNamespace(
        layout=JointLayout.create(NAMES), qpos=positions,
        vel=positions, torque=positions, kp=positions, kd=positions,
    )
    bad_frame = SimpleNamespace(
        layout=frame.layout, qpos=positions, vel=positions,
        torque=positions, kp=np.full(len(NAMES), 501.0, dtype=np.float32),
        kd=positions,
    )
    messages, transitions, errors = [], [], []
    platform = SimpleNamespace(
        _motor_guard=guard,
        _motor_override=SimpleNamespace(apply=lambda *_args, **_kwargs: bad_frame),
        motor_override_allow_in_zero_torque=False,
        runtime=SimpleNamespace(
            framework=SimpleNamespace(current_state_name="com.bxi.half_demo/running"),
            request_state=lambda *args, **kwargs: transitions.append((args, kwargs)),
        ),
        topic_prefix="hardware/", lock_in=Lock(), step=2,
        get_logger=lambda: SimpleNamespace(error=errors.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
        act_pub=SimpleNamespace(publish=messages.append),
    )
    platform._latch_motor_fault = lambda reason: BxiExample._latch_motor_fault(platform, reason)
    platform._publish_safe_motor_frame = lambda: BxiExample._publish_safe_motor_frame(platform)
    BxiExample.publish_motor_frame(platform, frame)
    assert guard.fault is not None
    assert len(transitions) == 1
    assert len(messages) == 1
    assert tuple(messages[0].actuators_name) == NAMES
    assert all(value == 0.0 for value in messages[0].kp)
    assert all(value == 0.0 for value in messages[0].torque)


def test_mask_change_latches_fault_and_requests_zero_torque(monkeypatch):
    pytest.importorskip("rclpy")
    from rcl_interfaces.msg import ParameterType
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3.bxi_example_demo import BxiExample

    transitions, errors, safe_frames = [], [], []
    future = SimpleNamespace(
        done=lambda: True,
        result=lambda: SimpleNamespace(values=[SimpleNamespace(
            type=ParameterType.PARAMETER_INTEGER,
            integer_value=0x60000000,
        )]),
    )
    platform = SimpleNamespace(
        _motor_guard=ready_guard(), motor_mask_client=object(),
        _motor_mask_future=future, _motor_mask_next_query_at=0.0,
        _motor_mask_last_verified_at=1.0, _motor_mask_logged=True,
        lock_in=Lock(), step=2,
        runtime=SimpleNamespace(request_state=lambda *args, **kwargs: transitions.append((args, kwargs))),
        get_logger=lambda: SimpleNamespace(error=errors.append),
    )
    platform._latch_motor_fault = lambda reason: BxiExample._latch_motor_fault(platform, reason)
    platform._publish_safe_motor_frame = lambda: safe_frames.append(True)
    BxiExample._poll_hardware_motor_mask(platform, 1.1)
    assert "motor_disable" in platform._motor_guard.fault
    assert safe_frames == [True]
    assert transitions[0][0] == ("com.bxi.basic_actions/zero_torque",)


def test_reset_rechecks_feedback_after_service_wait(monkeypatch):
    pytest.importorskip("rclpy")
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3.bxi_example_demo import BxiExample

    guard = ready_guard(now=1.0)
    calls = []
    platform = SimpleNamespace(
        _motor_guard=guard, _motor_mask_last_verified_at=1.0,
        lock_in=Lock(), _shutting_down=Event(),
        rest_srv=SimpleNamespace(
            wait_for_service=lambda **_: True,
            call_async=calls.append,
        ),
    )
    monkeypatch.setattr(time, "monotonic", lambda: 1.21)
    assert not BxiExample.robot_reset(platform, 1, False)
    assert not calls


def test_runtime_feedback_timeout_faults_without_command_frame(monkeypatch):
    pytest.importorskip("rclpy")
    from bxi_example_py_elf3.framework.platform import cpu_affinity

    monkeypatch.setattr(cpu_affinity, "bootstrap_process_scheduling", lambda: None)
    from bxi_example_py_elf3.bxi_example_demo import BxiExample

    guard = ready_guard(now=1.0)
    faults = []
    platform = SimpleNamespace(
        _motor_guard=guard, lock_in=Lock(),
        _latch_motor_fault=faults.append,
    )
    BxiExample._check_motor_runtime_safety(platform, 1.21)
    assert len(faults) == 1
    assert "stale" in faults[0]
