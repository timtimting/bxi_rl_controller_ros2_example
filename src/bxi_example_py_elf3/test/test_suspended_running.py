from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointCommandResolver,
    JointDefault,
    JointLayout,
    JointStateView,
)
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    DEFAULT_EXTRA_MOTOR_DISABLE_MASK,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods/com.bxi.suspended_running"
spec = spec_from_file_location("suspended_running_test_state", MOD_ROOT / "state.py")
running = module_from_spec(spec)
spec.loader.exec_module(running)


class Context:
    def __init__(self):
        self.robot_layout = running.TEST_LAYOUT
        self.robot_joints = JointStateView(
            self.robot_layout,
            running.IDLE_CENTER.astype(np.float64).copy(),
            np.zeros(19, dtype=np.float64),
            timestamp_ns=1,
        )
        self.frame = None
        self.requested = []

    def set_motor_target(self, frame):
        self.frame = frame

    def request_state(self, name, **kwargs):
        self.requested.append((name, kwargs))
        return True


class ClipHandle:
    def __init__(self, frames):
        self.frames = frames

    def get(self):
        return self.frames


def _bind_logger(state):
    errors = []
    state._bind_logger(SimpleNamespace(error=errors.append))
    return errors


def _follow_connected_command(ctx):
    for index, name in enumerate(ctx.robot_layout.names):
        ctx.robot_joints.position[index] = ctx.frame.qpos[ctx.frame.layout.index(name)]


def _assert_unconnected_joints_passive(frame):
    assert frame.layout.names == running.PARAMETERS.layout.names
    assert len(frame.qpos) == 29
    disconnected = [
        frame.layout.index(name)
        for name in frame.layout.names
        if name not in running.JOINT_NAMES
    ]
    assert len(disconnected) == 10
    np.testing.assert_array_equal(frame.kp[disconnected], 0)
    np.testing.assert_array_equal(frame.kd[disconnected], 0)


def test_trajectory_uses_only_connected_columns():
    path = MOD_ROOT / "assets/data.txt"
    original = np.loadtxt(path, delimiter=",")
    frames = running.load_trajectory(path)
    expected_indices = [
        index
        for index in range(len(ELF3_POLICY_JOINTS.names))
        if not DEFAULT_EXTRA_MOTOR_DISABLE_MASK & (1 << index)
    ]
    assert original.shape == (308, 29)
    assert frames.shape == (308, 19)
    assert running.JOINT_NAMES == tuple(
        ELF3_POLICY_JOINTS.names[index] for index in expected_indices
    )
    np.testing.assert_allclose(frames, original[:, expected_indices], rtol=1e-6)
    assert not any("knee" in name or "ankle" in name for name in running.JOINT_NAMES)


def test_remote_combinations_and_state_routes():
    config = yaml.safe_load((MOD_ROOT / "mod.yaml").read_text())
    assert set(config["states"]) == {"idle", "running"}
    assert config["events"]["enter_test"] == {"slot": "btn_8", "value": 1}
    assert config["events"]["exit_test"] == {"slot": "btn_8", "value": 2}
    assert config["events"]["running"] == {"slot": "btn_9", "value": 1}
    assert all(state["inference_hz"] == 50.0 for state in config["states"].values())
    remote = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "remote_controller/config/xbox_default.yaml").read_text()
    )
    bindings = {entry["output"]: entry["when"] for entry in remote["outputs"]["level"]}
    assert all(
        key in bindings["btn_8=1"]
        for key in ("shoulder.left_event", "shoulder.right_event", "button.y_event")
    )
    assert all(
        key in bindings["btn_8=2"]
        for key in ("shoulder.left_event", "shoulder.right_event", "button.b_event")
    )


def test_idle_and_run_only_command_connected_motors(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    session = running.RunningSession()
    idle = running.RunningIdleState(running.IDLE, 1, session)
    assert _bind_logger(idle) == []
    idle.on_prepare(ctx, SimpleNamespace(name=running.ZERO_TORQUE))
    idle.on_enter(ctx)
    for _ in range(155):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        idle.on_update(ctx, 0.02)
        if ctx.frame is not None:
            _follow_connected_command(ctx)
    assert session.ready
    _assert_unconnected_joints_passive(ctx.frame)

    frames = running.load_trajectory(MOD_ROOT / "assets/data.txt")
    state = running.RunningState(running.RUNNING, 2, session, ClipHandle(frames))
    assert _bind_logger(state) == []
    assert len(state.required_resources) == 1
    assert state.is_available(ctx)
    state.on_enter(ctx)
    observed_phases = set()
    for _ in range(500):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        observed_phases.add(state.phase)
        if ctx.frame is not None:
            _follow_connected_command(ctx)
    assert "loop_blend" in observed_phases
    assert state.phase == "playback"
    _assert_unconnected_joints_passive(ctx.frame)

    assert state.on_action(ctx, "stop")
    for _ in range(30):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.frame is not None:
            _follow_connected_command(ctx)
        if ctx.requested:
            break
    assert ctx.requested[-1][0] == running.IDLE


def test_command_resolves_for_half_and_full_feedback_layouts():
    ctx = Context()
    state = running.RunningIdleState(running.IDLE, 1, running.RunningSession())
    _bind_logger(state)
    state._command(ctx, running.IDLE_CENTER)
    _assert_unconnected_joints_passive(ctx.frame)
    defaults = JointCommandDefaults({
        name: JointDefault(position=0.0, kp=16.747, kd=1.066)
        for name in ("head_z_joint", "head_y_joint")
    })
    for layout in (
        running.TEST_LAYOUT,
        JointLayout(running.PARAMETERS.layout.names + ("head_z_joint", "head_y_joint")),
    ):
        resolved = MotorFrame.empty(layout)
        warnings = []
        JointCommandResolver(layout, defaults, warning_callback=warnings.append).resolve_into(
            ctx.frame, resolved
        )
        assert bool(warnings) == (layout.dof_num == 19)
        connected = [layout.index(name) for name in running.JOINT_NAMES]
        assert len(connected) == 19
        np.testing.assert_allclose(resolved.qpos[connected], running.IDLE_CENTER)
        np.testing.assert_allclose(resolved.kp[connected], running.KP)
        if layout.dof_num == 31:
            disconnected = [
                layout.index(name) for name in layout.names
                if name not in running.JOINT_NAMES and not name.startswith("head_")
            ]
            np.testing.assert_array_equal(resolved.kp[disconnected], 0)


def test_stale_connected_feedback_requests_zero_torque(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    state = running.RunningIdleState(running.IDLE, 1, running.RunningSession())
    errors = _bind_logger(state)
    state.on_prepare(ctx, SimpleNamespace(name=running.ZERO_TORQUE))
    state.on_enter(ctx)
    state.on_update(ctx, 0.02)
    now[0] = 100.25
    state.session.last_control_at = 100.23
    state.on_update(ctx, 0.02)
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "feedback is stale" in errors[-1]


def test_out_of_limit_command_requests_zero_torque():
    ctx = Context()
    state = running.RunningIdleState(running.IDLE, 1, running.RunningSession())
    _bind_logger(state)
    target = running.IDLE_CENTER.copy()
    target[running.TEST_LAYOUT.index("l_hip_y_joint")] = 100.0
    state._command(ctx, target)
    assert ctx.frame is None
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
