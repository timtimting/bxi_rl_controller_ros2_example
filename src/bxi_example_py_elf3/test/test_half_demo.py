from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bxi_example_py_elf3.framework.joints import (
    JointCommandDefaults,
    JointCommandResolver,
    JointDefault,
    JointLayout,
    JointStateView,
)
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.framework.mod_api.resource import ResourceLoadContext
from bxi_example_py_elf3.framework.runtime.transition import compile_transition
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    DEFAULT_EXTRA_MOTOR_DISABLE_MASK,
)
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods/com.bxi.half_demo"
spec = spec_from_file_location("half_demo_test_state", MOD_ROOT / "state.py")
running = module_from_spec(spec)
spec.loader.exec_module(running)


def test_mod_asset_allows_colcon_links_but_rejects_escapes(tmp_path):
    suffix = Path("mods/com.bxi.half_demo/assets/data.txt")
    source = tmp_path / "src" / suffix
    built = tmp_path / "build" / suffix
    installed = tmp_path / "install" / suffix
    for path in (source, built, installed):
        path.parent.mkdir(parents=True)
    source.write_text("trajectory")
    built.symlink_to(source)
    installed.symlink_to(built)

    context = ResourceLoadContext("com.bxi.half_demo", installed.parent.parent)
    assert context.asset("assets/data.txt").read_text() == "trajectory"

    outside = tmp_path / "outside.txt"
    outside.write_text("not an asset")
    (installed.parent / "outside.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="must come from its assets folder"):
        context.asset("assets/outside.txt")
    with pytest.raises(ValueError, match="must come from its assets folder"):
        context.asset("../assets/data.txt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        context.asset("assets/missing.txt")


class Context:
    def __init__(self):
        self.robot_layout = running.TEST_LAYOUT
        self.robot_joints = JointStateView(
            self.robot_layout,
            running.ARM_ZERO.astype(np.float64).copy(),
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
    assert set(config["states"]) == {"running", "arm_range"}
    assert config["events"]["enter_running"] == {"slot": "btn_7", "value": 1}
    assert config["events"]["emergency_zero_torque"] == {"slot": "btn_2", "value": 1}
    assert config["events"]["initial_pose"] == {"slot": "btn_3", "value": 1}
    assert config["events"]["zero_pose"] == {"slot": "btn_4", "value": 1}
    assert config["events"]["running"] == {"slot": "btn_9", "value": 1}
    assert config["events"]["arm_range"] == {"slot": "btn_8", "value": 1}
    assert all(state["inference_hz"] == 50.0 for state in config["states"].values())
    routes = {
        (route["from"], route["event"], route["to"])
        for route in config["routes"]
    }
    assert (
        "com.bxi.basic_actions/pd_brake", "enter_running", "running"
    ) in routes
    assert (
        "com.bxi.basic_actions/initial_pos", "arm_range", "arm_range"
    ) in routes
    for source in ("com.bxi.basic_actions/zero_torque", "com.bxi.basic_actions/initial_pos"):
        assert not any(route[0] == source and route[1] == "enter_running" for route in routes)
    for source in ("running", "arm_range"):
        assert (source, "emergency_zero_torque", "com.bxi.basic_actions/zero_torque") in routes
        assert not any(route[0] == source and route[1] in ("initial_pose", "zero_pose") for route in routes)
        assert {
            (action["event"], action["action"])
            for action in config["actions"] if action["from"] == source
        } >= {
            ("initial_pose", "to_initial_pose"),
            ("zero_pose", "to_zero_pose"),
        }
    remote = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "remote_controller/config/xbox_default.yaml").read_text()
    )
    bindings = {entry["output"]: entry["when"] for entry in remote["outputs"]["level"]}
    lb_b = next(
        option for option in bindings["btn_7=1"]["any"]
        if "shoulder.left_event" in option
    )
    assert "button.b_event" in lb_b
    assert {"released": "shoulder.right_event"} in lb_b
    assert all(
        key in bindings["btn_8=1"]
        for key in ("shoulder.left_event", "button.y_event")
    )
    assert {"released": "shoulder.right_event"} in bindings["btn_8=1"]
    assert "btn_8=3" in bindings
    assert not any(
        event["slot"] == "btn_8" and event["value"] == 3
        for event in config["events"].values()
    )
    for output, button in (("btn_3=1", "button.b_event"), ("btn_4=1", "button.y_event")):
        combinations = bindings[output]["any"]
        assert any("shoulder.right_event" in option and button in option for option in combinations)


def test_running_enters_directly_from_initial_pose_without_target_jump(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    ctx.robot_joints.position[:] = running.RUN_CENTER
    session = running.RunningSession()
    frames = running.load_trajectory(MOD_ROOT / "assets/data.txt")
    state = running.RunningState(running.RUNNING, 2, session, ClipHandle(frames))
    assert _bind_logger(state) == []
    assert state.is_available(ctx)
    state.on_enter(ctx)
    assert state.phase == "blend"
    np.testing.assert_allclose(state.blend_from, running.RUN_CENTER)
    now[0] += 0.02
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    expected = running.RUN_CENTER + running.minimum_jerk_progress(
        0.02 / running.BLEND_SEC
    ) * (frames[0] - running.RUN_CENTER)
    np.testing.assert_allclose(
        ctx.frame.qpos[running.PARAMETER_INDICES], expected, atol=1e-6
    )
    assert not ctx.requested


def test_direct_running_rejects_unsettled_initial_pose(monkeypatch):
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: 100.0))
    ctx = Context()
    ctx.robot_joints.position[:] = running.RUN_CENTER
    ctx.robot_joints.position[running.TEST_LAYOUT.index("l_shoulder_x_joint")] += np.deg2rad(11)
    state = running.RunningState(
        running.RUNNING, 2, running.RunningSession(),
        ClipHandle(running.load_trajectory(MOD_ROOT / "assets/data.txt")),
    )
    errors = _bind_logger(state)
    assert not state.is_available(ctx)
    state.on_enter(ctx)
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "initial pose is not centered" in errors[-1]


@pytest.mark.parametrize("source", ("running", "arm_range"))
def test_test_entry_accepts_eight_degree_pose_error(monkeypatch, source):
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: 100.0))
    ctx = Context()
    if source == "running":
        ctx.robot_joints.position[:] = running.RUN_CENTER
        state = running.RunningState(
            running.RUNNING, 2, running.RunningSession(),
            ClipHandle(running.load_trajectory(MOD_ROOT / "assets/data.txt")),
        )
    else:
        state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    ctx.robot_joints.position[running.TEST_LAYOUT.index("l_shoulder_y_joint")] += np.deg2rad(8)
    measured_start = ctx.robot_joints.position.copy()
    errors = _bind_logger(state)

    assert state.is_available(ctx)
    state.on_enter(ctx)
    assert not errors
    assert not ctx.requested
    if source == "running":
        np.testing.assert_allclose(state.blend_from, measured_start)
    else:
        np.testing.assert_allclose(state.entry_start, measured_start)


def test_running_loop_only_commands_connected_motors_and_returns_to_initial_pose(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    ctx.robot_joints.position[:] = running.RUN_CENTER
    session = running.RunningSession()
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
    for _ in range(500):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.frame is not None:
            _follow_connected_command(ctx)
        if ctx.requested:
            break
    assert ctx.requested[-1][0] == running.PD_BRAKE


@pytest.mark.parametrize("source", ("running", "arm_range"))
@pytest.mark.parametrize(
    "action,target_state,target_pose",
    (
        ("to_initial_pose", running.PD_BRAKE, running.RUN_CENTER),
        ("to_zero_pose", running.INITIAL_POS, running.ARM_ZERO),
    ),
)
def test_controlled_mode_exit_blends_from_both_tests(
    monkeypatch, source, action, target_state, target_pose
):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    if source == "running":
        ctx.robot_joints.position[:] = running.RUN_CENTER
        state = running.RunningState(
            running.RUNNING, 2, running.RunningSession(),
            ClipHandle(running.load_trajectory(MOD_ROOT / "assets/data.txt")),
        )
    else:
        state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    assert _bind_logger(state) == []
    state.on_enter(ctx)
    shoulder = running.TEST_LAYOUT.index("l_shoulder_y_joint")
    state.last_command[shoulder] = 0.25
    ctx.robot_joints.position[:] = state.last_command
    start = state.last_command.copy()

    assert state.on_action(ctx, action)
    assert state.stop_duration == pytest.approx(running.return_duration(start, target_pose))
    now[0] += 0.02
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    expected = start + running.minimum_jerk_progress(0.02 / state.stop_duration) * (
        target_pose - start
    )
    np.testing.assert_allclose(ctx.frame.qpos[running.PARAMETER_INDICES], expected, atol=1e-6)
    assert not ctx.requested

    for _ in range(600):
        _follow_connected_command(ctx)
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.requested:
            break
    assert ctx.requested[-1][0] == target_state
    transition = ctx.requested[-1][1]["transition"]
    assert transition == {
        "type": "running_blend",
        "duration": running.EXIT_BLEND_SEC,
        "curve": "smootherstep",
        "sample_from": False,
        "sample_to": False,
    }
    np.testing.assert_allclose(ctx.frame.qpos[running.PARAMETER_INDICES], target_pose, atol=1e-6)


def test_exit_target_can_change_while_returning(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    ctx.robot_joints.position[:] = running.RUN_CENTER
    state = running.RunningState(
        running.RUNNING, 2, running.RunningSession(),
        ClipHandle(running.load_trajectory(MOD_ROOT / "assets/data.txt")),
    )
    _bind_logger(state)
    state.on_enter(ctx)
    shoulder = running.TEST_LAYOUT.index("l_shoulder_y_joint")
    state.last_command[shoulder] = 0.3
    ctx.robot_joints.position[:] = state.last_command
    assert state.on_action(ctx, "to_zero_pose")
    now[0] += 0.04
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    mid_command = state.last_command.copy()
    assert state.on_action(ctx, "to_zero_pose")
    assert state.stop_started_at == pytest.approx(100.0)
    assert state.on_action(ctx, "to_initial_pose")
    assert state.stop_started_at == pytest.approx(now[0])
    np.testing.assert_allclose(state.stop_from, mid_command)
    np.testing.assert_allclose(state.stop_target, running.RUN_CENTER)
    now[0] += 0.02
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    expected = mid_command + running.minimum_jerk_progress(0.02 / state.stop_duration) * (
        running.RUN_CENTER - mid_command
    )
    np.testing.assert_allclose(ctx.frame.qpos[running.PARAMETER_INDICES], expected, atol=1e-6)


def test_exit_blend_keeps_position_and_gains_continuous():
    layout = running.TEST_LAYOUT
    source = MotorFrame.create(
        layout,
        np.full(layout.dof_num, 0.2),
        np.full(layout.dof_num, 80.0),
        np.full(layout.dof_num, 4.0),
    )
    target = MotorFrame.create(
        layout,
        np.zeros(layout.dof_num),
        np.full(layout.dof_num, 160.0),
        np.full(layout.dof_num, 8.0),
    )

    class BlendContext:
        robot_layout = layout
        last_motor_frame = source
        frame = None

        def resolve_motor_frame(self, natural, output):
            return output.update(natural.qpos, natural.kp, natural.kd)

        def set_motor_target(self, frame):
            self.frame = frame

    ctx = BlendContext()
    to_state = SimpleNamespace(get_entry_frame=lambda _ctx: target)
    plan = compile_transition("half_demo_exit", {
        "type": "running_blend",
        "duration": running.EXIT_BLEND_SEC,
        "curve": "smootherstep",
        "sample_from": False,
        "sample_to": False,
    })
    plan.validate_states(object(), to_state)
    session = plan.create_session(ctx, object(), to_state)
    session.update(ctx, 0.0)
    np.testing.assert_allclose(ctx.frame.qpos, source.qpos)
    np.testing.assert_allclose(ctx.frame.kp, source.kp)
    np.testing.assert_allclose(ctx.frame.kd, source.kd)
    session.update(ctx, running.EXIT_BLEND_SEC / 2)
    np.testing.assert_allclose(ctx.frame.qpos, 0.5 * (source.qpos + target.qpos))
    np.testing.assert_allclose(ctx.frame.kp, 0.5 * (source.kp + target.kp))
    np.testing.assert_allclose(ctx.frame.kd, 0.5 * (source.kd + target.kd))
    session.update(ctx, running.EXIT_BLEND_SEC / 2)
    np.testing.assert_allclose(ctx.frame.qpos, target.qpos)
    np.testing.assert_allclose(ctx.frame.kp, target.kp)
    np.testing.assert_allclose(ctx.frame.kd, target.kd)


def test_failed_controlled_return_still_requests_zero_torque(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    errors = _bind_logger(state)
    state.on_enter(ctx)
    shoulder = running.TEST_LAYOUT.index("l_shoulder_y_joint")
    state.last_command[shoulder] = 0.3
    ctx.robot_joints.position[shoulder] = 0.3
    assert state.on_action(ctx, "to_zero_pose")
    for _ in range(400):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.requested:
            break
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "did not reach" in errors[-1]


def test_exit_still_requires_five_degree_accuracy(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    _bind_logger(state)
    state.on_enter(ctx)
    assert state.on_action(ctx, "stop")
    now[0] += state.stop_duration + 0.02
    measured = running.ARM_ZERO.copy()
    measured[running.TEST_LAYOUT.index("l_shoulder_y_joint")] = np.deg2rad(8)
    state._stop(ctx, measured, now[0])
    assert not ctx.requested
    measured[running.TEST_LAYOUT.index("l_shoulder_y_joint")] = np.deg2rad(4)
    state._stop(ctx, measured, now[0])
    assert ctx.requested[-1][0] == running.INITIAL_POS


def test_command_resolves_for_half_and_full_feedback_layouts():
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 1, running.RunningSession())
    _bind_logger(state)
    state._command(ctx, running.ARM_ZERO)
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
        np.testing.assert_allclose(resolved.qpos[connected], running.ARM_ZERO)
        np.testing.assert_allclose(resolved.kp[connected], running.KP)
        if layout.dof_num == 31:
            disconnected = [
                layout.index(name) for name in layout.names
                if name not in running.JOINT_NAMES and not name.startswith("head_")
            ]
            np.testing.assert_array_equal(resolved.kp[disconnected], 0)


def test_arm_range_plan_only_moves_arms_inside_software_limits():
    segments = running.arm_range_segments()
    assert len(segments) == 27
    assert [segments[index][2][0] for index in (0, 3, 6, 9)] == [
        "l_wrist_z_joint", "l_wrist_y_joint", "l_wrist_x_joint", "l_elbow_y_joint",
    ]
    non_arm = [
        running.TEST_LAYOUT.index(name)
        for name in running.JOINT_NAMES
        if not any(part in name for part in ("shoulder", "elbow", "wrist"))
    ]
    for start, target, names, duration in segments:
        assert all(name.startswith(("l_", "r_")) for name in names)
        assert duration >= 1.5
        np.testing.assert_allclose(start[non_arm], running.ARM_ZERO[non_arm])
        np.testing.assert_allclose(target[non_arm], running.ARM_ZERO[non_arm])
        assert np.all(target >= running.POSITION_MIN + running.LIMIT_MARGIN_RAD)
        assert np.all(target <= running.POSITION_MAX - running.LIMIT_MARGIN_RAD)


def test_arm_range_rejects_nonzero_start_pose(monkeypatch):
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: 100.0))
    ctx = Context()
    ctx.robot_joints.position[running.TEST_LAYOUT.index("l_shoulder_y_joint")] = np.deg2rad(11)
    state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    errors = _bind_logger(state)
    assert not state.is_available(ctx)
    state.on_enter(ctx)
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "zero pose is not centered" in errors[-1]


def test_arm_range_first_command_starts_from_measured_pose(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    ctx.robot_joints.position[running.TEST_LAYOUT.index("l_hip_y_joint")] = 0.04
    ctx.robot_joints.position[running.TEST_LAYOUT.index("l_shoulder_y_joint")] = 0.03
    state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    _bind_logger(state)
    assert state.is_available(ctx)
    state.on_enter(ctx)
    measured_start = state.entry_start.copy()
    now[0] += 0.02
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    expected = measured_start + running.minimum_jerk_progress(
        0.02 / state.entry_duration
    ) * (state.segments[0][1] - measured_start)
    np.testing.assert_allclose(ctx.frame.qpos[running.PARAMETER_INDICES], expected, atol=1e-6)


def test_controlled_return_respects_range_speed(monkeypatch):
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: 100.0))
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 3, running.RunningSession())
    _bind_logger(state)
    state.on_enter(ctx)
    state.last_command[running.TEST_LAYOUT.index("l_shoulder_y_joint")] = np.deg2rad(150)
    assert state.on_action(ctx, "stop")
    assert state.stop_duration == pytest.approx(
        1.875 * 150 / running.RETURN_SPEED_DEG_S, rel=1e-6
    )
    assert 1.875 * 150 / state.stop_duration <= running.RETURN_SPEED_DEG_S + 1e-4


def test_arm_range_completes_and_returns_to_zero_mode(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    session = running.RunningSession()
    state = running.ArmRangeState(running.ARM_RANGE, 3, session)
    errors = _bind_logger(state)
    assert state.is_available(ctx)
    state.on_enter(ctx)
    visited = set()
    for _ in range(15000):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.frame is not None:
            _assert_unconnected_joints_passive(ctx.frame)
            _follow_connected_command(ctx)
        visited.add(state.segment_index)
        if ctx.requested:
            break
    assert not errors
    assert len(state.segments) in visited
    assert ctx.requested[-1][0] == running.INITIAL_POS


def test_arm_range_stop_and_tracking_fault(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    session = running.RunningSession()
    state = running.ArmRangeState(running.ARM_RANGE, 3, session)
    errors = _bind_logger(state)
    state.on_enter(ctx)
    assert state.on_action(ctx, "stop")
    for _ in range(30):
        now[0] += 0.02
        ctx.robot_joints.timestamp_ns += 1
        state.on_update(ctx, 0.02)
        if ctx.frame is not None:
            _follow_connected_command(ctx)
        if ctx.requested:
            break
    assert ctx.requested[-1][0] == running.INITIAL_POS

    ctx.requested.clear()
    state.on_enter(ctx)
    state.holding = True
    state.segment_started_at = now[0] - 0.21
    now[0] += 0.02
    ctx.robot_joints.timestamp_ns += 1
    state.on_update(ctx, 0.02)
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "tracking error" in errors[-1]


def test_stale_connected_feedback_requests_zero_torque(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(running, "time", SimpleNamespace(monotonic=lambda: now[0]))
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 1, running.RunningSession())
    errors = _bind_logger(state)
    state.on_enter(ctx)
    state.on_update(ctx, 0.02)
    now[0] = 100.25
    state.session.last_control_at = 100.23
    state.on_update(ctx, 0.02)
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
    assert "feedback is stale" in errors[-1]


def test_out_of_limit_command_requests_zero_torque():
    ctx = Context()
    state = running.ArmRangeState(running.ARM_RANGE, 1, running.RunningSession())
    _bind_logger(state)
    target = running.ARM_ZERO.copy()
    target[running.TEST_LAYOUT.index("l_hip_y_joint")] = 100.0
    state._command(ctx, target)
    assert ctx.frame is None
    assert ctx.requested[-1][0] == running.ZERO_TORQUE
