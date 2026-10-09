from pathlib import Path
import sys

import numpy as np
import yaml

from bxi_example_py_elf3.framework.joints import JointLayout, JointStateView
from bxi_example_py_elf3.framework.runtime.mod_loader import load_mod_runtime
from bxi_example_py_elf3.framework.runtime.state_builder import build_robot_states
from bxi_example_py_elf3.policies.joints import (
    ELF3_ISAAC_PARAMETERS,
    ELF3_POLICY_JOINTS,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = PACKAGE_ROOT / "mods"
sys.path.insert(0, str(MOD_ROOT / "com.bxi.basic_actions"))

from applause_state import (  # noqa: E402
    ApplauseState,
    HALF_LOWER_HOLD_JOINTS,
    PlaybackClip,
    load_clip,
)
from pd_brake_state import PdBrakeState  # noqa: E402


class ClipHandle:
    def __init__(self, clip):
        self.clip = clip

    def get(self):
        return self.clip


class Context:
    def __init__(self):
        lower = (
            "waist_y_joint",
            "waist_x_joint",
            "waist_z_joint",
            "l_hip_y_joint",
            "r_hip_y_joint",
        )
        arms = ELF3_POLICY_JOINTS.names[-14:]
        self.robot_layout = JointLayout.create(lower + arms)
        position = np.zeros(self.robot_layout.dof_num, dtype=np.float64)
        position[: len(lower)] = (0.11, 0.12, 0.13, 0.14, 0.15)
        velocity = np.zeros_like(position)
        self.robot_joints = JointStateView(self.robot_layout, position, velocity)
        self.requested = []

    def is_orientation_unsafe(self, _quat):
        return False

    def request_state(self, state_name, **kwargs):
        self.requested.append((state_name, kwargs))
        return True


def test_lower_hold_joints_match_connected_motor_mask():
    connected = 0x1FFF820F
    lower_names = ELF3_POLICY_JOINTS.names[:15]
    assert tuple(
        name for index, name in enumerate(lower_names) if connected & (1 << index)
    ) == HALF_LOWER_HOLD_JOINTS


def test_applause_uses_clip_for_arms_and_holds_available_lower_joints():
    clip = PlaybackClip(
        positions=np.array([[0.1] * 14, [0.2] * 14], dtype=np.float32),
        fps=10.0,
    )
    state = ApplauseState("applause", 1, ClipHandle(clip))
    ctx = Context()
    state.on_prepare(ctx, None)
    frame = state.get_entry_frame(ctx)
    layout = ELF3_ISAAC_PARAMETERS.layout
    for index, name in enumerate(ctx.robot_layout.names[:5]):
        assert np.isclose(frame.qpos[layout.index(name)], 0.11 + index * 0.01)
    assert np.isclose(
        frame.qpos[layout.index("l_knee_y_joint")],
        ELF3_ISAAC_PARAMETERS.default_position[layout.index("l_knee_y_joint")],
    )
    for name in ELF3_POLICY_JOINTS.names[-14:]:
        assert np.isclose(frame.qpos[layout.index(name)], 0.1)

    ctx.robot_joints.position[:5] = 1.0
    state.sample_running_frame(ctx, 0.1, advance=True)
    frame = state.sample_running_frame(ctx, 0.1, advance=True)
    assert np.isclose(frame.qpos[layout.index("waist_y_joint")], 0.11)
    assert np.isclose(frame.qpos[layout.index("l_shoulder_y_joint")], 0.2)


def test_applause_finishes_in_static_initial_pose():
    clip = PlaybackClip(np.zeros((1, 14), dtype=np.float32), fps=10.0)
    state = ApplauseState("applause", 1, ClipHandle(clip))
    ctx = Context()
    ctx.current_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    state.on_prepare(ctx, None)
    state.frame = 1.0
    state.on_update(ctx, 0.02)
    assert ctx.requested[0][0] == "com.bxi.basic_actions/pd_brake"


def test_applause_rejects_missing_or_invalid_connected_lower_feedback():
    clip = PlaybackClip(np.zeros((1, 14), dtype=np.float32), fps=10.0)
    state = ApplauseState("applause", 1, ClipHandle(clip))
    ctx = Context()
    assert state.is_available(ctx)
    ctx.robot_joints.position[0] = np.nan
    assert not state.is_available(ctx)
    ctx.robot_joints.position[0] = 0.0
    layout = JointLayout.create(ctx.robot_layout.names[1:])
    ctx.robot_layout = layout
    ctx.robot_joints = JointStateView(
        layout,
        np.zeros(layout.dof_num, dtype=np.float64),
        np.zeros(layout.dof_num, dtype=np.float64),
    )
    assert not state.is_available(ctx)


def test_initial_pose_requires_no_model():
    state = PdBrakeState("pd_brake", 2)
    frame = state.get_entry_frame(Context())
    assert state.required_resources == ()
    np.testing.assert_array_equal(frame.qpos, ELF3_ISAAC_PARAMETERS.default_position)


def test_half_body_mod_exposes_only_requested_states():
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config/elf3_state_machine.yaml").read_text()
    )
    runtime = load_mod_runtime(config, built_in_root=MOD_ROOT)
    try:
        states = build_robot_states(runtime.config, runtime.state_factories)
        assert set(states) == {
            "com.bxi.basic_actions/zero_torque",
            "com.bxi.basic_actions/initial_pos",
            "com.bxi.basic_actions/pd_brake",
            "com.bxi.basic_actions/imu_protection",
            "com.bxi.basic_actions/applause",
            "com.bxi.half_demo/running",
            "com.bxi.half_demo/arm_range",
        }
        assert {mod.id for mod in runtime.mods} == {
            "com.bxi.basic_actions",
            "com.bxi.half_demo",
        }
        assert len(states["com.bxi.basic_actions/applause"].required_resources) == 1
        assert len(states["com.bxi.half_demo/running"].required_resources) == 1
        assert all(
            not state.required_resources
            for name, state in states.items()
            if not name.endswith(("/applause", "/running"))
        )
    finally:
        runtime.close()


def test_applause_asset_contains_arm_positions():
    clip = load_clip(
        MOD_ROOT / "com.bxi.basic_actions/assets/applause.pkl",
        start_frame=600,
        tail_trim_frames=600,
    )
    assert clip.positions.shape[1] == 14
    assert np.isfinite(clip.positions).all()
