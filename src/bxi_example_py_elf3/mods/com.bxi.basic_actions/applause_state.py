from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bxi_example_py_elf3.policies.joints import (
    ELF3_ISAAC_PARAMETERS,
    ELF3_POLICY_JOINTS,
)
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.framework.mod_api import StateBehavior
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


APPLAUSE_ARM_JOINTS = ELF3_POLICY_JOINTS.names[-14:]
HALF_LOWER_HOLD_JOINTS = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",
    "r_hip_y_joint",
)


@dataclass(frozen=True)
class PlaybackClip:
    positions: NDArray[np.float32]
    fps: float


def load_clip(path: Path, *, start_frame: int, tail_trim_frames: int) -> PlaybackClip:
    with path.open("rb") as input_file:
        raw = pickle.load(input_file)
    if not isinstance(raw, dict):
        raise ValueError(f"playback clip must contain a map: {path}")
    positions = np.asarray(raw["dof_pos"], dtype=np.float32)[:, -14:]
    start = min(start_frame, positions.shape[0])
    end = max(start, positions.shape[0] - tail_trim_frames)
    positions = positions[start:end]
    if positions.shape[0] == 0:
        raise ValueError(f"playback clip is empty after trimming: {path}")
    return PlaybackClip(positions, float(raw["fps"]))


class ApplauseState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    def __init__(
        self,
        name: str,
        state_id: int,
        clip: ResourceHandle[PlaybackClip],
    ) -> None:
        super().__init__(name, state_id, resources=(clip,))
        self._clip = clip
        self.frame = 0.0
        self.playing = True
        layout = ELF3_ISAAC_PARAMETERS.layout
        self._arm_output_indices = np.fromiter(
            (layout.index(name) for name in APPLAUSE_ARM_JOINTS),
            dtype=np.intp,
            count=len(APPLAUSE_ARM_JOINTS),
        )
        self._lower_output_indices = np.fromiter(
            (layout.index(name) for name in HALF_LOWER_HOLD_JOINTS),
            dtype=np.intp,
            count=len(HALF_LOWER_HOLD_JOINTS),
        )
        self._target_position = ELF3_ISAAC_PARAMETERS.default_position.copy()

    @property
    def clip(self) -> PlaybackClip:
        return self._clip.get()

    def is_available(self, ctx: RobotControlContext) -> bool:
        robot_names = set(ctx.robot_layout.names)
        if not set(HALF_LOWER_HOLD_JOINTS).issubset(robot_names):
            return False
        indices = [ctx.robot_layout.index(name) for name in HALF_LOWER_HOLD_JOINTS]
        return bool(np.isfinite(ctx.robot_joints.position[indices]).all())

    def on_prepare(
        self, ctx: RobotControlContext, from_state: StateBehavior[RobotControlContext]
    ) -> None:
        self.frame = 0.0
        self.playing = True
        if not self.is_available(ctx):
            raise ValueError("cannot enter applause without finite half-body feedback")
        lower_robot_indices = np.fromiter(
            (ctx.robot_layout.index(name) for name in HALF_LOWER_HOLD_JOINTS),
            dtype=np.intp,
            count=len(HALF_LOWER_HOLD_JOINTS),
        )
        lower_position = ctx.robot_joints.position[lower_robot_indices]
        self._target_position[self._lower_output_indices] = lower_position

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.frame = 0.0
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._frame(ctx, 0)

    def _frame(self, ctx: RobotControlContext, index: int) -> MotorFrame:
        self._target_position[self._arm_output_indices] = self.clip.positions[index]
        parameters = ELF3_ISAAC_PARAMETERS
        return self._motor_frame(
            ctx,
            self._target_position,
            parameters.kp,
            parameters.kd,
            layout=parameters.layout,
        )

    def sample_running_frame(
        self, ctx: RobotControlContext, dt: float, *, advance: bool
    ) -> MotorFrame:
        index = min(int(self.frame), self.clip.positions.shape[0] - 1)
        frame = self._frame(ctx, index)
        if self.playing and advance:
            self.frame += self.clip.fps * dt
        return frame

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
            return
        if self.frame >= self.clip.positions.shape[0]:
            ctx.request_state(
                "com.bxi.basic_actions/pd_brake",
                trigger="applause_finished",
                transition={"profile": "dual_running_blend", "duration": 1.0},
            )
            return
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True
