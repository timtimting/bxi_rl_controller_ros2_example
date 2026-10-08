"""The robot_test suspended run, restricted to the connected half-body motors."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from bxi_example_py_elf3.framework.joints import JointLayout
from bxi_example_py_elf3.framework.mod_api import ResourceHandle, RobotControlState
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    DEFAULT_EXTRA_MOTOR_DISABLE_MASK,
)
from bxi_example_py_elf3.policies.joints import (
    ELF3_ISAAC_PARAMETERS,
    ELF3_POLICY_JOINTS,
)


SOURCE_NAMES = ELF3_POLICY_JOINTS.names
SOURCE_INDICES = np.asarray(
    [index for index in range(len(SOURCE_NAMES)) if not DEFAULT_EXTRA_MOTOR_DISABLE_MASK & (1 << index)],
    dtype=np.intp,
)
JOINT_NAMES = tuple(SOURCE_NAMES[index] for index in SOURCE_INDICES)
TEST_LAYOUT = JointLayout(JOINT_NAMES, label="ELF3 connected suspended run")
PARAMETERS = ELF3_ISAAC_PARAMETERS
PARAMETER_INDICES = np.asarray(
    [PARAMETERS.layout.index(name) for name in JOINT_NAMES], dtype=np.intp
)
KP = PARAMETERS.kp[PARAMETER_INDICES].copy()
KD = PARAMETERS.kd[PARAMETER_INDICES].copy()
RUN_CENTER = PARAMETERS.default_position[PARAMETER_INDICES].copy()
IDLE_CENTER = RUN_CENTER.copy()
IDLE_CENTER[TEST_LAYOUT.index("l_shoulder_x_joint")] = 0.3
IDLE_CENTER[TEST_LAYOUT.index("r_shoulder_x_joint")] = -0.3

# Software limits from robot_test/control/elf3.py, in TEST_LAYOUT order.
LIMITS = {
    "waist_y_joint": (-0.5236, 0.5236),
    "waist_x_joint": (-0.2618, 0.2618),
    "waist_z_joint": (-2.8798, 2.8798),
    "l_hip_y_joint": (-2.8798, 2.8798),
    "r_hip_y_joint": (-2.8798, 2.8798),
    "l_shoulder_y_joint": (-2.8798, 2.8798),
    "l_shoulder_x_joint": (-0.34907, 3.0543),
    "l_shoulder_z_joint": (-2.8798, 2.8798),
    "l_elbow_y_joint": (-0.95993, 1.6581),
    "l_wrist_x_joint": (-2.8798, 2.8798),
    "l_wrist_y_joint": (-1.309, 1.309),
    "l_wrist_z_joint": (-0.7854, 0.7854),
    "r_shoulder_y_joint": (-2.8798, 2.8798),
    "r_shoulder_x_joint": (-3.0543, 0.34907),
    "r_shoulder_z_joint": (-2.8798, 2.8798),
    "r_elbow_y_joint": (-0.95993, 1.6581),
    "r_wrist_x_joint": (-2.8798, 2.8798),
    "r_wrist_y_joint": (-1.309, 1.309),
    "r_wrist_z_joint": (-0.7854, 0.7854),
}
POSITION_MIN = np.asarray([LIMITS[name][0] for name in JOINT_NAMES])
POSITION_MAX = np.asarray([LIMITS[name][1] for name in JOINT_NAMES])
LIMIT_MARGIN_RAD = 0.02
FEEDBACK_TIMEOUT_SEC = 0.2
COMMAND_GAP_TIMEOUT_SEC = 0.05
CENTER_TOLERANCE_RAD = math.radians(5.0)
PREPARE_SEC = 3.0
BLEND_SEC = 0.5
TRAJECTORY_HZ = 50.0
ZERO_TORQUE = "com.bxi.basic_actions/zero_torque"
IDLE = "com.bxi.suspended_running/idle"
RUNNING = "com.bxi.suspended_running/running"


def minimum_jerk_progress(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def load_trajectory(path: Path) -> np.ndarray:
    try:
        source = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read running trajectory {path}: {exc}") from exc
    if source.shape[1] != len(SOURCE_NAMES) or source.shape[0] == 0:
        raise ValueError(
            f"running trajectory must have shape (frames, {len(SOURCE_NAMES)})"
        )
    trajectory = source[:, SOURCE_INDICES]
    if not np.isfinite(trajectory).all():
        raise ValueError("running trajectory contains a non-finite position")
    invalid = np.argwhere(
        (trajectory < POSITION_MIN + LIMIT_MARGIN_RAD)
        | (trajectory > POSITION_MAX - LIMIT_MARGIN_RAD)
    )
    if invalid.size:
        row, column = invalid[0]
        raise ValueError(
            f"running trajectory frame {row + 1} exceeds {JOINT_NAMES[column]} "
            "software position limit"
        )
    return np.asarray(trajectory, dtype=np.float32)


class RunningSession:
    def __init__(self) -> None:
        self.ready = False
        self.faulted = False
        self.last_feedback_stamp = None
        self.last_feedback_at = 0.0
        self.last_control_at = 0.0

    def measured(self, ctx) -> np.ndarray:
        now = time.monotonic()
        stamp = ctx.robot_joints.timestamp_ns
        if stamp != self.last_feedback_stamp:
            self.last_feedback_stamp = stamp
            self.last_feedback_at = now
        if now - self.last_feedback_at > FEEDBACK_TIMEOUT_SEC:
            raise ValueError("connected joint feedback is stale")
        positions = ctx.robot_joints.position
        result = np.asarray(
            [positions[ctx.robot_layout.index(name)] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        if not np.isfinite(result).all():
            raise ValueError("connected joint feedback contains non-finite positions")
        return result


class SuspendedRunBase(RobotControlState):
    def __init__(self, name: str, state_id: int, session: RunningSession, *, resources=()) -> None:
        super().__init__(name, state_id, resources=resources)
        self.session = session
        self.last_command = IDLE_CENTER.copy()
        self._output_position = PARAMETERS.default_position.copy()
        self._output_kp = np.zeros(PARAMETERS.layout.dof_num, dtype=np.float32)
        self._output_kd = np.zeros(PARAMETERS.layout.dof_num, dtype=np.float32)

    def _fault(self, ctx, reason: str) -> None:
        self.session.ready = False
        self.session.faulted = True
        self.logger.error(f"suspended running fault: {reason}; switching to zero torque")
        ctx.request_state(ZERO_TORQUE, trigger="running_safety_fault", force=True)

    def _checked_feedback(self, ctx) -> np.ndarray | None:
        try:
            now = time.monotonic()
            if self.session.last_control_at and now - self.session.last_control_at > COMMAND_GAP_TIMEOUT_SEC:
                raise ValueError("running control cycle gap exceeded 50 ms")
            self.session.last_control_at = now
            return self.session.measured(ctx)
        except (KeyError, ValueError) as exc:
            self._fault(ctx, str(exc))
            return None

    def _command(self, ctx, position, kp=KP, kd=KD) -> None:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (TEST_LAYOUT.dof_num,) or not np.isfinite(position).all():
            self._fault(ctx, "invalid running command position")
            return
        invalid = np.flatnonzero(
            (position < POSITION_MIN + LIMIT_MARGIN_RAD)
            | (position > POSITION_MAX - LIMIT_MARGIN_RAD)
        )
        if invalid.size:
            self._fault(ctx, f"running command exceeds {JOINT_NAMES[int(invalid[0])]} position limit")
            return
        self.last_command[:] = position
        self._output_position[PARAMETER_INDICES] = position
        self._output_kp[PARAMETER_INDICES] = kp
        self._output_kd[PARAMETER_INDICES] = kd
        self._apply_frame(
            ctx,
            self._motor_frame(
                ctx,
                self._output_position,
                self._output_kp,
                self._output_kd,
                layout=PARAMETERS.layout,
            ),
        )


class RunningIdleState(SuspendedRunBase):
    def __init__(self, name: str, state_id: int, session: RunningSession) -> None:
        super().__init__(name, state_id, session)
        self.entered_at = 0.0
        self.start = IDLE_CENTER.copy()
        self.from_running = False

    def is_available(self, ctx) -> bool:
        return not self.session.faulted and set(JOINT_NAMES).issubset(ctx.robot_layout.names)

    def on_prepare(self, ctx, from_state) -> None:
        self.from_running = from_state.name == RUNNING

    def on_enter(self, ctx) -> None:
        self.entered_at = time.monotonic()
        self.session.last_control_at = 0.0
        self.session.ready = False
        try:
            self.start[:] = self.session.measured(ctx)
        except (KeyError, ValueError) as exc:
            self._fault(ctx, str(exc))

    def on_update(self, ctx, dt: float) -> None:
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        elapsed = time.monotonic() - self.entered_at
        if self.from_running:
            target = IDLE_CENTER
            kp = KP * 1.1
        else:
            progress = minimum_jerk_progress(elapsed / PREPARE_SEC)
            target = self.start + progress * (IDLE_CENTER - self.start)
            kp = KP * 1.1 * min(elapsed / PREPARE_SEC, 1.0)
        self._command(ctx, target, kp, KD * 1.05)
        if not self.session.faulted and (self.from_running or elapsed >= PREPARE_SEC):
            error = np.max(np.abs(measured - IDLE_CENTER))
            if error <= CENTER_TOLERANCE_RAD:
                self.session.ready = True
            elif elapsed > (5.0 if self.from_running else PREPARE_SEC + 5.0):
                self._fault(ctx, f"connected joints did not settle at center: {math.degrees(error):.2f} deg")


class RunningState(SuspendedRunBase):
    def __init__(
        self,
        name: str,
        state_id: int,
        session: RunningSession,
        trajectory: ResourceHandle[np.ndarray],
    ) -> None:
        super().__init__(name, state_id, session, resources=(trajectory,))
        self._trajectory = trajectory
        self.frames: np.ndarray | None = None
        self.phase = "center_blend"
        self.phase_started_at = 0.0
        self.next_frame_at = 0.0
        self.frame_index = 0
        self.stopping = False
        self.stop_from = IDLE_CENTER.copy()
        self.stop_started_at = 0.0
        self.blend_from = RUN_CENTER.copy()

    def is_available(self, ctx) -> bool:
        return self.session.ready and not self.session.faulted

    def on_enter(self, ctx) -> None:
        self.frames = self._trajectory.get()
        self.phase = "center_blend"
        self.phase_started_at = time.monotonic()
        self.session.last_control_at = 0.0
        self.last_command[:] = IDLE_CENTER
        self.frame_index = 0
        self.stopping = False
        self.blend_from[:] = RUN_CENTER

    def on_action(self, ctx, action_name: str) -> bool:
        if action_name != "stop":
            return False
        if not self.stopping:
            self.stopping = True
            self.stop_started_at = time.monotonic()
            self.stop_from[:] = self.last_command
        return True

    def _stop(self, ctx, measured: np.ndarray, now: float) -> None:
        elapsed = now - self.stop_started_at
        if elapsed >= BLEND_SEC:
            error = np.max(np.abs(measured - IDLE_CENTER))
            if error <= CENTER_TOLERANCE_RAD:
                ctx.request_state(IDLE, trigger="running_stopped", force=True)
                return
            if elapsed >= BLEND_SEC + 5.0:
                self._fault(ctx, f"connected joints did not return to center: {math.degrees(error):.2f} deg")
                return
        target = self.stop_from + minimum_jerk_progress(elapsed / BLEND_SEC) * (IDLE_CENTER - self.stop_from)
        self._command(ctx, target, KP * 1.1, KD * 1.05)

    def on_update(self, ctx, dt: float) -> None:
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        now = time.monotonic()
        if self.stopping:
            self._stop(ctx, measured, now)
            return
        if self.phase == "center_blend":
            progress = (now - self.phase_started_at) / BLEND_SEC
            self._command(ctx, IDLE_CENTER + minimum_jerk_progress(progress) * (RUN_CENTER - IDLE_CENTER))
            if progress >= 1.0:
                self.phase = "settle"
                self.phase_started_at = now
            return
        if self.phase == "settle":
            self._command(ctx, RUN_CENTER)
            if now - self.phase_started_at >= 2.0:
                error = np.max(np.abs(measured - RUN_CENTER))
                if error <= CENTER_TOLERANCE_RAD:
                    self.phase = "blend"
                    self.phase_started_at = now
                elif now - self.phase_started_at > 5.0:
                    self._fault(ctx, "running center did not settle")
            return
        frames = self.frames
        if frames is None:
            self._fault(ctx, "running trajectory is unavailable")
            return
        if self.phase in ("blend", "loop_blend"):
            progress = (now - self.phase_started_at) / BLEND_SEC
            self._command(
                ctx,
                self.blend_from
                + minimum_jerk_progress(progress) * (frames[0] - self.blend_from),
            )
            if progress >= 1.0:
                self.phase = "playback"
                self.frame_index = 1
                self.next_frame_at = now + 1.0 / TRAJECTORY_HZ
            return
        if now < self.next_frame_at:
            self._command(ctx, self.last_command)
        elif self.frame_index >= len(frames):
            self.phase = "loop_blend"
            self.phase_started_at = now
            self.blend_from[:] = self.last_command
            self._command(ctx, self.last_command)
        else:
            self._command(ctx, frames[self.frame_index])
            self.frame_index += 1
            self.next_frame_at += 1.0 / TRAJECTORY_HZ
