"""Half-body suspended running and arm-range tests for connected motors."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from bxi_example_py_elf3.framework.joints import JointLayout
from bxi_example_py_elf3.framework.mod_api import ResourceHandle, RobotControlState
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
ENTRY_TOLERANCE_RAD = math.radians(10.0)
EXIT_TOLERANCE_RAD = math.radians(5.0)
BLEND_SEC = 0.5
EXIT_BLEND_SEC = 0.3
RETURN_SPEED_DEG_S = 60.0
TRAJECTORY_HZ = 50.0
ZERO_TORQUE = "com.bxi.basic_actions/zero_torque"
PD_BRAKE = "com.bxi.basic_actions/pd_brake"
INITIAL_POS = "com.bxi.basic_actions/initial_pos"
RUNNING = "com.bxi.half_demo/running"
ARM_RANGE = "com.bxi.half_demo/arm_range"

ARM_GROUPS = (
    "wrist_z_joint", "wrist_y_joint", "wrist_x_joint", "elbow_y_joint",
    "shoulder_z_joint", "shoulder_x_joint", "shoulder_y_joint",
)
ARM_COLLISION_RANGES_DEG = {
    "wrist_z_joint": (-45.0, 45.0),
    "wrist_y_joint": (-75.0, 75.0),
    "wrist_x_joint": (-165.0, 165.0),
    "elbow_y_joint": (-55.0, 67.9),
    "shoulder_z_joint": (-30.0, 165.0),
    "shoulder_x_joint": (-5.9, 175.0),
    "shoulder_y_joint": (-165.0, 165.0),
}
ARM_ZERO = np.zeros(TEST_LAYOUT.dof_num, dtype=np.float32)


def arm_range_segments() -> tuple:
    """Plan the robot_test wrist-to-shoulder sweep with conservative margins."""
    segments = []
    previous = ARM_ZERO.copy()
    for suffix in ARM_GROUPS:
        names = (f"l_{suffix}", f"r_{suffix}")
        ranges = []
        for name in names:
            model_low, model_high = ARM_COLLISION_RANGES_DEG[suffix]
            if name.startswith("r_") and suffix in ("shoulder_x_joint", "shoulder_z_joint"):
                model_low, model_high = -model_high, -model_low
            software_low, software_high = LIMITS[name]
            low = max(math.radians(model_low + 10.0), software_low + math.radians(2.0))
            high = min(math.radians(model_high - 10.0), software_high - math.radians(2.0))
            if low >= high:
                raise ValueError(f"empty safe arm range for {name}")
            ranges.append((low, high))

        folded = ARM_ZERO.copy()
        companion = ()
        if suffix.startswith("shoulder"):
            elbow_angle = -15.0 if suffix == "shoulder_z_joint" else -45.0
            companion = ("l_elbow_y_joint", "r_elbow_y_joint")
            for name in companion:
                folded[TEST_LAYOUT.index(name)] = math.radians(elbow_angle)

        low_target = folded.copy()
        high_target = folded.copy()
        if suffix.endswith(("x_joint", "z_joint")):
            low = max(ranges[0][0], -ranges[1][1])
            high = min(ranges[0][1], -ranges[1][0])
            if low >= high:
                raise ValueError(f"empty mirrored arm range for {suffix}")
            low_target[TEST_LAYOUT.index(names[0])] = low
            low_target[TEST_LAYOUT.index(names[1])] = -low
            high_target[TEST_LAYOUT.index(names[0])] = high
            high_target[TEST_LAYOUT.index(names[1])] = -high
        else:
            low = max(ranges[0][0], ranges[1][0])
            high = min(ranges[0][1], ranges[1][1])
            if low >= high:
                raise ValueError(f"empty bilateral arm range for {suffix}")
            for name in names:
                low_target[TEST_LAYOUT.index(name)] = low
                high_target[TEST_LAYOUT.index(name)] = high

        waypoints = ((folded,) if companion else ()) + (
            low_target, high_target, folded,
        ) + ((ARM_ZERO,) if companion else ())
        motion_names = names + companion
        for target in waypoints:
            travel_deg = max(
                abs(math.degrees(target[TEST_LAYOUT.index(name)] - previous[TEST_LAYOUT.index(name)]))
                for name in motion_names
            )
            duration = max(1.5, 1.875 * travel_deg / RETURN_SPEED_DEG_S)
            segments.append((previous.copy(), target.copy(), motion_names, duration))
            previous = target
    return tuple(segments)


def minimum_jerk_progress(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def return_duration(start: np.ndarray, target: np.ndarray) -> float:
    travel_deg = math.degrees(float(np.max(np.abs(start - target))))
    return max(BLEND_SEC, 1.875 * travel_deg / RETURN_SPEED_DEG_S)


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
    def __init__(
        self,
        name: str,
        state_id: int,
        session: RunningSession,
        default_stop_state: str,
        *,
        resources=(),
    ) -> None:
        super().__init__(name, state_id, resources=resources)
        self.session = session
        self.default_stop_state = default_stop_state
        self.last_command = ARM_ZERO.copy()
        self._output_position = PARAMETERS.default_position.copy()
        self._output_kp = np.zeros(PARAMETERS.layout.dof_num, dtype=np.float32)
        self._output_kd = np.zeros(PARAMETERS.layout.dof_num, dtype=np.float32)
        self.stopping = False
        self.stop_from = ARM_ZERO.copy()
        self.stop_target = ARM_ZERO.copy()
        self.stop_state = default_stop_state
        self.stop_trigger = ""
        self.stop_started_at = 0.0
        self.stop_duration = BLEND_SEC

    def _fault(self, ctx, reason: str) -> None:
        self.session.faulted = True
        self.logger.error(f"{self.name} fault: {reason}; switching to zero torque")
        ctx.request_state(ZERO_TORQUE, trigger="half_demo_safety_fault", force=True)

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

    def on_action(self, ctx, action_name: str) -> bool:
        target_state = {
            "stop": self.default_stop_state,
            "to_initial_pose": PD_BRAKE,
            "to_zero_pose": INITIAL_POS,
        }.get(action_name)
        if target_state is None:
            return False
        if self.stopping and self.stop_state == target_state:
            return True
        self.stopping = True
        self.stop_started_at = time.monotonic()
        self.stop_from[:] = self.last_command
        self.stop_target[:] = RUN_CENTER if target_state == PD_BRAKE else ARM_ZERO
        self.stop_state = target_state
        self.stop_trigger = (
            f"{self.name.rsplit('/', 1)[-1]}_stopped"
            if action_name == "stop" else action_name
        )
        self.stop_duration = return_duration(self.stop_from, self.stop_target)
        return True

    def _stop(self, ctx, measured: np.ndarray, now: float) -> None:
        elapsed = now - self.stop_started_at
        if elapsed >= self.stop_duration:
            error = float(np.max(np.abs(measured - self.stop_target)))
            if error > EXIT_TOLERANCE_RAD and elapsed >= self.stop_duration + 5.0:
                self._fault(
                    ctx,
                    f"connected joints did not reach {self.stop_state}: "
                    f"{math.degrees(error):.2f} deg",
                )
                return
        target = self.stop_from + minimum_jerk_progress(elapsed / self.stop_duration) * (
            self.stop_target - self.stop_from
        )
        self._command(ctx, target, KP * 1.1, KD * 1.05)
        if self.session.faulted or elapsed < self.stop_duration:
            return
        if np.max(np.abs(measured - self.stop_target)) <= EXIT_TOLERANCE_RAD:
            if not ctx.request_state(
                self.stop_state,
                trigger=self.stop_trigger,
                transition={
                    "type": "running_blend",
                    "duration": EXIT_BLEND_SEC,
                    "curve": "smootherstep",
                    "sample_from": False,
                    "sample_to": False,
                },
                force=True,
            ):
                self._fault(ctx, f"cannot enter {self.stop_state} after controlled return")


class RunningState(SuspendedRunBase):
    def __init__(
        self,
        name: str,
        state_id: int,
        session: RunningSession,
        trajectory: ResourceHandle[np.ndarray],
    ) -> None:
        super().__init__(name, state_id, session, PD_BRAKE, resources=(trajectory,))
        self._trajectory = trajectory
        self.frames: np.ndarray | None = None
        self.phase = "blend"
        self.phase_started_at = 0.0
        self.next_frame_at = 0.0
        self.frame_index = 0
        self.blend_from = RUN_CENTER.copy()

    def is_available(self, ctx) -> bool:
        if self.session.faulted or not set(JOINT_NAMES).issubset(ctx.robot_layout.names):
            return False
        try:
            measured = self.session.measured(ctx)
        except (KeyError, ValueError):
            return False
        return bool(np.max(np.abs(measured - RUN_CENTER)) <= ENTRY_TOLERANCE_RAD)

    def on_enter(self, ctx) -> None:
        self.frames = self._trajectory.get()
        self.phase = "blend"
        self.phase_started_at = time.monotonic()
        self.session.last_control_at = 0.0
        self.frame_index = 0
        self.stopping = False
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        error = np.max(np.abs(measured - RUN_CENTER))
        if error > ENTRY_TOLERANCE_RAD:
            self._fault(ctx, f"initial pose is not centered: {math.degrees(error):.2f} deg")
            return
        self.blend_from[:] = measured
        self.last_command[:] = measured

    def on_update(self, ctx, dt: float) -> None:
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        now = time.monotonic()
        if self.stopping:
            self._stop(ctx, measured, now)
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


class ArmRangeState(SuspendedRunBase):
    def __init__(self, name: str, state_id: int, session: RunningSession) -> None:
        super().__init__(name, state_id, session, INITIAL_POS)
        self.segments = arm_range_segments()
        self.entry_start = ARM_ZERO.copy()
        self.entry_duration = self.segments[0][3]
        self.segment_index = 0
        self.segment_started_at = 0.0
        self.holding = False

    def is_available(self, ctx) -> bool:
        if self.session.faulted or not set(JOINT_NAMES).issubset(ctx.robot_layout.names):
            return False
        try:
            measured = self.session.measured(ctx)
        except (KeyError, ValueError):
            return False
        return bool(np.max(np.abs(measured - ARM_ZERO)) <= ENTRY_TOLERANCE_RAD)

    def on_enter(self, ctx) -> None:
        self.session.last_control_at = 0.0
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        error = np.max(np.abs(measured - ARM_ZERO))
        if error > ENTRY_TOLERANCE_RAD:
            self._fault(ctx, f"zero pose is not centered: {math.degrees(error):.2f} deg")
            return
        self.last_command[:] = measured
        self.entry_start[:] = measured
        first_target = self.segments[0][1]
        first_names = self.segments[0][2]
        first_travel_deg = max(
            abs(math.degrees(first_target[TEST_LAYOUT.index(name)] - measured[TEST_LAYOUT.index(name)]))
            for name in first_names
        )
        self.entry_duration = max(1.5, 1.875 * first_travel_deg / RETURN_SPEED_DEG_S)
        self.segment_index = 0
        self.segment_started_at = time.monotonic()
        self.holding = False
        self.stopping = False

    def on_update(self, ctx, dt: float) -> None:
        measured = self._checked_feedback(ctx)
        if measured is None:
            return
        now = time.monotonic()
        if self.stopping:
            self._stop(ctx, measured, now)
            return
        if self.segment_index >= len(self.segments):
            self.on_action(ctx, "stop")
            self._stop(ctx, measured, now)
            return
        start, target, names, duration = self.segments[self.segment_index]
        if self.segment_index == 0:
            start = self.entry_start
            duration = self.entry_duration
        elapsed = now - self.segment_started_at
        if self.holding:
            self._command(ctx, target)
            if elapsed >= 0.2:
                error = max(
                    abs(measured[TEST_LAYOUT.index(name)] - target[TEST_LAYOUT.index(name)])
                    for name in names
                )
                if error > math.radians(5.0):
                    self._fault(
                        ctx,
                        f"arm range segment {self.segment_index + 1} tracking error "
                        f"{math.degrees(error):.2f} deg",
                    )
                    return
                self.segment_index += 1
                self.segment_started_at = now
                self.holding = False
            return
        self._command(ctx, start + minimum_jerk_progress(elapsed / duration) * (target - start))
        if elapsed >= duration and not self.session.faulted:
            self.segment_started_at = now
            self.holding = True
