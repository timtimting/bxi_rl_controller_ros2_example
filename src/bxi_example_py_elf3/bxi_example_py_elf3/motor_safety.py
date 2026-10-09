"""Fail-closed command and available-feedback checks for the half-body hardware."""

from __future__ import annotations

import math
from collections.abc import Sequence

from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    DEFAULT_EXTRA_MOTOR_DISABLE_MASK,
    HALF_DEMO_REQUIRED_MOTOR_DISABLE_MASK,
)
from bxi_example_py_elf3.policies.joints import (
    ELF3_CONNECTED_POSITION_LIMITS,
    ELF3_CONNECTED_TORQUE_LIMITS,
    ELF3_POLICY_JOINTS,
)


EXPECTED_MOTOR_DISABLE_MASK = HALF_DEMO_REQUIRED_MOTOR_DISABLE_MASK
EXPECTED_HARDWARE_JOINTS = (
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "r_hip_y_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
)
FEEDBACK_TIMEOUT_SEC = 0.2
STARTUP_FEEDBACK_FRAMES = 3
TRACKING_ERROR_RAD = 0.8
TRACKING_ERROR_DURATION_SEC = 0.75
MAX_COMMAND_VELOCITY_RAD_S = 10.0
MAX_COMMAND_KP = 500.0
MAX_COMMAND_KD = 20.0


class MotorSafetyFault(ValueError):
    """A motor-safety check failed; the caller must not publish the command."""


class HardwareMotorGuard:
    def __init__(self) -> None:
        configured_names = tuple(
            name for index, name in enumerate(ELF3_POLICY_JOINTS.names)
            if not EXPECTED_MOTOR_DISABLE_MASK & (1 << index)
        )
        if (
            DEFAULT_EXTRA_MOTOR_DISABLE_MASK != EXPECTED_MOTOR_DISABLE_MASK
            or configured_names != EXPECTED_HARDWARE_JOINTS
            or set(ELF3_CONNECTED_POSITION_LIMITS) != set(EXPECTED_HARDWARE_JOINTS)
            or set(ELF3_CONNECTED_TORQUE_LIMITS) != set(EXPECTED_HARDWARE_JOINTS)
        ):
            raise MotorSafetyFault("half-body motor mask, layout, or limits differ from the verified 19-joint configuration")
        self.names = EXPECTED_HARDWARE_JOINTS
        self._last_seen = {name: None for name in self.names}
        self._position = {name: 0.0 for name in self.names}
        self._tracking_since = {name: None for name in self.names}
        self._feedback_frames = 0
        self._last_full_frame_at: float | None = None
        self._mask_verified = False
        self.fault: str | None = None

    def verify_mask(self, mask: int) -> None:
        if mask != EXPECTED_MOTOR_DISABLE_MASK:
            raise MotorSafetyFault(
                f"hardware motor_disable=0x{mask:08X}, expected "
                f"0x{EXPECTED_MOTOR_DISABLE_MASK:08X} for {len(self.names)} joints"
            )
        self._mask_verified = True

    def observe_feedback(
        self,
        names: Sequence[str],
        position: Sequence[float],
        velocity: Sequence[float],
        effort: Sequence[float],
        *,
        now: float,
    ) -> bool:
        try:
            actual = tuple(names)
        except TypeError as exc:
            raise MotorSafetyFault("hardware feedback has no joint names") from exc
        if len(set(actual)) != len(actual):
            raise MotorSafetyFault("hardware feedback contains duplicate joint names")
        unknown = tuple(name for name in actual if name not in self._last_seen)
        if unknown:
            raise MotorSafetyFault(f"hardware feedback contains unexpected joints: {unknown}")
        if len(actual) == len(self.names) and actual != self.names:
            raise MotorSafetyFault("hardware feedback joint order differs from expected order")
        positions = self._values("position feedback", position, len(actual))
        self._values("velocity feedback", velocity, len(actual))
        if len(effort) > 0:
            self._values("effort feedback", effort, len(actual))
        for index, name in enumerate(actual):
            self._last_seen[name] = now
            self._position[name] = positions[index]
        complete = actual == self.names
        self._feedback_frames = self._feedback_frames + 1 if complete else 0
        if complete:
            self._last_full_frame_at = now
        return complete

    def invalidate_feedback(self) -> None:
        self._feedback_frames = 0
        self._last_full_frame_at = None

    def startup_ready(self, now: float) -> bool:
        if self.fault is not None or not self._mask_verified:
            return False
        if self._feedback_frames < STARTUP_FEEDBACK_FRAMES:
            return False
        return (
            self._last_full_frame_at is not None
            and now - self._last_full_frame_at <= FEEDBACK_TIMEOUT_SEC
            and all(
                seen is not None and now - seen <= FEEDBACK_TIMEOUT_SEC
                for seen in self._last_seen.values()
            )
        )

    def check_feedback(self, now: float) -> None:
        if self.fault is not None:
            raise MotorSafetyFault(self.fault)
        if not self._mask_verified:
            raise MotorSafetyFault("hardware motor disable mask has not been verified")
        stale = tuple(
            name
            for name, seen in self._last_seen.items()
            if seen is None or now - seen > FEEDBACK_TIMEOUT_SEC
        )
        if stale:
            raise MotorSafetyFault(f"motor feedback missing or stale: {stale}")
        if (
            self._last_full_frame_at is None
            or now - self._last_full_frame_at > FEEDBACK_TIMEOUT_SEC
        ):
            raise MotorSafetyFault("complete 19-joint feedback frame is stale")

    def check_command(
        self,
        names: Sequence[str],
        position: Sequence[float],
        velocity: Sequence[float],
        torque: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        *,
        now: float,
        track_position: bool,
    ) -> None:
        self.check_feedback(now)
        if tuple(names) != self.names:
            raise MotorSafetyFault("final motor command must contain the expected 19 joints")
        validated = {
            field: self._values(field, values, len(self.names))
            for field, values in (
                ("position", position),
                ("velocity", velocity),
                ("torque", torque),
                ("kp", kp),
                ("kd", kd),
            )
        }
        for index, name in enumerate(self.names):
            lower, upper = ELF3_CONNECTED_POSITION_LIMITS[name]
            target = validated["position"][index]
            if not lower - 1e-5 <= target <= upper + 1e-5:
                raise MotorSafetyFault(f"{name} position command exceeds software limit")
            if abs(validated["velocity"][index]) > MAX_COMMAND_VELOCITY_RAD_S:
                raise MotorSafetyFault(f"{name} velocity command exceeds software limit")
            if abs(validated["torque"][index]) > ELF3_CONNECTED_TORQUE_LIMITS[name]:
                raise MotorSafetyFault(f"{name} torque command exceeds software limit")
            gain_p, gain_d = validated["kp"][index], validated["kd"][index]
            if not 0.0 <= gain_p <= MAX_COMMAND_KP or not 0.0 <= gain_d <= MAX_COMMAND_KD:
                raise MotorSafetyFault(f"{name} gains exceed software limit")
            if not track_position or gain_p == 0.0:
                self._tracking_since[name] = None
                continue
            if abs(target - self._position[name]) <= TRACKING_ERROR_RAD:
                self._tracking_since[name] = None
                continue
            since = self._tracking_since[name]
            if since is None:
                self._tracking_since[name] = now
            elif now - since >= TRACKING_ERROR_DURATION_SEC:
                raise MotorSafetyFault(f"{name} tracking error persists above {TRACKING_ERROR_RAD} rad")

    def safe_position(self) -> list[float]:
        return [self._position[name] for name in self.names]

    def latch(self, reason: str) -> bool:
        if self.fault is not None:
            return False
        self.fault = reason
        return True

    @staticmethod
    def _values(field: str, values: Sequence[float], count: int) -> tuple[float, ...]:
        try:
            converted = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MotorSafetyFault(f"{field} contains invalid values") from exc
        if len(converted) != count or not all(map(math.isfinite, converted)):
            raise MotorSafetyFault(f"{field} is incomplete or non-finite")
        return converted
