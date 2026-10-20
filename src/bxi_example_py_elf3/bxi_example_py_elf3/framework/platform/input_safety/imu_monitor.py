"""IMU input and protection state shared by ROS callbacks and the control loop.

Callers hold the platform input lock while using this object. Logging and ROS
state requests remain with the platform adapter, outside that lock.
"""

from dataclasses import dataclass

import numpy as np

from bxi_example_py_elf3.framework.mod_api import MotorFrame
from bxi_example_py_elf3.framework.platform.input_safety.imu_diagnostics import (
    ImuEvidence,
    ImuFrameRecorder,
)
from bxi_example_py_elf3.framework.platform.input_safety.imu_freshness import (
    ImuFallObserver,
    ImuFaultObserver,
    ImuFrameAssessment,
    ImuStampObserver,
    assess_imu_frame,
)


@dataclass(frozen=True, slots=True)
class ImuStartupReport:
    evidence: ImuEvidence | None
    warn_waiting: bool


@dataclass(frozen=True, slots=True)
class ImuRuntimeReport:
    last_valid_at: float | None
    timeout_check: tuple[float, int, bool] | None
    evidence: ImuEvidence | None
    protection_latched: bool
    fall_latched: bool


@dataclass(frozen=True, slots=True)
class ImuFrameReport:
    assessment: ImuFrameAssessment
    stamp_ns: int
    stamp_status: str
    stamp_age_ms: float | None
    gap_ms: float | None
    streak: int
    warn_frame: bool
    fall_streak: int
    warn_orientation: bool
    fall_confirmed: bool
    frame_evidence: ImuEvidence | None
    fall_evidence: ImuEvidence | None
    followups: tuple[ImuEvidence, ...]
    first_frame: bool


class ImuSafetyMonitor:
    def __init__(
        self, *, required: bool, startup_timeout_sec: float,
        runtime_timeout_sec: float, guard_kp: float, guard_kd: float,
        allow_release: bool = False,
    ):
        if runtime_timeout_sec <= 0.0:
            raise ValueError("/imu_runtime_timeout_sec must be positive")
        self.required = required
        self.startup_timeout_sec = startup_timeout_sec
        self.runtime_timeout_sec = runtime_timeout_sec
        self.guard_kp = guard_kp
        self.guard_kd = guard_kd
        self.allow_release = allow_release

        self.omega = np.zeros(3, dtype=np.double)
        self.linear_acceleration = np.zeros(3, dtype=np.double)
        self.quat_xyzw = np.zeros(4, dtype=np.double)
        self.quat_wxyz = np.zeros(4, dtype=np.double)
        self.received = False
        self.last_received_at = None
        self.protection_latched = False
        self.fall_protection_latched = False
        self.orientation_guard_active = False
        self.latest_frame_fault = False
        self.inference_paused = False
        self.guard_frame: MotorFrame | None = None
        self.guard_zero_torque = False
        self.next_fall_request_warning = 0.0
        self.first_received_logged = False
        self.stamp_observer = ImuStampObserver()
        self.fault_observer = ImuFaultObserver(confirm_count=5)
        self.fall_observer = ImuFallObserver()
        self.frame_recorder = ImuFrameRecorder()
        self.startup_started_at = None
        self.startup_timeout_logged = False
        self.next_startup_warning = 0.0

    def startup_check(self, now: float) -> ImuStartupReport | None:
        if not self.required or self.received:
            return None
        evidence = None
        if self.startup_started_at is None:
            self.startup_started_at = now
        elif (
            not self.startup_timeout_logged
            and now - self.startup_started_at >= self.startup_timeout_sec
        ):
            evidence = self.frame_recorder.trigger(
                "startup_timeout", at=now,
                timeout_ms=self.startup_timeout_sec * 1000,
            )
            self.startup_timeout_logged = True
        warn_waiting = now >= self.next_startup_warning
        if warn_waiting:
            self.next_startup_warning = now + 2.0
        return ImuStartupReport(evidence=evidence, warn_waiting=warn_waiting)

    def set_orientation_guard_active(self, active: bool) -> None:
        if active != self.orientation_guard_active:
            self.fall_observer.reset_streak(clear_pending=True)
            self.orientation_guard_active = active

    def block_reason(self, now: float) -> str | None:
        if self.fall_protection_latched:
            return "fall_protection"
        if self.required:
            if self.protection_latched:
                return "data_protection"
            if self.latest_frame_fault:
                return "invalid_frame"
        if self.orientation_guard_active and self.fall_observer.consecutive > 0:
            return "orientation_limit"
        if not self.required:
            return None
        if self.last_received_at is None:
            return "missing_frame"
        if now - self.last_received_at >= self.runtime_timeout_sec:
            return "receive_timeout"
        return None

    def set_inference_blocked(self, blocked: bool) -> bool:
        changed = blocked != self.inference_paused
        self.inference_paused = blocked
        if not blocked:
            self.guard_frame = None
            self.guard_zero_torque = False
        return changed

    def safety_frame(self, joints, state_name: str) -> MotorFrame:
        frame = self.guard_frame
        if frame is None or frame.layout.names != joints.layout.names:
            frame = MotorFrame.empty(joints.layout)
            np.copyto(frame.qpos, joints.position, casting="same_kind")
            self.guard_frame = frame
            self.guard_zero_torque = False
        self.guard_zero_torque |= (
            self.fall_protection_latched
            or state_name == "com.bxi.basic_actions/zero_torque"
        )
        if self.guard_zero_torque:
            frame.kp.fill(0.0)
            frame.kd.fill(0.0)
        else:
            frame.kp.fill(self.guard_kp)
            frame.kd.fill(self.guard_kd)
        return frame

    def check_runtime_timeout(self, now: float, period_sec: float) -> ImuRuntimeReport:
        last_received_at = self.last_received_at
        timeout_check = None
        evidence = None
        if (
            self.required
            and self.received
            and last_received_at is not None
            and not self.protection_latched
            and not self.fall_protection_latched
        ):
            timeout_check = self.fault_observer.check_timeout(
                now, last_received_at, self.runtime_timeout_sec, period_sec,
            )
            if timeout_check is not None and timeout_check[2]:
                self.protection_latched = True
                evidence = self.frame_recorder.trigger(
                    "runtime_timeout", at=now,
                    last_valid_age_ms=timeout_check[0] * 1000,
                    timeout_ms=self.runtime_timeout_sec * 1000,
                    confirm_checks=self.fault_observer.confirm_count,
                )
        return ImuRuntimeReport(
            last_valid_at=last_received_at,
            timeout_check=timeout_check,
            evidence=evidence,
            protection_latched=self.protection_latched,
            fall_latched=self.fall_protection_latched,
        )

    def can_release(self, now: float) -> bool:
        return (
            self.last_received_at is not None
            and now - self.last_received_at < self.runtime_timeout_sec
            and self.fault_observer.frame_streak == 0
            and self.fault_observer.timeout_streak == 0
        )

    def observe_frame(
        self, *, received_at: float, frame_id: str, stamp,
        quat_xyzw, omega, acceleration,
    ) -> ImuFrameReport:
        stamp_ns = 0 if stamp is None else stamp.sec * 1_000_000_000 + stamp.nanosec
        stamp_status, stamp_age_ms = "unchecked", None
        if self.required:
            stamp_status, stamp_age_ms = self.stamp_observer.observe(stamp)
        last_valid_at = self.last_received_at
        gap_ms = (
            (received_at - last_valid_at) * 1000
            if last_valid_at is not None else None
        )
        assessment = assess_imu_frame(
            quat_xyzw, omega, acceleration,
            stamp_status=stamp_status, require_stamp=self.required,
            receive_gap_ms=gap_ms, timeout_ms=self.runtime_timeout_sec * 1000,
        )
        reasons = assessment.reasons
        accepted = assessment.accepted
        already_protected = self.protection_latched or self.fall_protection_latched
        self.latest_frame_fault = bool(reasons)
        streak, frame_fault_confirmed = self.fault_observer.observe_frame(bool(reasons))
        warn_frame = bool(reasons) and not already_protected and (
            streak < self.fault_observer.confirm_count or frame_fault_confirmed
        )
        followups = self.frame_recorder.record(
            received_at=received_at, frame_id=frame_id, stamp_ns=stamp_ns,
            stamp_status=stamp_status, transport_age_ms=stamp_age_ms,
            valid=assessment.payload_valid, accepted=accepted,
            reason=",".join(reasons) or None, quat_xyzw=quat_xyzw,
            omega=omega, acceleration=acceleration,
        )
        fall_confirmed = False
        warn_orientation = False
        if accepted:
            self.quat_xyzw[:] = quat_xyzw
            self.quat_wxyz[:] = quat_xyzw[3], *quat_xyzw[:3]
            self.omega[:] = omega
            self.linear_acceleration[:] = acceleration
            self.received = True
            self.last_received_at = received_at
            self.fault_observer.reset_timeout()
            if self.orientation_guard_active:
                prior_fall_streak = self.fall_observer.consecutive
                fall_confirmed = self.fall_observer.observe(
                    self.quat_xyzw, stamp_ns=stamp_ns,
                )
                fall_streak = self.fall_observer.consecutive
                warn_orientation = (
                    fall_streak > 0
                    and prior_fall_streak < self.fall_observer.confirm_frames
                )
            else:
                self.fall_observer.reset_streak(clear_pending=True)
                fall_streak = 0
        else:
            self.fall_observer.reset_streak()
            fall_streak = 0
        fall_evidence = None
        if fall_confirmed:
            fall_evidence = self.frame_recorder.trigger(
                "fall_confirmed", at=received_at,
                threshold_deg=60, confirm_frames=self.fall_observer.confirm_frames,
            )
        frame_evidence = None
        if (
            frame_fault_confirmed
            and self.required
            and self.received
            and not self.protection_latched
            and not self.fall_protection_latched
        ):
            self.protection_latched = True
            frame_evidence = self.frame_recorder.trigger(
                "frame_fault_confirmed", at=received_at,
                reasons=tuple(reasons), confirm_frames=self.fault_observer.confirm_count,
                last_valid_age_ms=(
                    (received_at - last_valid_at) * 1000
                    if last_valid_at is not None else None
                ),
            )
        first_frame = accepted and not self.first_received_logged
        if accepted:
            self.first_received_logged = True
        return ImuFrameReport(
            assessment=assessment, stamp_ns=stamp_ns,
            stamp_status=stamp_status, stamp_age_ms=stamp_age_ms,
            gap_ms=gap_ms, streak=streak, warn_frame=warn_frame,
            fall_streak=fall_streak, warn_orientation=warn_orientation,
            fall_confirmed=fall_confirmed, frame_evidence=frame_evidence,
            fall_evidence=fall_evidence, followups=tuple(followups),
            first_frame=first_frame,
        )
