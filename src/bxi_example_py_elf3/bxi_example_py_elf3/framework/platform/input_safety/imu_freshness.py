"""Transport-age diagnostics for IMU ROS messages (not device sample age)."""

from dataclasses import dataclass
import math
import time

from bxi_example_py_elf3.framework.mod_api.geometry import orientation_is_unsafe


@dataclass(frozen=True)
class ImuFrameAssessment:
    quaternion_norm: float
    payload_valid: bool
    accepted: bool
    reasons: tuple[str, ...]


def assess_imu_frame(
    quat_xyzw, omega, acceleration, *, stamp_status, require_stamp,
    receive_gap_ms, timeout_ms,
):
    quaternion_norm = math.sqrt(sum(value * value for value in quat_xyzw))
    quaternion_finite = all(math.isfinite(value) for value in quat_xyzw)
    omega_finite = all(math.isfinite(value) for value in omega)
    acceleration_finite = all(math.isfinite(value) for value in acceleration)
    quaternion_valid = quaternion_finite and 0.9 <= quaternion_norm <= 1.1
    payload_valid = quaternion_valid and omega_finite and acceleration_finite
    reasons = []
    if require_stamp and stamp_status != "fresh":
        reasons.append(f"stamp_{stamp_status}")
    if not quaternion_finite:
        reasons.append("nonfinite_quaternion")
    elif not quaternion_valid:
        reasons.append("quaternion_norm_out_of_range")
    if not omega_finite:
        reasons.append("nonfinite_angular_velocity")
    if not acceleration_finite:
        reasons.append("nonfinite_linear_acceleration")
    if require_stamp and receive_gap_ms is not None and receive_gap_ms >= timeout_ms:
        reasons.append("receive_gap")
    return ImuFrameAssessment(
        quaternion_norm=quaternion_norm,
        payload_valid=payload_valid,
        accepted=payload_valid and (not require_stamp or stamp_status == "fresh"),
        reasons=tuple(reasons),
    )


class ImuStampObserver:
    def __init__(self, stale_sec=0.1, future_sec=0.1):
        self.stale_ns = int(stale_sec * 1e9)
        self.future_ns = int(future_sec * 1e9)
        self.last_stamp_ns = None

    def observe(self, stamp, receive_ns=None):
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns <= 0:
            return "missing", None
        if receive_ns is None:
            receive_ns = time.time_ns()
        age_ms = (receive_ns - stamp_ns) / 1e6
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            status = "repeated" if stamp_ns == self.last_stamp_ns else "reversed"
        elif receive_ns - stamp_ns > self.stale_ns:
            status = "stale"
        elif stamp_ns - receive_ns > self.future_ns:
            status = "future"
        else:
            status = "fresh"
        self.last_stamp_ns = stamp_ns
        return status, age_ms


class ImuFaultObserver:
    """Confirm bad frames or a sustained receive gap without counting spin cycles."""

    def __init__(self, confirm_count=5):
        if (
            not isinstance(confirm_count, int)
            or isinstance(confirm_count, bool)
            or confirm_count < 2
        ):
            raise ValueError("IMU fault confirmation requires at least two checks")
        self.confirm_count = confirm_count
        self.frame_streak = 0
        self.timeout_streak = 0
        self._next_timeout_check_at = None

    def observe_frame(self, faulted):
        previous = self.frame_streak
        self.frame_streak = (
            min(previous + 1, self.confirm_count) if faulted else 0
        )
        confirmed = self.frame_streak == self.confirm_count and previous < self.confirm_count
        return self.frame_streak, confirmed

    def reset_timeout(self):
        self.timeout_streak = 0
        self._next_timeout_check_at = None

    def check_timeout(self, now, last_valid_at, timeout_sec, check_period_sec):
        if last_valid_at is None or now - last_valid_at < timeout_sec:
            self.reset_timeout()
            return None
        if self._next_timeout_check_at is not None and now < self._next_timeout_check_at:
            return None
        previous = self.timeout_streak
        self.timeout_streak = min(previous + 1, self.confirm_count)
        self._next_timeout_check_at = now + check_period_sec
        return (
            now - last_valid_at,
            self.timeout_streak,
            self.timeout_streak == self.confirm_count and previous < self.confirm_count,
        )


class ImuFallObserver:
    """Confirm unsafe orientation from consecutive accepted IMU messages."""

    def __init__(self, confirm_frames=5):
        if (
            not isinstance(confirm_frames, int)
            or isinstance(confirm_frames, bool)
            or confirm_frames < 2
        ):
            raise ValueError("fall confirmation requires at least two IMU frames")
        self.confirm_frames = confirm_frames
        self.consecutive = 0
        self.last_stamp_ns = None
        self._pending = False

    def reset_streak(self, *, clear_pending=False):
        self.consecutive = 0
        if clear_pending:
            self._pending = False

    def observe(self, quat_xyzw, *, stamp_ns=0):
        if stamp_ns > 0:
            if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
                self.reset_streak()
                return False
            self.last_stamp_ns = stamp_ns
        else:
            if self.last_stamp_ns is not None:
                self.reset_streak()
            self.last_stamp_ns = None

        if not orientation_is_unsafe(quat_xyzw):
            self.reset_streak()
            return False
        was_confirmed = self.consecutive >= self.confirm_frames
        self.consecutive = min(self.consecutive + 1, self.confirm_frames)
        if self.consecutive == self.confirm_frames and not was_confirmed:
            self._pending = True
            return True
        return False

    def snapshot_confirmed(self):
        confirmed = self._pending or self.consecutive >= self.confirm_frames
        self._pending = False
        return confirmed
