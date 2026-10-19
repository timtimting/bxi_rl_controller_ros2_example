"""Transport-age diagnostics for IMU ROS messages (not device sample age)."""

import time

from bxi_example_py_elf3.framework.mod_api.geometry import orientation_is_unsafe


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

    def reset_streak(self):
        self.consecutive = 0

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
