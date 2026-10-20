"""Remote message validation and receive watchdog; caller provides locking."""

from __future__ import annotations

import math
from collections.abc import Sequence


class RemoteInputMonitor:
    """Keep stale or invalid commands from refreshing the receive watchdog.

    Receive age uses a monotonic clock. Source age is checked separately against
    the ROS clock, when a publisher supplies a nonzero header timestamp.
    """

    def __init__(self, timeout_sec: float = 0.3, max_age_sec: float = 0.3):
        for name, value in (("timeout_sec", timeout_sec), ("max_age_sec", max_age_sec)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"remote {name} must be finite and positive")
        self.timeout_sec = timeout_sec
        self.max_age_ns = int(max_age_sec * 1_000_000_000)
        self.last_received_at: float | None = None
        self.last_stamp_ns: int | None = None
        self.source_age_ms: float | None = None
        self.accepted = 0
        self.rejected = 0
        self.timeouts = 0
        self.timed_out = False
        self.last_rejection: str | None = None

    def observe(
        self,
        velocity: Sequence[float],
        *,
        stamp_ns: int,
        received_at: float,
        ros_now_ns: int,
    ) -> str:
        reason = None
        age_ns = ros_now_ns - stamp_ns
        if len(velocity) != 3 or not all(math.isfinite(value) for value in velocity):
            reason = "invalid_velocity"
        elif stamp_ns > 0:
            if age_ns > self.max_age_ns:
                reason = "stale_stamp"
            elif age_ns < -self.max_age_ns:
                reason = "future_stamp"
            elif self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
                reason = "nonincreasing_stamp"
        if reason is not None:
            self.rejected += 1
            self.last_rejection = reason
            return reason

        recovering = self.last_received_at is not None and not self.is_fresh(received_at)
        self.last_received_at = received_at
        self.last_stamp_ns = stamp_ns if stamp_ns > 0 else None
        self.source_age_ms = age_ns / 1_000_000 if stamp_ns > 0 else None
        self.accepted += 1
        self.timed_out = False
        return "recovered" if recovering else "fresh"

    def is_fresh(self, now: float) -> bool:
        return (
            self.last_received_at is not None
            and now - self.last_received_at < self.timeout_sec
        )

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.last_rejection = reason

    def expire(self, now: float) -> bool:
        """Return True only on the first control cycle after a receive timeout."""
        if self.last_received_at is None or self.is_fresh(now) or self.timed_out:
            return False
        self.timed_out = True
        self.timeouts += 1
        return True

    def snapshot(self, now: float) -> dict[str, object]:
        return {
            "fresh": self.is_fresh(now),
            "receive_age_ms": (
                None if self.last_received_at is None
                else max(0.0, now - self.last_received_at) * 1000
            ),
            "source_age_ms_at_receive": self.source_age_ms,
            "source_stamp_available": self.last_stamp_ns is not None,
            "timeout_sec": self.timeout_sec,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "timeouts": self.timeouts,
            "last_rejection": self.last_rejection,
        }
