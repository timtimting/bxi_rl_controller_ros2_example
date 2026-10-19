"""Bounded IMU frame evidence for one-shot protection logs."""

from collections import deque
from dataclasses import dataclass, field
import json
import math

from bxi_example_py_elf3.framework.mod_api.geometry import quaternion_to_euler_array


@dataclass(frozen=True, slots=True)
class ImuFrameRecord:
    sequence: int
    received_at: float
    frame_id: str
    stamp_ns: int
    stamp_status: str
    transport_age_ms: float | None
    valid: bool
    accepted: bool
    reason: str | None
    quat_xyzw: tuple[float, float, float, float]
    omega: tuple[float, float, float]
    acceleration: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ImuEvidence:
    incident_id: int
    trigger: str
    phase: str
    triggered_at: float
    details: dict[str, object]
    frames: tuple[ImuFrameRecord, ...]
    last_valid_before_window: ImuFrameRecord | None = None


@dataclass(slots=True)
class _PendingEvidence:
    incident_id: int
    trigger: str
    triggered_at: float
    frames: list[ImuFrameRecord] = field(default_factory=list)


class ImuFrameRecorder:
    """Keep recent samples and capture a few later samples for each incident."""

    def __init__(self, pre_frames=5, post_frames=3):
        if pre_frames < 1 or post_frames < 1:
            raise ValueError("IMU evidence frame counts must be positive")
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self._recent = deque(maxlen=pre_frames)
        self._pending: list[_PendingEvidence] = []
        self._last_valid: ImuFrameRecord | None = None
        self._sequence = 0
        self._incident_id = 0

    def record(
        self, *, received_at, frame_id, stamp_ns, stamp_status, transport_age_ms,
        valid, reason, quat_xyzw, omega, acceleration, accepted=None,
    ) -> tuple[ImuEvidence, ...]:
        if accepted is None:
            accepted = valid
        self._sequence += 1
        frame = ImuFrameRecord(
            sequence=self._sequence,
            received_at=received_at,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            stamp_status=stamp_status,
            transport_age_ms=transport_age_ms,
            valid=valid,
            accepted=accepted,
            reason=reason,
            quat_xyzw=tuple(quat_xyzw),
            omega=tuple(omega),
            acceleration=tuple(acceleration),
        )
        self._recent.append(frame)
        if accepted:
            self._last_valid = frame
        completed = []
        pending = []
        for capture in self._pending:
            capture.frames.append(frame)
            if len(capture.frames) >= self.post_frames:
                completed.append(ImuEvidence(
                    capture.incident_id, capture.trigger, "after",
                    capture.triggered_at, {}, tuple(capture.frames),
                ))
            else:
                pending.append(capture)
        self._pending = pending
        return tuple(completed)

    def trigger(self, trigger: str, *, at: float, **details: object) -> ImuEvidence:
        self._incident_id += 1
        capture = _PendingEvidence(self._incident_id, trigger, at)
        self._pending.append(capture)
        self._pending = self._pending[-3:]
        details["last_any_frame_age_ms"] = (
            round((at - self._recent[-1].received_at) * 1000, 1)
            if self._recent else None
        )
        return ImuEvidence(
            capture.incident_id, trigger, "before", at,
            dict(details), tuple(self._recent),
            self._last_valid
            if self._last_valid is not None
            and self._last_valid.sequence < self._recent[0].sequence
            else None,
        )


def format_imu_evidence(evidence: ImuEvidence) -> str:
    def number(value, digits=4):
        if value is None:
            return None
        value = float(value)
        return round(value, digits) if math.isfinite(value) else str(value)

    def frame_payload(frame):
        angles = None
        if frame.valid:
            roll, pitch, _ = quaternion_to_euler_array(frame.quat_xyzw)
            angles = [number(math.degrees(roll), 1), number(math.degrees(pitch), 1)]
        return {
            "rx_seq": frame.sequence,
            "relative_ms": number((frame.received_at - evidence.triggered_at) * 1000, 1),
            "frame_id": frame.frame_id,
            "stamp_ns": frame.stamp_ns,
            "stamp_status": frame.stamp_status,
            "transport_age_ms": number(frame.transport_age_ms, 1),
            "payload_valid": frame.valid,
            "accepted": frame.accepted,
            "reason": frame.reason,
            "roll_pitch_deg": angles,
            "quat_xyzw": [number(v, 5) for v in frame.quat_xyzw],
            "quat_norm": number(math.sqrt(sum(v * v for v in frame.quat_xyzw)), 5),
            "omega_rad_s": [number(v) for v in frame.omega],
            "accel_m_s2": [number(v) for v in frame.acceleration],
        }

    return json.dumps({
        "incident_id": evidence.incident_id,
        "trigger": evidence.trigger,
        "phase": evidence.phase,
        "details": evidence.details,
        "frames": [frame_payload(frame) for frame in evidence.frames],
        "last_valid_before_window": (
            frame_payload(evidence.last_valid_before_window)
            if evidence.last_valid_before_window is not None else None
        ),
    }, separators=(",", ":"), allow_nan=False)
