from types import SimpleNamespace
import json

import numpy as np
import pytest

from bxi_example_py_elf3.imu_freshness import ImuFallObserver, ImuStampObserver
from bxi_example_py_elf3.imu_diagnostics import ImuFrameRecorder, format_imu_evidence


def stamp(ns):
    return SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def test_transport_stamp_diagnostics():
    observer = ImuStampObserver(stale_sec=0.1, future_sec=0.1)
    assert observer.observe(stamp(0), 2_000_000_000) == ("missing", None)
    assert observer.observe(stamp(2_000_000_000), 2_001_000_000)[0] == "fresh"
    assert observer.observe(stamp(2_000_000_000), 2_002_000_000)[0] == "repeated"
    assert observer.observe(stamp(1_999_000_000), 2_003_000_000)[0] == "reversed"
    assert observer.observe(stamp(2_010_000_000), 2_200_000_000)[0] == "stale"
    assert observer.observe(stamp(2_500_000_000), 2_300_000_000)[0] == "future"


def orientation(roll_degrees):
    half_angle = np.deg2rad(roll_degrees) / 2
    return np.array([np.sin(half_angle), 0.0, 0.0, np.cos(half_angle)])


@pytest.mark.parametrize("frames", [0, 1, True, 1.5])
def test_fall_confirmation_requires_multiple_frames(frames):
    with pytest.raises(ValueError):
        ImuFallObserver(confirm_frames=frames)


def test_fall_requires_five_consecutive_unsafe_imu_messages():
    observer = ImuFallObserver()
    unsafe = orientation(65)
    safe = orientation(0)
    for stamp_ns in range(1, 5):
        assert not observer.observe(unsafe, stamp_ns=stamp_ns)
        assert not observer.snapshot_confirmed()
    assert not observer.observe(safe, stamp_ns=5)
    for stamp_ns in range(6, 10):
        assert not observer.observe(unsafe, stamp_ns=stamp_ns)
    assert observer.observe(unsafe, stamp_ns=10)
    assert observer.snapshot_confirmed()
    assert observer.snapshot_confirmed()  # Sustained fall remains unsafe.
    assert not observer.observe(unsafe, stamp_ns=11)  # Log only on threshold crossing.
    assert not observer.observe(safe, stamp_ns=12)
    assert not observer.snapshot_confirmed()


def test_fall_streak_restarts_after_invalid_or_replayed_imu_message():
    observer = ImuFallObserver(confirm_frames=3)
    unsafe = orientation(65)
    assert not observer.observe(unsafe, stamp_ns=10)
    assert not observer.observe(unsafe, stamp_ns=11)
    assert not observer.observe(unsafe, stamp_ns=11)
    assert not observer.observe(unsafe, stamp_ns=10)
    assert not observer.observe(unsafe, stamp_ns=12)
    observer.reset_streak()  # The IMU callback rejected an invalid quaternion.
    assert not observer.observe(unsafe, stamp_ns=13)
    assert not observer.observe(unsafe, stamp_ns=14)
    assert observer.observe(unsafe, stamp_ns=15)


def test_confirmed_fall_is_not_lost_between_control_snapshots():
    observer = ImuFallObserver(confirm_frames=2)
    assert not observer.observe(orientation(65), stamp_ns=1)
    assert observer.observe(orientation(65), stamp_ns=2)
    assert not observer.observe(orientation(0), stamp_ns=3)
    assert observer.snapshot_confirmed()
    assert not observer.snapshot_confirmed()


def record_imu_evidence_frame(recorder, sequence, *, valid=True, quat=None, reason=None):
    return recorder.record(
        received_at=sequence * 0.01,
        frame_id="imu_link",
        stamp_ns=sequence * 10_000_000,
        stamp_status="fresh",
        transport_age_ms=2.0,
        valid=valid,
        reason=reason,
        quat_xyzw=quat if quat is not None else (0.0, 0.0, 0.0, 1.0),
        omega=(0.1, 0.2, 0.3),
        acceleration=(0.0, 0.0, 9.81),
    )


def test_imu_evidence_keeps_five_previous_and_three_following_frames():
    recorder = ImuFrameRecorder()
    for sequence in range(1, 8):
        assert not record_imu_evidence_frame(recorder, sequence)
    before = json.loads(format_imu_evidence(
        recorder.trigger("runtime_timeout", at=0.08, last_valid_age_ms=10.0)
    ))
    assert before["trigger"] == "runtime_timeout"
    assert before["phase"] == "before"
    assert before["details"]["last_valid_age_ms"] == 10.0
    assert before["details"]["last_any_frame_age_ms"] == 10.0
    assert [frame["rx_seq"] for frame in before["frames"]] == [3, 4, 5, 6, 7]
    assert before["frames"][-1]["relative_ms"] == -10.0
    assert before["frames"][-1]["frame_id"] == "imu_link"
    assert before["frames"][-1]["roll_pitch_deg"] == [0.0, 0.0]
    assert before["frames"][-1]["quat_norm"] == 1.0
    assert before["last_valid_before_window"] is None
    assert not record_imu_evidence_frame(recorder, 8)
    assert not record_imu_evidence_frame(recorder, 9)
    after, = record_imu_evidence_frame(recorder, 10)
    after = json.loads(format_imu_evidence(after))
    assert after["incident_id"] == before["incident_id"]
    assert after["phase"] == "after"
    assert [frame["rx_seq"] for frame in after["frames"]] == [8, 9, 10]
    assert not record_imu_evidence_frame(recorder, 11)


def test_imu_evidence_preserves_nonfinite_invalid_input_without_invalid_json():
    recorder = ImuFrameRecorder()
    record_imu_evidence_frame(
        recorder, 1, valid=False,
        quat=(float("nan"), float("inf"), 0.0, 1.0),
        reason="nonfinite_quaternion",
    )
    evidence = json.loads(format_imu_evidence(
        recorder.trigger("startup_timeout", at=0.02)
    ))
    frame = evidence["frames"][0]
    assert frame["payload_valid"] is False
    assert frame["reason"] == "nonfinite_quaternion"
    assert frame["roll_pitch_deg"] is None
    assert frame["quat_xyzw"][:2] == ["nan", "inf"]


def test_imu_timeout_without_any_frame_reports_empty_history_then_recovery():
    recorder = ImuFrameRecorder()
    before = recorder.trigger("startup_timeout", at=1.0)
    payload = json.loads(format_imu_evidence(before))
    assert payload["frames"] == []
    assert payload["details"]["last_any_frame_age_ms"] is None
    for sequence in range(1, 3):
        assert not record_imu_evidence_frame(recorder, sequence)
    after, = record_imu_evidence_frame(recorder, 3)
    assert after.incident_id == before.incident_id
    assert [frame.sequence for frame in after.frames] == [1, 2, 3]


def test_imu_evidence_keeps_last_valid_frame_outside_invalid_window():
    recorder = ImuFrameRecorder()
    record_imu_evidence_frame(recorder, 1)
    for sequence in range(2, 9):
        record_imu_evidence_frame(
            recorder, sequence, valid=False, reason="nonfinite_angular_velocity"
        )
    evidence = json.loads(format_imu_evidence(
        recorder.trigger("runtime_timeout", at=0.2)
    ))
    assert [frame["rx_seq"] for frame in evidence["frames"]] == [4, 5, 6, 7, 8]
    assert all(not frame["payload_valid"] for frame in evidence["frames"])
    assert evidence["last_valid_before_window"]["rx_seq"] == 1
    assert evidence["last_valid_before_window"]["payload_valid"]
