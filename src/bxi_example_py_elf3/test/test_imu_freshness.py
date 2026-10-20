from types import SimpleNamespace
import json

import numpy as np
import pytest

from bxi_example_py_elf3.framework.platform.input_safety.imu_freshness import (
    ImuFallObserver,
    ImuFaultObserver,
    ImuStampObserver,
    assess_imu_frame,
)
from bxi_example_py_elf3.framework.platform.input_safety.imu_diagnostics import (
    ImuFrameRecorder,
    format_imu_evidence,
)
from bxi_example_py_elf3.framework.mod_api import OrientationGuardedState
from bxi_example_py_elf3.framework.joints import JointLayout, JointStateBuffer
from bxi_example_py_elf3.framework.platform.input_safety.imu_monitor import ImuSafetyMonitor


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


@pytest.mark.parametrize(
    ("stamp_status", "quat", "gap_ms", "expected_reason", "accepted"),
    [
        ("fresh", (0.0, 0.0, 0.0, 1.0), 5.0, None, True),
        ("stale", (0.0, 0.0, 0.0, 1.0), 5.0, "stamp_stale", False),
        ("fresh", (0.0, 0.0, 0.0, 0.5), 5.0,
         "quaternion_norm_out_of_range", False),
        ("fresh", (0.0, 0.0, 0.0, 1.0), 110.0, "receive_gap", True),
    ],
)
def test_imu_frame_assessment(stamp_status, quat, gap_ms, expected_reason, accepted):
    assessment = assess_imu_frame(
        quat, (0.0, 0.0, 0.0), (0.0, 0.0, 9.81),
        stamp_status=stamp_status, require_stamp=True,
        receive_gap_ms=gap_ms, timeout_ms=100.0,
    )
    assert assessment.accepted is accepted
    assert (assessment.reasons[0] if assessment.reasons else None) == expected_reason


def test_nonfinite_imu_vector_is_rejected():
    assessment = assess_imu_frame(
        (0.0, 0.0, 0.0, 1.0), (float("nan"), 0.0, 0.0),
        (0.0, 0.0, 9.81), stamp_status="fresh", require_stamp=True,
        receive_gap_ms=5.0, timeout_ms=100.0,
    )
    assert not assessment.payload_valid
    assert not assessment.accepted
    assert assessment.reasons == ("nonfinite_angular_velocity",)


def test_frame_faults_warn_from_first_frame_and_confirm_on_fifth():
    observer = ImuFaultObserver(confirm_count=5)
    for streak in range(1, 5):
        assert observer.observe_frame(True) == (streak, False)
    assert observer.observe_frame(False) == (0, False)
    for streak in range(1, 5):
        assert observer.observe_frame(True) == (streak, False)
    assert observer.observe_frame(True) == (5, True)
    assert observer.observe_frame(True) == (5, False)


def test_receive_timeout_requires_five_separate_checks_and_resets_on_recovery():
    observer = ImuFaultObserver(confirm_count=5)
    assert observer.check_timeout(1.09, 1.0, 0.1, 0.02) is None
    for streak, now in enumerate((1.101, 1.122, 1.143, 1.164, 1.185), 1):
        age, observed, confirmed = observer.check_timeout(now, 1.0, 0.1, 0.02)
        assert age >= 0.1
        assert observed == streak
        assert confirmed is (streak == 5)
        assert observer.check_timeout(now + 0.001, 1.0, 0.1, 0.02) is None
    assert observer.check_timeout(1.2, 1.19, 0.1, 0.02) is None
    assert observer.check_timeout(1.291, 1.19, 0.1, 0.02)[1:] == (1, False)
    observer.reset_timeout()
    assert observer.check_timeout(1.312, 1.19, 0.1, 0.02)[1:] == (1, False)


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


def test_leaving_guarded_state_discards_pending_fall():
    observer = ImuFallObserver(confirm_frames=2)
    assert not observer.observe(orientation(65), stamp_ns=1)
    assert observer.observe(orientation(65), stamp_ns=2)
    observer.reset_streak(clear_pending=True)
    assert not observer.snapshot_confirmed()
    assert not observer.observe(orientation(65), stamp_ns=3)
    assert observer.observe(orientation(65), stamp_ns=4)


def test_mod_orientation_guard_requests_zero_torque_only_after_confirmation():
    from unittest.mock import Mock

    class Motion(OrientationGuardedState):
        def on_update(self, ctx, dt):
            pass

    state = Motion(
        "com.bxi.basic_actions/normal", 1,
        protection_state="com.bxi.basic_actions/zero_torque",
    )
    ctx = SimpleNamespace(
        current_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        is_orientation_unsafe=Mock(return_value=False),
        request_state=Mock(return_value=True),
    )
    assert state.orientation_guard_enabled
    assert not state.orientation_limit_reached(ctx)
    ctx.request_state.assert_not_called()

    ctx.is_orientation_unsafe.return_value = True
    assert state.orientation_limit_reached(ctx)
    ctx.request_state.assert_called_once_with(
        "com.bxi.basic_actions/zero_torque",
        trigger="imu_fall_confirmed",
        force=True,
    )


def test_mod_orientation_guard_requires_explicit_safe_state():
    class Motion(OrientationGuardedState):
        def on_update(self, ctx, dt):
            pass

    with pytest.raises(ValueError, match="protection_state"):
        Motion("com.example/motion", 1, protection_state="")


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


def test_stamp_rejection_preserves_payload_validity_but_not_last_accepted_frame():
    recorder = ImuFrameRecorder()
    record_imu_evidence_frame(recorder, 1)
    for sequence in range(2, 8):
        recorder.record(
            received_at=sequence * 0.01, frame_id="imu_link", stamp_ns=1,
            stamp_status="stale", transport_age_ms=150.0,
            valid=True, accepted=False, reason="stamp_stale",
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            omega=(0.0, 0.0, 0.0), acceleration=(0.0, 0.0, 9.81),
        )
    evidence = json.loads(format_imu_evidence(
        recorder.trigger("frame_fault_confirmed", at=0.08)
    ))
    assert evidence["frames"][-1]["payload_valid"] is True
    assert evidence["frames"][-1]["accepted"] is False
    assert evidence["last_valid_before_window"]["rx_seq"] == 1


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


def monitor(*, required=True):
    return ImuSafetyMonitor(
        required=required, startup_timeout_sec=1.0, runtime_timeout_sec=0.1,
        guard_kp=20.0, guard_kd=1.0,
    )


def monitor_frame(imu, at, *, stamp_ns=1, quat=(0.0, 0.0, 0.0, 1.0)):
    return imu.observe_frame(
        received_at=at, frame_id="elf3",
        stamp=SimpleNamespace(sec=1, nanosec=stamp_ns),
        quat_xyzw=quat, omega=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 9.81),
    )


def test_monitor_startup_timeout_is_reported_once_without_valid_frame():
    imu = monitor()
    assert imu.startup_check(0.0).warn_waiting
    report = imu.startup_check(1.1)
    assert report.evidence.details["timeout_ms"] == 1000.0
    assert imu.startup_check(1.2).evidence is None
    imu.stamp_observer.observe = lambda _stamp: ("fresh", 1.0)
    monitor_frame(imu, 1.3)
    assert imu.startup_check(1.4) is None


def test_monitor_invalid_stamps_pause_immediately_and_latch_after_five_frames():
    imu = monitor()
    imu.stamp_observer.observe = lambda _stamp: ("fresh", 1.0)
    monitor_frame(imu, 1.0)
    imu.stamp_observer.observe = lambda _stamp: ("stale", 150.0)
    for count in range(1, 6):
        report = monitor_frame(imu, 1.0 + count * 0.01, stamp_ns=count + 1)
        assert report.assessment.reasons == ("stamp_stale",)
        assert imu.last_received_at == 1.0
        assert imu.protection_latched is (count == 5)
        assert imu.block_reason(1.0 + count * 0.01) is not None
    assert report.frame_evidence.details["confirm_frames"] == 5
    assert not imu.can_release(1.06)


def test_monitor_timeout_confirms_across_five_control_checks():
    imu = monitor()
    imu.stamp_observer.observe = lambda _stamp: ("fresh", 1.0)
    monitor_frame(imu, 1.0)
    for count, now in enumerate((1.2, 1.221, 1.242, 1.263, 1.284), 1):
        report = imu.check_runtime_timeout(now, 0.02)
        assert report.timeout_check[1] == count
        assert report.protection_latched is (count == 5)
    assert report.evidence.details["last_valid_age_ms"] == pytest.approx(284.0)
    assert imu.block_reason(1.285) == "data_protection"


def test_monitor_orientation_confirmation_only_while_guard_is_active():
    imu = monitor(required=False)
    roll_half_angle = np.deg2rad(65) / 2
    quat = (np.sin(roll_half_angle), 0.0, 0.0, np.cos(roll_half_angle))
    for count in range(6):
        monitor_frame(imu, 1.0 + count * 0.01, stamp_ns=count + 1, quat=quat)
    assert imu.fall_observer.consecutive == 0
    imu.set_orientation_guard_active(True)
    for count in range(5):
        report = monitor_frame(imu, 1.1 + count * 0.01, stamp_ns=count + 7, quat=quat)
        assert report.fall_confirmed is (count == 4)
        assert imu.block_reason(1.1 + count * 0.01) == "orientation_limit"
    assert imu.fall_observer.snapshot_confirmed()
    imu.set_orientation_guard_active(False)
    assert not imu.fall_observer.snapshot_confirmed()
    assert imu.block_reason(1.2) is None


def test_monitor_guard_frame_never_energizes_zero_torque():
    imu = monitor()
    joints = JointStateBuffer(JointLayout(("test_joint",)), dtype=np.float64)
    joints.update([0.3], [0.0])
    first = imu.safety_frame(joints.view, "com.bxi.basic_actions/normal")
    assert first.qpos.tolist() == pytest.approx([0.3])
    assert first.kp.tolist() == [20.0]
    joints.update([0.5], [0.0])
    second = imu.safety_frame(joints.view, "com.bxi.basic_actions/zero_torque")
    assert second is first
    assert second.qpos.tolist() == pytest.approx([0.3])
    assert second.kp.tolist() == [0.0]
    imu.set_inference_blocked(False)
    assert imu.safety_frame(
        joints.view, "com.bxi.basic_actions/normal"
    ).qpos.tolist() == pytest.approx([0.5])
