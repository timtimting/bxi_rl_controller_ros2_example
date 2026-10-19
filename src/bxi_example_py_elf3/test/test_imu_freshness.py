from types import SimpleNamespace

import numpy as np
import pytest

from bxi_example_py_elf3.imu_freshness import ImuFallObserver, ImuStampObserver


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
