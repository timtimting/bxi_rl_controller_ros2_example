"""Topology allocation and real Linux thread affinity without hardware IO."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    CpuAffinityPlan, CpuAffinityRole as Role, CpuTopology, configure_current_thread,
)
from bxi_example_py_elf3.framework.platform.communication_trace import (
    CommunicationExecutor, CommunicationTrace,
)


def plan(groups, scores):
    groups = tuple(frozenset(group) for group in groups)
    return CpuAffinityPlan.from_topology(CpuTopology(
        allowed_cpus=frozenset().union(*groups), physical_cores=groups,
        capacity_by_cpu={cpu: score for group, score in zip(groups, scores) for cpu in group},
    ))


def assert_separate_physical_cores(p):
    roles = (Role.CONTROL, Role.COMMUNICATION, Role.COMPUTE, Role.BACKGROUND)
    occupied = {}
    for role in roles:
        occupied[role] = {
            index for index, group in enumerate(p.topology.physical_cores)
            if group & p.roles[role]
        }
    for i, role in enumerate(roles):
        assert p.roles[role] <= p.allowed_cpus
        assert p.roles[role]
        for other in roles[i + 1:]:
            assert not occupied[role] & occupied[other], (role, other)


def test_i7_1360p_keeps_ros_on_a_separate_performance_core():
    p = plan([(0, 1), (2, 3), (4, 5), (6, 7)] + [(i,) for i in range(8, 16)],
             [5000] * 4 + [3700] * 8)
    assert p.roles[Role.CONTROL] == {6}
    assert p.reserved_control_core == {6, 7}
    assert p.roles[Role.COMMUNICATION] == {4, 5}
    assert p.roles[Role.COMPUTE] == {0, 1, 2, 3}
    assert p.roles[Role.BACKGROUND] == set(range(8, 16))
    assert_separate_physical_cores(p)


def test_rk3588_style_capacity_topology_keeps_ros_on_a_big_core():
    p = plan([(i,) for i in range(8)], [512] * 4 + [1024] * 4)
    assert p.roles[Role.CONTROL] == {7}
    assert p.roles[Role.COMMUNICATION] == {6}
    assert p.roles[Role.COMPUTE] == {4, 5}
    assert p.roles[Role.BACKGROUND] == {0, 1, 2, 3}
    assert_separate_physical_cores(p)


def test_slower_turbo_bins_remain_available_for_compute():
    groups = [(i, i + 1) for i in range(0, 16, 2)] + [(i,) for i in range(16, 24)]
    p = plan(groups, [5000] * 4 + [5200, 5200] + [5000] * 2 + [3700] * 8)
    assert p.roles[Role.CONTROL] == {10}
    assert p.roles[Role.COMMUNICATION] == {8, 9}
    assert p.roles[Role.COMPUTE] == set(range(8)) | {12, 13, 14, 15}
    assert p.roles[Role.BACKGROUND] == set(range(16, 24))
    assert_separate_physical_cores(p)


def test_homogeneous_four_core_machine_reserves_background_too():
    p = plan([(i,) for i in range(4)], [1] * 4)
    assert p.roles[Role.CONTROL] == {3}
    assert p.roles[Role.COMMUNICATION] == {2}
    assert p.roles[Role.COMPUTE] == {1}
    assert p.roles[Role.BACKGROUND] == {0}
    assert_separate_physical_cores(p)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_small_cpu_allowance_reports_unavoidable_role_sharing(count):
    p = plan([(i,) for i in range(count)], [1] * count)
    assert "role sharing" in p.describe()
    for role in Role:
        assert p.roles[role]
        assert p.roles[role] <= set(range(count))
    if count >= 2:
        assert not p.roles[Role.CONTROL] & p.roles[Role.COMMUNICATION]
    if count == 3:
        assert not p.roles[Role.COMPUTE] & p.roles[Role.COMMUNICATION]


def test_noncontiguous_allowance_never_allocates_an_unavailable_sibling():
    p = plan([(2, 3), (6,), (11,), (15,)], [5, 5, 4, 3])
    assert_separate_physical_cores(p)
    for cpus in p.roles.values():
        assert cpus <= {2, 3, 6, 11, 15}


def test_bootstrap_inheritance_and_compute_expansion_in_an_isolated_process():
    # Affinity changes remain inside this child; pytest's own allowance is intact.
    code = '''
import json, os
from threading import Thread
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    bootstrap_process_scheduling, configure_current_thread, CpuAffinityRole as R)
p = bootstrap_process_scheduling()
result = {"baseline": sorted(os.sched_getaffinity(0)), "communication": sorted(p.roles[R.COMMUNICATION])}
def worker():
    result["inherited"] = sorted(os.sched_getaffinity(0))
    configure_current_thread(p.roles[R.COMPUTE])
    result["compute_actual"] = sorted(os.sched_getaffinity(0))
    result["compute_expected"] = sorted(p.roles[R.COMPUTE])
t = Thread(target=worker); t.start(); t.join()
print(json.dumps(result))
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["baseline"] == result["communication"] == result["inherited"]
    assert result["compute_actual"] == result["compute_expected"]


def test_executor_workers_rebind_even_when_spawned_from_control_cpu():
    p = CpuAffinityPlan.discover()
    barrier = Barrier(3)
    observed, errors = [], []

    # Supply only the ROS-free constructor state; use real ThreadPoolExecutor
    # workers and real Linux affinity/scheduler calls for the behavior under test.
    def base_init(executor, **kwargs):
        executor._sigint_gc = None
        executor._executor = ThreadPoolExecutor(max_workers=kwargs["num_threads"])

    with patch("rclpy.executors.MultiThreadedExecutor.__init__", base_init):
        executor = CommunicationExecutor(
            None, None, num_threads=3, cpu_affinity=p.roles[Role.COMMUNICATION]
        )

    def callback():
        barrier.wait(timeout=3)
        return set(os.sched_getaffinity(0)), os.sched_getscheduler(0)

    def control():
        try:
            configure_current_thread(p.roles[Role.CONTROL])
            futures = [executor._executor.submit(callback) for _ in range(3)]
            observed.extend(f.result(timeout=4) for f in futures)
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=control)
    try:
        worker.start()
        worker.join(timeout=6)
        assert not worker.is_alive()
        assert not errors
        assert len(observed) == 3
        assert all(cpus == p.roles[Role.COMMUNICATION] and policy == os.SCHED_OTHER
                   for cpus, policy in observed)
    finally:
        executor._executor.shutdown(wait=True)


def test_watchdog_explicitly_uses_background_cpus():
    p = CpuAffinityPlan.discover()
    trace, ready, records = CommunicationTrace(), Event(), []

    def log(_severity, text):
        data = json.loads(text.split(" ", 1)[1])
        records.append(data)
        if data["event"] == "started":
            ready.set()

    try:
        trace.start(log, cpu_affinity=p.roles[Role.BACKGROUND])
        assert ready.wait(2)
        started = next(r for r in records if r["event"] == "started")
        assert set(started["cpu_affinity"]) == p.roles[Role.BACKGROUND]
        assert started["scheduling_policy"] == os.SCHED_OTHER
    finally:
        trace.stop()


def test_maintenance_binds_background_before_any_work():
    from bxi_example_py_elf3.framework.platform.runtime import RobotControlRuntime

    p = CpuAffinityPlan.discover()
    runtime = RobotControlRuntime.__new__(RobotControlRuntime)
    runtime.cpu_affinity_plan = p
    runtime.config = SimpleNamespace(maintenance_hz=5, statistics_interval_sec=60)
    runtime._stop_event = Event()
    runtime._stop_event.set()
    observed = []
    runtime._log = lambda *_args: observed.append(
        (set(os.sched_getaffinity(0)), os.sched_getscheduler(0))
    )
    worker = Thread(target=runtime._maintenance_loop)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert observed == [(p.roles[Role.BACKGROUND], os.SCHED_OTHER)]
