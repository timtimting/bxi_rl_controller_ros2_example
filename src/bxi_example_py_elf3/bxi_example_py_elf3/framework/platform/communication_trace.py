"""Bounded, observational communication flight recorder for CPython/Humble.

Hot paths only replace immutable samples or append rare incidents. The watchdog
never takes platform/framework locks. Formatting, graph queries and stack walks
run on its own thread. A publish return is NOT proof of remote delivery.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import platform
import socket
import sys
from threading import Event, Thread, get_ident, get_native_id
import time

from rclpy.executors import MultiThreadedExecutor
from .cpu_affinity import configure_current_thread


@dataclass(frozen=True, slots=True)
class StreamSample:
    count: int
    mono_ns: int
    stamp_ns: int
    source_age_ms: float | None
    dds_to_callback_ms: float | None
    source_to_dds_ms: float | None


class CommunicationTrace:
    """One writer per stream/lane; immutable snapshots exchanged under the GIL."""

    def __init__(self, timeout_sec=0.2):
        self.timeout_ns = int(timeout_sec * 1e9)
        if self.timeout_ns <= 0:
            raise ValueError("communication trace timeout must be positive")
        self._streams = {}
        self._active = {}
        self._incidents = deque(maxlen=32)
        self._history = deque(maxlen=10)
        self._pending = deque(maxlen=32)
        self._last_fault = False
        self._last_report_ns = 0
        self._last_poll = None
        self._stop = Event()
        self._thread = None
        self._log = None
        self._context = None
        self._cpu_affinity = None
        self._identity = {"pid": os.getpid(), "host": socket.gethostname(),
                          "architecture": platform.machine()}
        try:
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
                self._identity["boot_id"] = f.read().strip()
        except OSError:
            pass

    def enter(self, stage, *, lane="control", stamp_ns=0):
        self._active[lane] = (stage, time.monotonic_ns(), get_ident(), stamp_ns, get_native_id())

    def leave(self, lane):
        self._active.pop(lane, None)

    def record(self, stream, *, stamp_ns=0, ros_now_ns=0, info=None):
        now = time.monotonic_ns()
        wall = time.time_ns()
        previous = self._streams.get(stream)
        source = int((info or {}).get("source_timestamp", 0))
        received = int((info or {}).get("received_timestamp", 0))
        sample = StreamSample(
            1 if previous is None else previous.count + 1,
            now, stamp_ns,
            (ros_now_ns - stamp_ns) / 1e6 if stamp_ns and ros_now_ns else None,
            (wall - received) / 1e6 if received > 0 else None,
            (received - source) / 1e6 if source > 0 and received > 0 else None,
        )
        self._streams[stream] = sample
        # Save transient gaps too: a callback may recover before the next poll.
        if previous is not None and now - previous.mono_ns > self.timeout_ns:
            self._incidents.append({
                "kind": "stream_gap", "stream": stream, "mono_ns": now,
                "gap_ms": (now - previous.mono_ns) / 1e6,
                "previous_stamp_ns": previous.stamp_ns, "stamp_ns": stamp_ns,
            })
        for label, value in (
            ("source_age_ms", sample.source_age_ms),
            ("dds_to_callback_ms", sample.dds_to_callback_ms),
            ("source_to_dds_ms", sample.source_to_dds_ms),
        ):
            if value is not None and abs(value) > self.timeout_ns / 1e6:
                self._incidents.append({
                    "kind": label, "stream": stream, "mono_ns": now,
                    "value": value, "stamp_ns": stamp_ns,
                })

    def start(self, log, context=None, *, cpu_affinity=None):
        if self._thread is not None:
            return
        self._log, self._context = log, context
        self._cpu_affinity = cpu_affinity
        self._stop.clear()
        self._thread = Thread(target=self._run, name="bxi-communication-watchdog", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self):
        if self._cpu_affinity is not None:
            try:
                configure_current_thread(self._cpu_affinity, realtime_priority=0)
            except (AttributeError, OSError) as exc:
                self._emit("warning", {"event": "affinity_error", "error": str(exc)})
        self._emit("info", {
            "event": "started", "threshold_ms": self.timeout_ns / 1e6,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "scheduling_policy": os.sched_getscheduler(0),
        })
        while not self._stop.wait(0.05):
            try:
                self.poll()
            except Exception as exc:
                # Diagnostics must never shut down control or silently disappear.
                self._emit("warning", {"event": "recorder_error", "error": str(exc)})
                self._stop.wait(1.0)

    def poll(self):
        now, wall = time.monotonic_ns(), time.time_ns()
        previous_poll = self._last_poll
        self._last_poll = (now, wall)
        if previous_poll is not None:
            elapsed = now - previous_poll[0]
            clock_delta = wall - previous_poll[1] - elapsed
            if elapsed > self.timeout_ns:
                self._pending.append({"kind": "watchdog_gap", "gap_ms": elapsed / 1e6})
            if abs(clock_delta) > 100_000_000:
                self._pending.append({"kind": "wall_clock_jump", "delta_ms": clock_delta / 1e6})
        for _ in range(32):
            try:
                self._pending.append(self._incidents.popleft())
            except IndexError:
                break

        streams, active = self._streams.copy(), self._active.copy()
        stream_details = {
            name: {
                "count": s.count, "age_ms": (now - s.mono_ns) / 1e6,
                "stamp_ns": s.stamp_ns, "source_age_ms": s.source_age_ms,
                "dds_to_callback_ms": s.dds_to_callback_ms,
                "source_to_dds_ms": s.source_to_dds_ms,
            } for name, s in streams.items()
        }
        active_details = {
            name: {"stage": a[0], "age_ms": (now - a[1]) / 1e6,
                   "thread_id": a[2], "native_tid": a[4], "stamp_ns": a[3]}
            for name, a in active.items()
        }
        # A stream arms on its first sample. Command publishing arms the control
        # lane, so ordinary startup/service waits do not look like walking stalls.
        stalled = [name for name, s in streams.items() if now - s.mono_ns > self.timeout_ns]
        blocked = [name for name, a in active.items()
                   if now - a[1] > self.timeout_ns and
                   (name != "control" or "command_tx" in streams)]
        self._history.append({
            "mono_ns": now,
            "counts": {name: s.count for name, s in streams.items()},
            "control": active_details.get("control"),
        })
        faulty = bool(stalled or blocked)
        if not faulty and not self._pending and not self._last_fault:
            return
        if now - self._last_report_ns < 1_000_000_000 and self._last_report_ns:
            return
        report = {
            "event": "stall" if faulty else ("recovered" if self._last_fault else "incident"),
            "mono_ns": now, "wall_ns": wall,
            "stalled_streams": stalled, "blocked_lanes": blocked,
            "streams": stream_details, "active": active_details,
            "incidents": list(self._pending), "recent": list(self._history),
        }
        if faulty and not self._last_fault:
            frames = sys._current_frames()
            stacks = {}
            for lane, a in active.items():
                frame, stack = frames.get(a[2]), []
                for _ in range(8):
                    if frame is None:
                        break
                    stack.append(f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}")
                    frame = frame.f_back
                stacks[lane] = stack
                try:
                    active_details[lane]["cpu_affinity"] = sorted(os.sched_getaffinity(a[4]))
                    active_details[lane]["scheduling_policy"] = os.sched_getscheduler(a[4])
                    active_details[lane]["priority"] = os.sched_getparam(a[4]).sched_priority
                    with open(f"/proc/self/task/{a[4]}/schedstat", encoding="ascii") as f:
                        active_details[lane]["schedstat_runtime_wait_timeslices"] = f.read().strip()
                except (OSError, ProcessLookupError):
                    pass
            report["stacks"] = stacks
            del frames
        self._emit("warning", report)
        self._pending.clear()
        self._last_report_ns, self._last_fault = now, faulty

    def _emit(self, severity, report):
        if self._log is None:
            return
        if self._context is not None:
            try:
                report["ros"] = self._context()
            except Exception as exc:
                report["ros"] = {"query_error": str(exc)}
        self._log(severity, "communication_trace " + json.dumps(
            {**self._identity, **report}, ensure_ascii=False, separators=(",", ":")
        ))


class CommunicationExecutor(MultiThreadedExecutor):
    """Retain take_message metadata that the Humble executor otherwise drops.

    Only registered subscriptions use the metadata bridge; all worker callbacks
    use the optional communication affinity. Group admission and business
    callback signatures remain those of Humble rclpy. The two executor hooks
    and constructor pool wiring must be revalidated when upgrading ROS versions.
    """

    def __init__(self, trace, clock, *, cpu_affinity=None, **kwargs):
        super().__init__(**kwargs)
        if cpu_affinity is not None:
            # Humble creates an empty pool in its constructor. Replace it before
            # any work is submitted, retaining the requested worker count.
            workers = self._executor._max_workers
            self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="bxi-ros",
                initializer=configure_current_thread, initargs=(frozenset(cpu_affinity),),
            )
        self._trace, self._trace_clock = trace, clock
        self._traced_subscriptions = {}

    def trace_subscription(self, subscription, stream):
        self._traced_subscriptions[subscription] = stream

    def _take_subscription(self, sub):
        if sub not in self._traced_subscriptions:
            return super()._take_subscription(sub)
        with sub.handle:
            return sub.handle.take_message(sub.msg_type, sub.raw)

    async def _execute_subscription(self, sub, taken):
        stream = self._traced_subscriptions.get(sub)
        if stream is None:
            await super()._execute_subscription(sub, taken)
            return
        if taken is None:
            return
        msg, info = taken
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        stamp_ns = 0 if stamp is None else stamp.sec * 1_000_000_000 + stamp.nanosec
        self._trace.enter("callback", lane=stream, stamp_ns=stamp_ns)
        try:
            self._trace.record(stream, stamp_ns=stamp_ns,
                               ros_now_ns=self._trace_clock().nanoseconds, info=info)
            await super()._execute_subscription(sub, msg)
        finally:
            self._trace.leave(stream)
