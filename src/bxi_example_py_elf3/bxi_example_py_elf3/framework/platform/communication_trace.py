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

    def __init__(self, timeout_sec=0.2, summary_interval_sec=5.0):
        self.timeout_ns = int(timeout_sec * 1e9)
        if self.timeout_ns <= 0:
            raise ValueError("communication trace timeout must be positive")
        self.summary_interval_ns = int(summary_interval_sec * 1e9)
        if self.summary_interval_ns <= 0:
            raise ValueError("communication summary interval must be positive")
        self._streams = {}
        self._active = {}
        self._incidents = deque(maxlen=32)
        self._history = deque(maxlen=10)
        self._pending = deque(maxlen=32)
        self._last_fault = False
        self._last_summary_ns = 0
        self._last_conditions = frozenset()
        self._reported_conditions = set()
        self._reported_incidents = set()
        self._last_incident_ns = 0
        self._diagnostic_id = 0
        self._last_poll = None
        self._stop = Event()
        self._thread = None
        self._log = None
        self._context = None
        self._summary_context = None
        self._last_ros_context = {}
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

    def start(self, log, context=None, *, cpu_affinity=None, summary_context=None):
        if self._thread is not None:
            return
        self._log, self._context = log, context
        self._summary_context = summary_context
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
                self._last_incident_ns = now
            if abs(clock_delta) > 100_000_000:
                self._pending.append({"kind": "wall_clock_jump", "delta_ms": clock_delta / 1e6})
                self._last_incident_ns = now
        for _ in range(32):
            try:
                self._pending.append(self._incidents.popleft())
                self._last_incident_ns = now
            except IndexError:
                break
        # Rearm transient-incident detail after a quiet interval. A continuous
        # stream of stale stamps must not cause a full JSON dump every frame.
        if not self._pending and now - self._last_incident_ns >= self.summary_interval_ns:
            self._reported_incidents.clear()

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
        conditions = frozenset(
            [("stream", name) for name in stalled]
            + [("lane", name, active[name][0]) for name in blocked]
        )
        incident_types = {(i["kind"], i.get("stream")) for i in self._pending}
        recovering = self._last_fault and not faulty
        # Recovery gaps are summarized with old/new stamps instead of dumping
        # the entire graph again. New clock/DDS anomalies still retain detail.
        new_incidents = incident_types - self._reported_incidents
        new_incidents = {
            key for key in new_incidents
            if not (key[0] == "stream_gap" and
                    (recovering or ("stream", key[1]) in self._reported_conditions))
        }
        detail = bool(conditions - self._reported_conditions or new_incidents)
        changed = conditions != self._last_conditions
        if (not detail and not changed and
                now - self._last_summary_ns < self.summary_interval_ns):
            return
        event = "stall" if faulty else ("recovered" if recovering else "incident")
        if detail:
            self._diagnostic_id += 1
        self._write_summary(event, now, streams, active, stalled, blocked, detail)
        if detail:
            report = {
                "event": event, "diagnostic_id": self._diagnostic_id,
                "mono_ns": now, "wall_ns": wall,
                "stalled_streams": stalled, "blocked_lanes": blocked,
                "streams": stream_details, "active": active_details,
                "incidents": list(self._pending), "recent": list(self._history),
            }
        if detail and faulty:
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
        if detail:
            self._emit("warning", report)
            self._reported_conditions.update(conditions)
            self._reported_incidents.update(incident_types)
        self._pending.clear()
        self._last_summary_ns, self._last_fault = now, faulty
        self._last_conditions = conditions
        if recovering:
            self._reported_conditions.clear()

    def _write_summary(self, event, now, streams, active, stalled, blocked, detail):
        if self._log is None:
            return
        labels = {"command_tx": "电机命令", "joint_rx": "关节反馈",
                  "imu_rx": "IMU", "remote_rx": "遥控输入", "control": "控制线程"}
        stages = {"waiting_next_cycle": "等待下周期", "input_snapshot": "输入快照",
                  "framework_lock": "等待框架锁", "state_update": "状态/推理",
                  "command_publish": "命令发布", "callback": "回调执行"}
        title = {"stall": "通信告警", "recovered": "通信接收/发布恢复",
                 "incident": "通信瞬时异常"}[event]
        problems = [f"{labels.get(name, name)}停更{(now - streams[name].mono_ns) / 1e9:.2f}s"
                    for name in stalled]
        problems += [f"{labels.get(name, name)}卡在{stages.get(active[name][0], active[name][0])}"
                     f"{(now - active[name][1]) / 1e6:.1f}ms" for name in blocked]
        rates = []
        for name in ("command_tx", "joint_rx", "imu_rx", "remote_rx"):
            sample = streams.get(name)
            if sample is None:
                rates.append(f"{labels[name]}=--")
                continue
            earliest = next((h for h in self._history if name in h["counts"]), None)
            elapsed = 0 if earliest is None else now - earliest["mono_ns"]
            rate = (f"{max(0, sample.count - earliest['counts'][name]) * 1e9 / elapsed:.1f}Hz"
                    if elapsed > 0 else "频率待统计")
            rates.append(f"{labels[name]}={rate}")
        context = self._last_ros_context
        if self._summary_context is not None:
            try:
                context = self._summary_context()
            except Exception:
                context = {}
        control = active.get("control")
        control_stage = "未观测" if control is None else stages.get(control[0], control[0])
        incident_notes = []
        for kind, stream in sorted({(i["kind"], i.get("stream", "")) for i in self._pending}):
            items = [i for i in self._pending if i["kind"] == kind and i.get("stream", "") == stream]
            value = max((i.get("gap_ms", i.get("delta_ms", i.get("value", 0)))
                         for i in items), key=abs)
            incident_notes.append(f"{labels.get(stream, stream)}:{kind}={value:.1f}ms")
            if kind == "stream_gap":
                worst = max(items, key=lambda i: i["gap_ms"])
                incident_notes.append(f"stamp_ns={worst['previous_stamp_ns']}→{worst['stamp_ns']}")
        fields = [f"{title}[诊断#{self._diagnostic_id}]：" + ("，".join(problems) or "当前无停更/阻塞"),
                  "近期更新频率：" + "/".join(rates), f"控制={control_stage}",
                  f"状态={str(context.get('state', '未知')).rsplit('/', 1)[-1]}"]
        if "remote_input_fresh" in context:
            fields.append("遥控有效=" + ("是" if context["remote_input_fresh"] else "否"))
        if incident_notes:
            fields.append("事件=" + ",".join(incident_notes))
        if detail:
            fields.append(f"完整诊断见 communication_trace 的 diagnostic_id={self._diagnostic_id}")
        self._log("info" if event == "recovered" else "warning", "；".join(fields))

    def _emit(self, severity, report):
        if self._log is None:
            return
        if self._context is not None:
            try:
                report["ros"] = self._context()
                self._last_ros_context = report["ros"]
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
