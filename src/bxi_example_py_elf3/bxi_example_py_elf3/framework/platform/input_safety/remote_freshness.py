"""Observe USB/protocol/publish/receive independently."""

import json
import math


SOURCE_STATUSES = frozenset({
    "no_source", "device_closed", "reader_stalled", "event_source_unverified",
    "waiting_valid_frame", "waiting_usb", "fresh", "protocol_stale", "usb_stale",
})


class RemotePathMonitor:
    def __init__(self, started_at, timeout_sec=0.5, summary_sec=5.0):
        self.started_at = started_at
        self.timeout = timeout_sec
        self.summary_sec = summary_sec
        self.received_at = None
        self.sample = None
        self.identity = None
        self.last_sequence = None
        self.sequence_advanced_at = None
        self.last_stamp = None
        self.invalid = None
        self.last_reported = None
        self.next_report = 0.0

    def observe(self, text, now, ros_now_ns):
        try:
            sample = json.loads(text)
            if sample.get("schema") != 1:
                raise ValueError("schema")
            for name in ("stamp_ns", "instance", "sample_seq", "published", "timeout_ms"):
                if type(sample[name]) is not int or sample[name] < 0:
                    raise ValueError(name)
            if sample["timeout_ms"] == 0:
                raise ValueError("timeout_ms")
            for name in ("usb_read_age_ms", "valid_frame_age_ms", "reader_age_ms", "publish_age_ms"):
                if not isinstance(sample[name], (float, int)) or not math.isfinite(sample[name]) or sample[name] < -1:
                    raise ValueError(name)
            for name in ("source", "mode", "status", "device"):
                if not isinstance(sample[name], str):
                    raise ValueError(name)
            if sample["status"] not in SOURCE_STATUSES:
                raise ValueError("unknown_status")
            if type(sample["publish_on_change"]) is not bool:
                raise ValueError("publish_on_change")
            stamp = sample["stamp_ns"]
            if stamp and abs(ros_now_ns - stamp) > self.timeout * 1e9:
                raise ValueError("stale_or_future_status_stamp")
            identity = (sample["instance"], sample["source"], sample.get("io_kind"), sample["device"])
            if identity == self.identity and stamp and self.last_stamp and stamp <= self.last_stamp:
                raise ValueError("nonincreasing_status_stamp")
            sequence = sample["sample_seq"]
            if identity == self.identity and self.last_sequence is not None and sequence < self.last_sequence:
                raise ValueError("sample_sequence_reversed")
            if identity != self.identity or sequence != self.last_sequence:
                self.sequence_advanced_at = now
            self.identity, self.last_sequence, self.last_stamp = identity, sequence, stamp
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            self.invalid = str(error)
            return False
        self.sample, self.received_at, self.invalid = sample, now, None
        return True

    def status(self, now, command_age_ms):
        if self.invalid:
            return "invalid_status"
        if self.received_at is None:
            return "waiting_status" if now - self.started_at < 3 else "status_unavailable"
        if now - self.received_at > self.timeout:
            return "status_link_stale"
        s = self.sample
        if s["status"] not in ("fresh", "event_source_unverified"):
            return s["status"]
        if not s.get("active", True):
            return "input_not_active"
        if s["mode"] == "periodic" and now - self.sequence_advanced_at > s["timeout_ms"] / 1000:
            return "valid_frames_not_advancing"
        if s["publish_on_change"]:
            return "event_source_unverified" if s["mode"] != "periodic" else "on_change_idle_or_fresh"
        if s["publish_age_ms"] < 0 or s["publish_age_ms"] > self.timeout * 1000:
            return "publisher_stale"
        if command_age_ms is None or command_age_ms > self.timeout * 1000:
            return "demo_receive_stale"
        return "event_source_unverified" if s["mode"] != "periodic" else "fresh"

    def input_unsafe(self, now, command_age_ms):
        # Missing diagnostics from an older publisher remain observational;
        # explicit upstream failures from a live diagnostic stream are gated.
        return self.status(now, command_age_ms) in {
            "invalid_status", "status_link_stale", "device_closed",
            "reader_stalled", "waiting_usb", "waiting_valid_frame",
            "protocol_stale", "usb_stale", "valid_frames_not_advancing",
            "publisher_stale", "input_not_active", "no_source",
        }

    def report(self, now, command_age_ms):
        status = self.status(now, command_age_ms)
        healthy = status in ("fresh", "event_source_unverified", "waiting_status", "on_change_idle_or_fresh")
        key = (status, (self.sample or {}).get("source"), (self.sample or {}).get("io_kind"))
        if key == self.last_reported and (
            healthy or status == "status_unavailable" or now < self.next_report
        ):
            return None
        self.last_reported, self.next_report = key, now + self.summary_sec
        s = self.sample or {}
        severity = "info" if healthy else "warning"
        if not s:
            text = f"remote_path status={status}; no remote_controller/freshness data received"
        else:
            text = (
                f"remote_path status={status} source={s['source']} device={s['device']} "
                f"io={s.get('io_kind', '--')} "
                f"usb_age_ms={s['usb_read_age_ms']:.0f} "
                f"valid_age_ms={s['valid_frame_age_ms']:.0f} "
                f"publish_age_ms={s['publish_age_ms']:.0f} "
                f"demo_age_ms={command_age_ms if command_age_ms is not None else '--'}"
            )
            if s.get("last_stop_status"):
                text += f" last_stop={s['last_stop_status']}"
        if self.invalid:
            text += f" reason={self.invalid}"
        return severity, text
