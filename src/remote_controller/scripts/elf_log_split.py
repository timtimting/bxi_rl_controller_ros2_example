#!/usr/bin/env python3
"""Keep the ELF launch log readable while retaining full trace evidence."""

import json
import re
import sys
import time
from pathlib import Path


MARKER = "communication_trace {"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def route_line(line, main_log, trace_log, trace_name, hardware_summary):
    line = ANSI_ESCAPE.sub("", line)
    prefix, marker, payload = line.partition(MARKER)
    if not marker:
        main_log.write(line)
        return
    trace_log.write(line)
    try:
        detail = json.loads("{" + payload)
        if not isinstance(detail, dict):
            raise ValueError("trace must be a JSON object")
    except (json.JSONDecodeError, ValueError):
        main_log.write(line)
        return
    if detail.get("side") != "hardware" or detail.get("event") == "started":
        return

    event = detail.get("event", "unknown")
    command = detail.get("command_rx") or {}
    joint = detail.get("joint_tx") or {}
    motor = detail.get("motor_tx") or {}
    lane = detail.get("joint_lane") or {}
    if not all(isinstance(item, dict) for item in (command, joint, motor, lane)):
        main_log.write(line)
        return

    def stalled(stream):
        age = stream.get("age_ms")
        return isinstance(age, (int, float)) and age > 200

    key = (event, lane.get("stage"), stalled(command), stalled(joint), stalled(motor))
    now = time.monotonic()
    if key == hardware_summary[0] and now - hardware_summary[1] < 5:
        return
    hardware_summary[:] = [key, now]
    main_log.write(
        f"{prefix}hardware_trace event={event} "
        f"command_age_ms={command.get('age_ms', '--')} "
        f"joint_age_ms={joint.get('age_ms', '--')} "
        f"motor_age_ms={motor.get('age_ms', '--')} "
        f"lane={lane.get('stage', '--')} "
        f"motor_timeouts={detail.get('motor_timeouts', '--')} "
        f"watchdog_drops={detail.get('watchdog_drops', '--')} "
        f"detail={trace_name}\n"
    )


def main():
    main_path, trace_path = (Path(path) for path in sys.argv[1:3])
    sys.stdin.reconfigure(errors="replace")
    try:
        main_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with main_path.open("a", buffering=1) as main_log, trace_path.open("a", buffering=1) as trace_log:
            main_log.write(f"ELF log started; full communication trace: {trace_path}\n")
            hardware_summary = [None, 0.0]
            for line in sys.stdin:
                try:
                    route_line(line, main_log, trace_log, str(trace_path), hardware_summary)
                except OSError as error:
                    print(f"ELF log write failed ({error}); forwarding remaining output to journal", file=sys.stderr)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    break
            else:
                return
    except OSError as error:
        print(f"ELF log unavailable ({error}); forwarding output to journal", file=sys.stderr)
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
