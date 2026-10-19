import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "elf_log_split.py"
STARTER = SCRIPT.with_name("start_elf_logged.sh")


def test_full_trace_is_kept_out_of_main_log(tmp_path):
    main = tmp_path / "run_elf.log"
    trace = tmp_path / "trace" / "run_elf.trace.log"
    hardware = {"side": "hardware", "event": "stall", "command_rx": {"age_ms": 250},
                "joint_tx": {"age_ms": 2}, "motor_tx": {"age_ms": 400},
                "joint_lane": {"stage": "motor_feedback_wait"}, "motor_timeouts": 3}
    controller = {"event": "stall", "diagnostic_id": 7, "stalled_streams": ["joint_rx"]}
    lines = [
        "[INFO] booted\n",
        "[hardware] communication_trace " + json.dumps(hardware) + "\n",
        "[hardware] communication_trace " + json.dumps(hardware) + "\n",
        "[demo] communication_trace " + json.dumps(controller) + "\n",
        "[WARN] communication summary #7\n",
    ]
    subprocess.run([sys.executable, str(SCRIPT), str(main), str(trace)],
                   input="".join(lines), text=True, check=True)
    main_text = main.read_text()
    trace_text = trace.read_text()
    assert "[INFO] booted" in main_text
    assert "[WARN] communication summary #7" in main_text
    assert "hardware_trace event=stall" in main_text
    assert main_text.count("hardware_trace event=stall") == 1
    assert '"stalled_streams"' not in main_text
    assert trace_text.count("communication_trace {") == 3


def test_log_storage_failure_preserves_output_in_journal(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "/proc/elf_main.log",
         str(tmp_path / "trace.log")],
        input="[WARN] fault\n", text=True, capture_output=True, check=True,
    )
    assert result.stdout == "[WARN] fault\n"
    assert "ELF log unavailable" in result.stderr


def test_launcher_creates_matching_main_and_trace_files(tmp_path):
    fake_ros2 = tmp_path / "ros2"
    fake_ros2.write_text(
        "#!/bin/sh\n"
        "printf '[INFO] robot started\\n'\n"
        "printf '[demo] communication_trace {\"event\":\"started\"}\\n'\n"
    )
    fake_ros2.chmod(0o755)
    log_root = tmp_path / "logs"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "BXI_LOG_ROOT": str(log_root)}
    subprocess.run([str(STARTER)], env=env, check=True)
    main_logs = list(log_root.glob("*_elf.log"))
    trace_logs = list((log_root / "trace").glob("*_elf.trace.log"))
    assert len(main_logs) == len(trace_logs) == 1
    assert "robot started" in main_logs[0].read_text()
    assert "communication_trace" not in main_logs[0].read_text().splitlines()[-1]
    assert '"event":"started"' in trace_logs[0].read_text()
