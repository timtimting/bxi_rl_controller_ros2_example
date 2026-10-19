#!/usr/bin/env bash
set -uo pipefail

log_root="${BXI_LOG_ROOT:-/var/log/bxi_log}"
export BXI_LOG_ROOT="$log_root"
stamp="$(date +%Y-%m-%d_%H-%M-%S)"
main_log="$log_root/${stamp}_elf.log"
trace_log="$log_root/trace/${stamp}_elf.trace.log"

ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py enable_imu:=false 2>&1 |
  python3 -u "$(dirname "$0")/elf_log_split.py" "$main_log" "$trace_log"
