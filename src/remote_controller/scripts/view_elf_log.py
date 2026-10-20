#!/usr/bin/env python3
"""Follow the main ELF log with terminal-only highlighting."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


RESET = "\033[0m"
COLORS = {
    "process": "\033[2m",
    "time": "\033[2m",
    "debug": "\033[90m",
    "info": "\033[32m",
    "warn": "\033[1;33m",
    "error": "\033[1;31m",
    "fatal": "\033[1;37;41m",
    "state": "\033[1;36m",
    "remote": "\033[1;35m",
    "imu": "\033[1;34m",
    "hardware": "\033[1;33m",
    "traceback": "\033[1;31m",
}
TOKENS = re.compile(
    r"(?P<process>^\[[^\]\n]+-\d+\])"
    r"|(?P<time>\[\d{9,}\.\d+\])"
    r"|\[(?P<severity>DEBUG|INFO|WARN|ERROR|FATAL)\]"
    r"|(?P<state>state transition(?: queued for resource preparation)?)"
    r"|(?P<remote>remote_path|remote input watchdog)"
    r"|(?P<imu>\bIMU\b)"
    r"|(?P<hardware>motor_loss:|hardware_trace|can\[\d+\] rx error cnt)"
    r"|(?P<traceback>Traceback \(most recent call last\):|RCLError)"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def colorize(line: str, enabled: bool) -> str:
    line = ANSI_ESCAPE.sub("", line)
    if not enabled:
        return line

    def replace(match: re.Match[str]) -> str:
        group = match.lastgroup
        if group == "severity":
            token = match.group(group)
            color = COLORS[token.lower()]
            return f"{color}[{token}]{RESET}"
        return f"{COLORS[group]}{match.group(0)}{RESET}"

    return TOKENS.sub(replace, line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, help="log file (default: latest *_elf.log)"
    )
    parser.add_argument("--lines", type=int, default=80, help="initial lines to show")
    parser.add_argument("--once", action="store_true", help="print and exit instead of following")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()
    if args.lines < 0:
        parser.error("--lines must be nonnegative")

    path = args.path
    if path is None:
        root = Path(os.environ.get("BXI_LOG_ROOT", "/var/log/bxi_log"))
        logs = list(root.glob("*_elf.log"))
        if not logs:
            parser.error(f"no *_elf.log found in {root}")
        path = max(logs, key=lambda item: item.name)

    use_color = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    command = ["tail", "-n", str(args.lines)]
    if not args.once:
        command.append("-F")
        print(f"Following {path} (Ctrl+C to stop)", file=sys.stderr)
    command.extend(("--", str(path)))

    with subprocess.Popen(
        command, stdout=subprocess.PIPE, text=True, errors="replace"
    ) as process:
        try:
            for line in process.stdout:
                sys.stdout.write(colorize(line, use_color))
                sys.stdout.flush()
        except (KeyboardInterrupt, BrokenPipeError):
            if process.poll() is None:
                process.terminate()
            return 0
        return process.wait()


if __name__ == "__main__":
    sys.exit(main())
