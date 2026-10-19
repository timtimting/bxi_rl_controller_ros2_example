import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view_elf_log.py"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class ViewElfLogTest(unittest.TestCase):
    def test_color_only_changes_terminal_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run_elf.log"
            raw = (
                "[demo-3] [INFO] [1792422385.1] state transition: to=normal\n"
                "[demo-3] [WARN] remote_path status=usb_stale IMU\n"
                "[hardware-2] motor_loss: motor[28]:1371.92 %%%\n"
            )
            log.write_text(raw)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--once", "--color", "always", str(log)],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("\x1b[32m[INFO]\x1b[0m", result.stdout)
            self.assertIn("\x1b[1;33m[WARN]\x1b[0m", result.stdout)
            self.assertIn("\x1b[1;36mstate transition\x1b[0m", result.stdout)
            self.assertIn("\x1b[1;35mremote_path\x1b[0m", result.stdout)
            self.assertIn("\x1b[1;34mIMU\x1b[0m", result.stdout)
            self.assertIn("\x1b[1;33mmotor_loss:\x1b[0m", result.stdout)
            self.assertEqual(ANSI.sub("", result.stdout), raw)
            self.assertEqual(log.read_text(), raw)

    def test_piped_output_is_plain_and_uses_latest_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2026-10-18_elf.log").write_text("older\n")
            (root / "2026-10-19_elf.log").write_text("[INFO] newer\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--once"],
                env={**os.environ, "BXI_LOG_ROOT": str(root)},
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout, "[INFO] newer\n")


if __name__ == "__main__":
    unittest.main()
