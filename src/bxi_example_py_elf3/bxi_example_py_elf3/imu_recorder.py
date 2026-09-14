"""Persist the hardware IMU topic and robot state to one CSV per run."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def _default_output_dir() -> Path:
    """Place recordings beside the workspace, outside volatile /tmp storage."""
    try:
        # <workspace>/install/share/bxi_example_py_elf3 -> <workspace>/csv
        share_path = Path(get_package_share_path("bxi_example_py_elf3"))
        if (
            share_path.parent.name == "share"
            and share_path.parents[1].name == "install"
        ):
            return share_path.parents[2] / "csv"
    except (IndexError, LookupError, OSError):
        pass
    return Path.home() / "bxi_imu_csv"


def _stamp(message: Imu) -> tuple[int, int]:
    return message.header.stamp.sec, message.header.stamp.nanosec


class ImuRecorder(Node):
    def __init__(self) -> None:
        super().__init__("imu_recorder")
        imu_topic = str(
            self.declare_parameter("imu_topic", "/hardware/imu_data").value
        )
        state_topic = str(
            self.declare_parameter(
                "robot_state_topic", "/hardware/state_machine_info"
            ).value
        )
        configured_dir = str(self.declare_parameter("output_dir", "").value).strip()
        output_dir = Path(os.path.expandvars(os.path.expanduser(configured_dir)))
        if not configured_dir:
            output_dir = _default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self._unique_output_path(output_dir, timestamp)
        self._file = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames(),
        )
        self._writer.writeheader()
        self._file.flush()
        self._rows_since_flush = 0
        self._robot_state_name = ""
        self._robot_state_id = ""
        self._robot_state_mode = ""
        self._robot_state_json = ""

        self.create_subscription(
            Imu,
            imu_topic,
            self._on_imu,
            qos_profile_sensor_data,
        )
        self.create_subscription(String, state_topic, self._on_robot_state, 10)
        self.get_logger().info(
            f"recording {imu_topic} and {state_topic} to {output_path}"
        )

    @staticmethod
    def _unique_output_path(output_dir: Path, timestamp: str) -> Path:
        base = output_dir / f"imu_record_{timestamp}"
        candidate = base.with_suffix(".csv")
        index = 1
        while candidate.exists():
            candidate = output_dir / f"imu_record_{timestamp}_{index}.csv"
            index += 1
        return candidate

    @staticmethod
    def _fieldnames() -> list[str]:
        fields = [
            "record_time_sec",
            "imu_stamp_sec",
            "imu_stamp_nanosec",
            "imu_frame_id",
            "robot_state",
            "robot_state_id",
            "robot_state_mode",
            "robot_state_json",
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "linear_acceleration_x",
            "linear_acceleration_y",
            "linear_acceleration_z",
        ]
        fields.extend(f"orientation_covariance_{index}" for index in range(9))
        fields.extend(f"angular_velocity_covariance_{index}" for index in range(9))
        fields.extend(f"linear_acceleration_covariance_{index}" for index in range(9))
        return fields

    def _on_robot_state(self, message: String) -> None:
        try:
            snapshot = json.loads(message.data)
            current = snapshot.get("current", {})
            self._robot_state_name = str(current.get("name", ""))
            self._robot_state_id = str(current.get("id", ""))
            self._robot_state_mode = str(snapshot.get("mode", ""))
            self._robot_state_json = json.dumps(
                snapshot, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("received invalid robot state message")

    def _on_imu(self, message: Imu) -> None:
        stamp_sec, stamp_nanosec = _stamp(message)
        row = {
            "record_time_sec": f"{self.get_clock().now().nanoseconds * 1e-9:.9f}",
            "imu_stamp_sec": stamp_sec,
            "imu_stamp_nanosec": stamp_nanosec,
            "imu_frame_id": message.header.frame_id,
            "robot_state": self._robot_state_name,
            "robot_state_id": self._robot_state_id,
            "robot_state_mode": self._robot_state_mode,
            "robot_state_json": self._robot_state_json,
            "orientation_x": f"{message.orientation.x:.9f}",
            "orientation_y": f"{message.orientation.y:.9f}",
            "orientation_z": f"{message.orientation.z:.9f}",
            "orientation_w": f"{message.orientation.w:.9f}",
            "angular_velocity_x": f"{message.angular_velocity.x:.9f}",
            "angular_velocity_y": f"{message.angular_velocity.y:.9f}",
            "angular_velocity_z": f"{message.angular_velocity.z:.9f}",
            "linear_acceleration_x": f"{message.linear_acceleration.x:.9f}",
            "linear_acceleration_y": f"{message.linear_acceleration.y:.9f}",
            "linear_acceleration_z": f"{message.linear_acceleration.z:.9f}",
        }
        for prefix, values in (
            ("orientation_covariance", message.orientation_covariance),
            ("angular_velocity_covariance", message.angular_velocity_covariance),
            ("linear_acceleration_covariance", message.linear_acceleration_covariance),
        ):
            row.update(
                {
                    f"{prefix}_{index}": f"{value:.9f}"
                    for index, value in enumerate(values)
                }
            )

        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 250:
            self._file.flush()
            self._rows_since_flush = 0

    def destroy_node(self) -> bool:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
