import csv
import json
import os
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String


FIELDS = (
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
)


def _values(message: Imu) -> dict[str, float]:
    return {
        "orientation_x": message.orientation.x,
        "orientation_y": message.orientation.y,
        "orientation_z": message.orientation.z,
        "orientation_w": message.orientation.w,
        "angular_velocity_x": message.angular_velocity.x,
        "angular_velocity_y": message.angular_velocity.y,
        "angular_velocity_z": message.angular_velocity.z,
        "linear_acceleration_x": message.linear_acceleration.x,
        "linear_acceleration_y": message.linear_acceleration.y,
        "linear_acceleration_z": message.linear_acceleration.z,
    }


def _stamp(message: Imu) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class ImuCompareRecorder(Node):
    def __init__(self) -> None:
        super().__init__("imu_compare_recorder")
        hipnuc_topic = self.declare_parameter(
            "hipnuc_topic", "/hardware/imu_data"
        ).value
        hardware_topic = self.declare_parameter(
            "hardware_topic", "/hardware/imu_data_hardware"
        ).value
        robot_state_topic = self.declare_parameter(
            "robot_state_topic", "/hardware/state_machine_info"
        ).value
        output_dir = os.path.expanduser(
            self.declare_parameter("output_dir", "/tmp/bxi").value
        )
        self.max_pair_dt = float(
            self.declare_parameter("max_pair_dt_sec", 0.02).value
        )
        self._hipnuc_queue = deque(maxlen=500)
        self._hardware_queue = deque(maxlen=500)
        self._robot_state = ""
        self._robot_state_id = ""
        self._robot_state_mode = ""
        self._rows_since_flush = 0

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = self._unique_output_path(output_dir, timestamp)
        self._file = open(output_csv, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._file.flush()

        self.create_subscription(
            Imu,
            hipnuc_topic,
            lambda message: self._on_imu("hipnuc", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            hardware_topic,
            lambda message: self._on_imu("hardware", message),
            qos_profile_sensor_data,
        )
        self.create_subscription(String, robot_state_topic, self._on_robot_state, 10)
        self.get_logger().info(
            f"recording {hipnuc_topic} and {hardware_topic} to {output_csv}"
        )

    @staticmethod
    def _unique_output_path(output_dir: str, timestamp: str) -> str:
        base = os.path.join(output_dir, f"imu_compare_{timestamp}")
        candidate = f"{base}.csv"
        index = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{index}.csv"
            index += 1
        return candidate

    @staticmethod
    def _fieldnames() -> list[str]:
        fields = [
            "record_time_sec",
            "robot_state",
            "robot_state_id",
            "robot_state_mode",
            "hipnuc_stamp_sec",
            "hardware_stamp_sec",
            "receive_dt_sec",
        ]
        for source in ("hipnuc", "hardware"):
            fields.extend(f"{source}_{field}" for field in FIELDS)
        fields.extend(f"delta_{field}" for field in FIELDS)
        return fields

    def _on_robot_state(self, message: String) -> None:
        try:
            snapshot = json.loads(message.data)
            current = snapshot.get("current", {})
            self._robot_state = str(current.get("name", ""))
            self._robot_state_id = str(current.get("id", ""))
            self._robot_state_mode = str(snapshot.get("mode", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("received invalid robot state message")

    def _on_imu(self, source: str, message: Imu) -> None:
        receive_time = self.get_clock().now().nanoseconds * 1e-9
        own_queue = self._hipnuc_queue if source == "hipnuc" else self._hardware_queue
        other_queue = self._hardware_queue if source == "hipnuc" else self._hipnuc_queue
        own_queue.append((receive_time, message))

        if not other_queue:
            return
        other_index, (other_receive, other_message) = min(
            enumerate(other_queue),
            key=lambda item: abs(item[1][0] - receive_time),
        )
        if abs(other_receive - receive_time) > self.max_pair_dt:
            return
        del other_queue[other_index]
        own_receive, own_message = own_queue.pop()
        if source == "hipnuc":
            self._write_pair(own_receive, own_message, other_receive, other_message)
        else:
            self._write_pair(other_receive, other_message, own_receive, own_message)

    def _write_pair(
        self,
        hipnuc_receive: float,
        hipnuc_message: Imu,
        hardware_receive: float,
        hardware_message: Imu,
    ) -> None:
        hipnuc = _values(hipnuc_message)
        hardware = _values(hardware_message)
        row = {
            "record_time_sec": f"{max(hipnuc_receive, hardware_receive):.9f}",
            "robot_state": self._robot_state,
            "robot_state_id": self._robot_state_id,
            "robot_state_mode": self._robot_state_mode,
            "hipnuc_stamp_sec": f"{_stamp(hipnuc_message):.9f}",
            "hardware_stamp_sec": f"{_stamp(hardware_message):.9f}",
            "receive_dt_sec": f"{hipnuc_receive - hardware_receive:.9f}",
        }
        row.update({f"hipnuc_{key}": f"{value:.9f}" for key, value in hipnuc.items()})
        row.update({f"hardware_{key}": f"{value:.9f}" for key, value in hardware.items()})
        row.update(
            {
                f"delta_{key}": f"{hipnuc[key] - hardware[key]:.9f}"
                for key in FIELDS
            }
        )
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 250:
            self._file.flush()
            self._rows_since_flush = 0

    def destroy_node(self) -> bool:
        self._file.flush()
        self._file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuCompareRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
