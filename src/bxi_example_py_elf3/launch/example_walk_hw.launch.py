import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    declare_hardware_launch_arguments,
    hardware_and_imu_nodes_from_context,
)

def generate_launch_description():

    onnx_file_name = "mods/com.bxi.basic_actions/assets/model_normal.onnx"
    onnx_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), onnx_file_name)

    return LaunchDescription(
        declare_hardware_launch_arguments()
        + [
            DeclareLaunchArgument("imu_serial_port", default_value="/dev/ttyIMU_2"),
            DeclareLaunchArgument("imu_baud_rate", default_value="921600"),
            DeclareLaunchArgument("imu_record_dir", default_value="/tmp/bxi"),
            OpaqueFunction(
                function=hardware_and_imu_nodes_from_context,
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_mjlab",
                name="bxi_example_py_elf3_mjlab",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/onnx_file": onnx_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
