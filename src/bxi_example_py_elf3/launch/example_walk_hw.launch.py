import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    declare_hardware_launch_arguments,
    hardware_node_from_context,
)

def generate_launch_description():

    onnx_file_name = "mods/com.bxi.basic_actions/assets/model_normal.onnx"
    onnx_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), onnx_file_name)

    return LaunchDescription(
        declare_hardware_launch_arguments()
        + [
            DeclareLaunchArgument("imu_serial_port", default_value="/dev/ttyIMU_2"),
            DeclareLaunchArgument("imu_baud_rate", default_value="921600"),
            DeclareLaunchArgument(
                "imu_compare_csv",
                default_value="/tmp/bxi/bxi_imu_compare.csv",
            ),
            OpaqueFunction(
                function=lambda context: [
                    hardware_node_from_context(
                        context, imu_topic="/hardware/imu_data_hardware"
                    )
                ]
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="imu_compare_recorder",
                name="imu_compare_recorder",
                output="screen",
                parameters=[
                    {"output_csv": LaunchConfiguration("imu_compare_csv")}
                ],
                emulate_tty=True,
            ),
            Node(
                # HiPNUC serial driver built inside this example workspace.
                package="hipnuc_imu",
                executable="talker",
                name="IMU_publisher",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "serial_port": LaunchConfiguration("imu_serial_port"),
                        "baud_rate": LaunchConfiguration("imu_baud_rate"),
                        "frame_id": "imu_link",
                        "imu_switch": True,
                        "imu_topic": "/hardware/imu_data",
                        "euler_switch": False,
                        "magnetic_switch": False,
                        "temperature_switch": False,
                        "pressure_switch": False,
                    }
                ],
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
