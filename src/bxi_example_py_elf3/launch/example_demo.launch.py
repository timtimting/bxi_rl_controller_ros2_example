import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    xml_file_name = "data/mujoco_simulation/elf3.xml"
    xml_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), xml_file_name)
    state_machine_config = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "config/elf3_state_machine.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_depth_camera",
                default_value="false",
                description="Start the optional RealSense depth camera publisher.",
            ),
            Node(
                package="mujoco",
                executable="simulation",
                name="simulation_mujoco",
                output="screen",
                parameters=[
                    {"simulation/model_file": xml_file},
                ],
                emulate_tty=True,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('realsense_depth_pub'),
                        'launch',
                        'realsense_depth_pub_zlab_origin.launch.py'
                    ])
                ]),
                condition=IfCondition(LaunchConfiguration("enable_depth_camera")),
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_demo",
                name="bxi_example_py_elf3_demo",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/state_machine_config": state_machine_config},
                    {"/hot_reload": True},
                ],
                emulate_tty=True,
            ),
        ]
    )
