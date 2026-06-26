import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    xml_file_name = "data/mujoco_simulation/elf3.xml"
    xml_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), xml_file_name)
    
    onnx_file_name = "data/mjlab_model/model_normal.onnx"
    onnx_file = os.path.join(get_package_share_path("bxi_example_py_elf3"), onnx_file_name)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dof_num",
                default_value="31",
                description="控制节点最终发布/接收的关节数量",
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
                arguments=[("__log_level:=debug")],
            ),

            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_mjlab",
                name="bxi_example_py_elf3_mjlab",
                output="screen",
                parameters=[
                    {"/topic_prefix": "simulation/"},
                    {"/dof_num": LaunchConfiguration("dof_num")},
                    {"/onnx_file": onnx_file},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
