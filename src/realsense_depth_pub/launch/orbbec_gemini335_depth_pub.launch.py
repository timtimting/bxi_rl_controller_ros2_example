from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument('orbbec_camera_name', default_value='orbbec'),
        DeclareLaunchArgument('serial_number', default_value=''),
        DeclareLaunchArgument('usb_port', default_value=''),
        DeclareLaunchArgument('depth_width', default_value='0'),
        DeclareLaunchArgument('depth_height', default_value='0'),
        DeclareLaunchArgument('depth_fps', default_value='60'),
        DeclareLaunchArgument('show_fps_enable', default_value='true'),
        DeclareLaunchArgument('publish_full', default_value='true'),
        DeclareLaunchArgument('out_w', default_value='64'),
        DeclareLaunchArgument('out_h', default_value='36'),
        DeclareLaunchArgument('hfov', default_value='89.24'),
        DeclareLaunchArgument('vfov', default_value='58.06'),
        DeclareLaunchArgument('min_dist', default_value='0.2'),
        DeclareLaunchArgument('max_dist', default_value='2.5'),
        DeclareLaunchArgument('publish_rate_hz', default_value='60.0'),
        DeclareLaunchArgument('enable_spatial_filter', default_value='true'),
        DeclareLaunchArgument('spat_alpha', default_value='0.45'),
        DeclareLaunchArgument('spat_delta', default_value='20.0'),
        DeclareLaunchArgument('spat_holes', default_value='2'),
        DeclareLaunchArgument('enable_temporal_filter', default_value='true'),
        DeclareLaunchArgument('temp_alpha', default_value='0.45'),
        DeclareLaunchArgument('temp_delta', default_value='20.0'),
        DeclareLaunchArgument('enable_hole_filling_filter', default_value='true'),
        DeclareLaunchArgument('hole1', default_value='1'),
        DeclareLaunchArgument('hole2', default_value='2'),
        DeclareLaunchArgument('input_depth_topic', default_value='/orbbec/depth/image_raw'),
        DeclareLaunchArgument('input_depth_info_topic', default_value='/orbbec/depth/camera_info'),
        DeclareLaunchArgument('frame_depth', default_value='camera_depth_optical_frame'),
        DeclareLaunchArgument('topic_depth', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('topic_depth_info', default_value='/camera/depth/camera_info'),
        DeclareLaunchArgument('topic_small', default_value='/camera/depth/image_64x36'),
        DeclareLaunchArgument('topic_small_info', default_value='/camera/depth/camera_info_64x36'),
    ]

    orbbec_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('orbbec_camera'),
                'launch',
                'gemini_330_series.launch.py',
            ])
        ]),
        launch_arguments={
            'camera_name': LaunchConfiguration('orbbec_camera_name'),
            'serial_number': LaunchConfiguration('serial_number'),
            'usb_port': LaunchConfiguration('usb_port'),
            'depth_width': LaunchConfiguration('depth_width'),
            'depth_height': LaunchConfiguration('depth_height'),
            'depth_fps': LaunchConfiguration('depth_fps'),
            'enable_depth': 'true',
            'enable_color': 'false',
            'enable_left_ir': 'false',
            'enable_right_ir': 'false',
            'enable_accel': 'false',
            'enable_gyro': 'false',
            'enable_point_cloud': 'false',
            'show_fps_enable': LaunchConfiguration('show_fps_enable'),
        }.items(),
    )

    bridge = Node(
        package='realsense_depth_pub',
        executable='orbbec_depth_bridge_node',
        name='orbbec_depth_bridge',
        output='screen',
        parameters=[{
            'input_depth_topic': LaunchConfiguration('input_depth_topic'),
            'input_depth_info_topic': LaunchConfiguration('input_depth_info_topic'),
            'topic_depth': LaunchConfiguration('topic_depth'),
            'topic_depth_info': LaunchConfiguration('topic_depth_info'),
            'topic_small': LaunchConfiguration('topic_small'),
            'topic_small_info': LaunchConfiguration('topic_small_info'),
            'frame_depth': LaunchConfiguration('frame_depth'),
            'out_w': LaunchConfiguration('out_w'),
            'out_h': LaunchConfiguration('out_h'),
            'hfov': LaunchConfiguration('hfov'),
            'vfov': LaunchConfiguration('vfov'),
            'min_dist': LaunchConfiguration('min_dist'),
            'max_dist': LaunchConfiguration('max_dist'),
            'publish_full': LaunchConfiguration('publish_full'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'enable_spatial_filter': LaunchConfiguration('enable_spatial_filter'),
            'spat_alpha': LaunchConfiguration('spat_alpha'),
            'spat_delta': LaunchConfiguration('spat_delta'),
            'spat_holes': LaunchConfiguration('spat_holes'),
            'enable_temporal_filter': LaunchConfiguration('enable_temporal_filter'),
            'temp_alpha': LaunchConfiguration('temp_alpha'),
            'temp_delta': LaunchConfiguration('temp_delta'),
            'enable_hole_filling_filter': LaunchConfiguration('enable_hole_filling_filter'),
            'hole1': LaunchConfiguration('hole1'),
            'hole2': LaunchConfiguration('hole2'),
            'use_input_frame_id': False,
        }],
    )

    return LaunchDescription(declared_arguments + [orbbec_driver, bridge])
