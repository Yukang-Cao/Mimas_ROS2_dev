from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('esc_to_odom'),
            'params',
            'xmaxx_to_odom.yaml'
        ]),
        description='Path to the configuration file for esc_to_odom node'
    )
    
    telem_topic_arg = DeclareLaunchArgument(
        'telem_topic',
        default_value='/telem',
        description='Topic name for telemetry data'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='odom',
        description='Topic name for odometry output'
    )
    
    # ESC to odometry node
    esc_to_odom_node = Node(
        package="esc_to_odom",
        executable="esc_to_odom",
        name="esc_to_odom",
        output="screen",
        parameters=[LaunchConfiguration('config_file')],
        remappings=[
            ("/telem", LaunchConfiguration('telem_topic')),
            ("odom", LaunchConfiguration('odom_topic')),
        ],
    )
    
    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        telem_topic_arg,
        odom_topic_arg,
        
        # Nodes
        esc_to_odom_node,
    ])
