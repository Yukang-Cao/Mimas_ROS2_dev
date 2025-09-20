from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    
    esc_to_odom_node = Node(
            package="esc_to_odom",              # your package name
            executable="esc_to_odom",          # entry point from setup.cfg
            name="esc_to_odom",
            output="screen",
            parameters=["/home/alphatruck/ros2_ws/src/esc_to_odom/params/xmaxx_to_odom.yaml"],  # replace with actual path
            remappings=[
                ("/telem", "/telem"),            # adjust if your telemetry topic differs
                ("odom", "odom"),
            ],
        )
    
    ekf_config_path = "/home/alphatruck/ros2_ws/src/esc_to_odom/params/ekf_wheel.yaml"
    
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        remappings=[('odometry/filtered', '/odometry/filtered')]
    )
    
    
    
    
    return LaunchDescription([
        esc_to_odom_node,
        ekf_node,
    ])
