#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    uge_mpc_pkg = get_package_share_directory('uge_mpc_ros2')
    # Configuration paths
    config_file_path = os.path.join(uge_mpc_pkg, 'params', 'controller_config.yaml')
    vehicle_config_file_path = os.path.join(uge_mpc_pkg, 'params', 'vehicle_config.yaml')
    # urdf_file_path = os.path.join(uge_mpc_pkg, 'urdf', 'alphatruck.urdf')
    
    # Launch file paths
    # perception_launch_path = os.path.join(perception_pkg, 'launch', 'perception_full.launch.py')

    # Define launch arguments for runtime flexibility

    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='10.0',
        description='Frequency (Hz) of the control loop'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )


    # Step 6: Launch local planner with delay to ensure all prerequisites are ready
    uge_controller_node_ld = Node(
        package='uge_mpc_ros2',
        executable='uge_controller_node',
        name='uge_controller_node',
        output='screen',
        parameters=[
            # Pass the required parameters to the node
            {'config_file_path': config_file_path},
            {'vehicle_config_file_path': vehicle_config_file_path},
            {'control_frequency': LaunchConfiguration('control_frequency')},
            {'map_frame': 'map'},
            {'base_link_frame': 'base_link'},
            {'seed': 2025},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    return LaunchDescription([
        # Launch arguments
        control_frequency_arg,
        use_sim_time_arg,
        uge_controller_node_ld
    ])
