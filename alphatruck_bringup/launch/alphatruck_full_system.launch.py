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
    bringup_pkg = get_package_share_directory('alphatruck_bringup')
    perception_pkg = get_package_share_directory('perception')
    
    # Configuration paths
    config_file_path = os.path.join(bringup_pkg, 'params', 'experiment_config.yaml')
    urdf_file_path = os.path.join(bringup_pkg, 'urdf', 'alphatruck.urdf')
    
    # Launch file paths
    perception_launch_path = os.path.join(perception_pkg, 'launch', 'perception_full.launch.py')

    # Define launch arguments for runtime flexibility
    controller_type_arg = DeclareLaunchArgument(
        'controller_type',
        default_value='cu_mppi_map_conditioned_std',
        description='The type of controller to use (e.g., mppi_pytorch, cu_mppi_unsupervised_std, cu_mppi_map_conditioned_std)'
    )

    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='20.0',
        description='Frequency (Hz) of the control loop'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # LiDAR configuration arguments (pass through to perception)
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port for LiDAR connection'
    )

    serial_baudrate_arg = DeclareLaunchArgument(
        'serial_baudrate',
        default_value='1000000',
        description='Baudrate for LiDAR connection'
    )

    # Robot description parameter (read URDF file directly)
    with open(urdf_file_path, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # Static transform publisher for map -> odom (replace with real localization later)
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # Step 1: Launch perception system (LiDAR + costmap processing)
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch_path),
        launch_arguments={
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baudrate': LaunchConfiguration('serial_baudrate'),
            'config_file_path': config_file_path,
        }.items()
    )

    # Step 2: Launch dummy odometry publisher (needed for controller)
    dummy_odom_node = Node(
        package='perception',
        executable='dummy_odom_publisher',
        name='dummy_odom_publisher',
        output='screen',
    )

    # Step 3: Launch test goal publisher (for testing and demonstration)
    test_goal_node = Node(
        package='perception',
        executable='test_goal_publisher',
        name='test_goal_publisher',
        output='screen',
    )

    # Step 4: Launch local planner with delay to ensure all prerequisites are ready
    local_planner_node = Node(
        package='controllers',
        executable='local_planner_node',
        name='local_planner_node',
        output='screen',
        parameters=[
            # Pass the required parameters to the node
            {'config_file_path': config_file_path},
            {'controller_type': LaunchConfiguration('controller_type')},
            {'control_frequency': LaunchConfiguration('control_frequency')},
            {'map_frame': 'map'},
            {'base_link_frame': 'base_link'},
            {'use_external_sdf': True},
            {'seed': 2025},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    return LaunchDescription([
        # Launch arguments
        controller_type_arg,
        control_frequency_arg,
        use_sim_time_arg,
        serial_port_arg,
        serial_baudrate_arg,
        
        # Robot description and TF
        robot_state_publisher,
        static_tf_map_odom,
        
        # Sequential launch: Perception -> Dummy Nodes -> Planner
        perception_launch,     # Step 1: LiDAR + costmap processing (t=0)
        dummy_odom_node,       # Step 2: Dummy odometry (t=4s)
        test_goal_node,        # Step 3: Test goal publisher (t=5s)
        local_planner_node,    # Step 4: Local planner (t=6s)
    ])
