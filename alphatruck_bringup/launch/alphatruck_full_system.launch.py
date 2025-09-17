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
        # default_value='cu_mppi_map_conditioned_log',
        # default_value = 'cu_mppi_unsupervised_log',
        default_value='mppi_pytorch',
        # default_value='log_mppi_pytorch',
        description='The type of controller to use (e.g., mppi_pytorch, cu_mppi_unsupervised_std, cu_mppi_map_conditioned_std)'
    )

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

    # LiDAR configuration arguments (pass through to perception)
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB1',
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

    # broadcast TF from the motion capture system's pose topic
    mocap_tf_broadcaster_node = Node(
        package='mocap_tf_broadcaster',
        executable='broadcaster_node',
        name='mocap_tf_broadcaster_node',
        output='screen'
    )

    # Static transform publisher for map -> odom (replace with real localization later)
    #static_tf_map_odom = Node(
    #    package='tf2_ros',
    #    executable='static_transform_publisher',
    #    name='static_tf_map_odom',
    #    arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    #    output='screen'
    #)

    # Step 1: Launch perception system (LiDAR + costmap processing)
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch_path),
        launch_arguments={
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baudrate': LaunchConfiguration('serial_baudrate'),
            'config_file_path': config_file_path,
        }.items()
    )

    # # Step 2: Launch dummy odometry publisher (needed for controller)
    # dummy_odom_node = Node(
    #     package='perception',
    #     executable='dummy_odom_publisher',
    #     name='dummy_odom_publisher',
    #     output='screen',
    # )
    
    # Step 2: Launch RF2O laser odometry
    rf2o_pkg = get_package_share_directory('rf2o_laser_odometry')
    rf2o_launch_path = os.path.join(rf2o_pkg, 'launch', 'rf2o_laser_odometry.launch.py')
    
    rf2o_laser_odometry_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rf2o_launch_path)
    )

    # Step 3: Launch VectorNav IMU
    vectornav_pkg = get_package_share_directory('vectornav')
    vectornav_launch_path = os.path.join(vectornav_pkg, 'launch', 'vectornav.launch.py')
    
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vectornav_launch_path)
    )

    # Step 4: Launch EKF for sensor fusion
    ekf_config_path = os.path.join(bringup_pkg, 'config', 'ekf.yaml')
    
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        remappings=[('odometry/filtered', '/odometry/filtered')]
    )

    # Step 5: Launch test goal publisher (for testing and demonstration)
    test_goal_node = Node(
        package='perception',
        executable='test_goal_publisher',
        name='test_goal_publisher',
        output='screen',
    )

    # Step 6: Launch local planner with delay to ensure all prerequisites are ready
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
        robot_state_publisher,       # For base_link -> lidar_link etc
        #static_tf_map_odom,
        # NOTE: dont' forget to start the mocap node if using this
        mocap_tf_broadcaster_node,   # For world -> base_link
        
        # Sequential launch with time delay: Perception -> Laser Odom -> IMU -> EKF -> Planner
        perception_launch,              # Step 1: LiDAR + costmap processing (t=0)
        # TimerAction(
        #    period=5.0,
        #   actions=[rf2o_laser_odometry_launch]  # Step 2: laser odometry (t=5s)
        #),
        #TimerAction(
        #    period=7.0,
        #    actions=[imu_launch]                  # Step 3: VectorNav IMU (t=7s)
        #),
        #TimerAction(
        #    period=8.0,
        #    actions=[ekf_node]                    # Step 4: EKF sensor fusion (t=8s)
        #),
        TimerAction(
            period=9.0,
            actions=[test_goal_node]              # Step 5: Test goal publisher (t=9s)
        ),
        TimerAction(
            period=7.5,
            actions=[
                Node(
                    package='xmaxx_bringup',
                    executable='xmaxx_interface_node',
                    name='xmaxx_interface_node',
                    output='screen',
                    parameters=[
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}
                    ]
                )
            ]                                    # Step 5.5: Xmaxx hardware interface (t=9.5s)
        ),
        TimerAction(
            period=10.0,
            actions=[local_planner_node]         # Step 6: Local planner (t=10s)
        )
    ])
