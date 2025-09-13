# src/alphatruck_bringup/launch/start_planner.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Dynamically find the path to the configuration file in the installed workspace
    bringup_pkg = get_package_share_directory('alphatruck_bringup')
    config_file_path = os.path.join(bringup_pkg, 'params', 'experiment_config.yaml')
    urdf_file_path = os.path.join(bringup_pkg, 'urdf', 'alphatruck.urdf')

    # Define launch arguments for runtime flexibility
    controller_type_arg = DeclareLaunchArgument(
        'controller_type',
        # Set the default controller here
        default_value='cu_mppi_unsupervised_std',
        description='The type of controller to use (e.g., mppi_pytorch, cu_mppi_unsupervised_std)'
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

    # Note: joint_state_publisher not needed for fixed joints only

    # Static transform publisher for map -> odom (replace with real localization later)
    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )#TODO: replace with real localization later

    # Define the local planner node
    local_planner_node = Node(
        package='controllers',
        # The executable name defined in controllers/setup.py entry_points
        executable='local_planner_node',
        name='local_planner_node',
        output='screen',
        parameters=[
            # Pass the required parameters to the node
            {'config_file_path': config_file_path},
            {'controller_type': LaunchConfiguration('controller_type')},
            {'control_frequency': LaunchConfiguration('control_frequency')},
            
            # ROS-specific parameters
            {'map_frame': 'map'},
            {'base_link_frame': 'base_link'},
            # Use external SDF from perception package (set to False if using internal generation)
            {'use_external_sdf': True},
            {'seed': 2025},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        # Optional: Remap topics if your system uses different names
        # remappings=[
        #     ('/odom', '/alphatruck/odom'),
        #     ('/cmd_vel', '/alphatruck/cmd_vel'),
        # ]
    )

    return LaunchDescription([
        # controller_type_arg,
        # control_frequency_arg,
        use_sim_time_arg,
        robot_state_publisher,
        # static_tf_map_odom,
        # local_planner_node
    ])