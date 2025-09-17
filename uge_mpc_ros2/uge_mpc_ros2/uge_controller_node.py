# uge_mpc_ros2/subscriber_node.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
# from nav_msgs.msg import Odometry   # Uncomment if you want to use odometry later
import numpy as np
import yaml
import os
import sys
import time
import traceback
import math
import copy
import warnings
from collections import deque
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

# ROS Message Types
from geometry_msgs.msg import Twist, PoseStamped, Point, TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray, Marker
from ament_index_python.packages import get_package_share_directory
from uge_mpc_ros2.scripts.uge_gpu import Config as UGE_Config
from uge_mpc_ros2.scripts.uge_gpu import UAE_Numba
from uge_mpc_ros2.utils.vehicle import load_config_yaml, vehicle_from_config
from uge_mpc_ros2.scripts.uae_method_3d_TO import UAE_method as UAE_method_3d
# TF2
import tf2_ros
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion
# Explicit import needed for do_transform_pose
import tf2_geometry_msgs 

import numba
from numba import cuda as numba_cuda
gpu = numba_cuda.get_current_device()
print(numba_cuda.is_available())
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e6) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100

class UGEControllerNode(Node):
    def __init__(self):
        super().__init__('uge_controller')

        # Parameters
        self.declare_parameters(
                    namespace='',
                    parameters=[
                        ('config_file_path', 'path/to/your/experiment_config.yaml'),
                        ('vehicle_config_file_path', 'path/to/your/vehicle_config.yaml'),
                        ('control_frequency', 8.0),
                        ('map_frame', 'map'),
                        ('base_link_frame', 'base_link'),
                        ('seed', 2025)
                    ]
                )
        
        
        self.config_file_path = self.get_parameter('config_file_path').value
        self.vehicle_config_file_path = self.get_parameter('vehicle_config_file_path').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.control_dt = 1.0 / self.get_parameter('control_frequency').value
        self.seed = self.get_parameter('seed').value

        
        
        
        self.config = self.load_configuration(self.config_file_path)
        if self.config is None:
            self.get_logger().fatal(f"Configuration file not found at: {self.config_file_path}")
            return

        self.vehicle = vehicle_from_config(self.vehicle_config_file_path)

        self.T = self.config['T']
        self.control_dt = self.config['control_dt']
        self.step_val = self.config['step_val']


        self.v_bounds = self.vehicle.v_bounds
        self.delta_bounds = self.vehicle.delta_bounds

        self.Sigma0_val = self.config['Sigma0']
        self.Sigma0 = np.diag(np.array(self.Sigma0_val, dtype=np.float32)).astype(np.float32)
        self.Q_val = self.config['Q']
        self.Q = np.diag(np.array([self.Q_val[0]**2, np.deg2rad(self.Q_val[1])**2], dtype=np.float32)).astype(np.float32)
        self.R_val = self.config['R']
        self.R = np.diag(np.array([self.R_val[0]**2, np.deg2rad(self.R_val[1])**2], dtype=np.float32)).astype(np.float32)
        self.num_control_rollouts = self.config['num_control_rollouts']
        self.rec_min_control_rollouts = self.config['rec_min_control_rollouts']



        self.uge_config = UGE_Config(
            T=self.T,
            dt=self.control_dt,
            num_control_rollouts=self.num_control_rollouts,
            seed=self.seed,
            max_threads_per_block=max_threads_per_block,
            max_square_block_dim=max_square_block_dim,
            max_blocks=max_blocks,
            max_rec_blocks=max_rec_blocks,
            min_control_rollouts=self.rec_min_control_rollouts)

        self.uae = UAE_method_3d(self.vehicle, 
                            dtype=np.float32, 
                            discount_factor=1.0, 
                            dist_weight=1.5, 
                            theta_weight=1.0, 
                            terminal_weight=15.0, 
                            action_weight=0.5, 
                            obstacle_weight=100.0)

        # UAE PARAMS
        self.trajs = 8
        self.iters = 6
        self.candidates_per_traj = 16
        self.decay_sharpness = 2.0
        # --- Warm-up JIT (don’t count compile time) ---
        dummy_x0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        dummy_base_actions = np.zeros((self.T, 2), dtype=np.float32)
        self.uae.optimize3D(dummy_x0, dummy_base_actions, self.R, self.Sigma0, self.Q, iters=1)
        self.get_logger().info("JIT warm-up complete")
        self.get_logger().info("="*50)


        self.uae_params = dict(
                    # Task specification
                    dt = self.uge_config.dt, 
                    x0 = np.array([0.0, 0.0, 0.0], dtype=np.float32), # Start state
                    xgoal = None, # Goal position (x, y, theta)
                    # For risk-aware min time planning
                    goal_tolerance = 0.2,
                    dist_weight = 8.0, #  Weight for dist-to-goal cost.
                    obs_penalty = 2e3, # MPC 
                    # dist_weight = 1e4, #  Weight for dist-to-goal cost.

                    lambda_weight = 0.5, # Temperature param in MPPI
                    num_opt = 1, # Number of steps in each solve() function call.

                    vehicle_length = self.vehicle.length,
                    vehicle_width = self.vehicle.width,
                    vehicle_wheelbase = self.vehicle.wheelbase,
                    # Control and sample specification
                    u_std = self.R.diagonal(), # Noise std for sampling linear and angular velocities.
                    vrange = np.array([self.v_bounds[0], self.v_bounds[1]]), # Linear velocity range.
                    wrange = self.delta_bounds, # steering angle range.
                    costmap = None, # circles to occupancy grid
                    map_mins = None,
                    costmap_meta = None
                )

        self.uae_cost_planner = UAE_Numba(self.uge_config)
        self.uae_cost_planner.setup(self.uae_params)

        # --- State Variables ---j
        self.current_velocity = 0.0
        self.base_actions = None
        self.current_pose = None
        self.global_goal = None
        self.latest_costmap_msg = None
        self.latest_sdf_msg = None
        self.goal_received_once = False  # Flag to prevent repeated goal resets
        
        self.viz = True
        self.smoothing_window = 2               # number of control steps to average
        self.control_buffer = deque(maxlen=self.smoothing_window)
        self._smoothing_warmed_up = False
        # --- TF2 Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        
        #--- Publishers ---#
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.setup_visualization_publishers()
        #--- Subscriptions ---#
        # QoS profile for sensor-like topics
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,depth=1)

        # Subscriptions
        # self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, sensor_qos)
        #TODO: add pose subscription for the robot
        self.create_subscription(PoseStamped, '/vrpn_mocap/titan_alphatruck/pose', self.pose_callback, sensor_qos)
        self.create_subscription(TwistStamped, '/vrpn_mocap/titan_alphatruck/twist', self.twist_callback, sensor_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap_inflated', self.costmap_callback, sensor_qos)
        
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)

        # initilization complete
        self.get_logger().info("UGE-MPC controller initialized")

    # --- Callbacks ---
    def pose_callback(self, msg: PoseStamped):
        # self.get_logger().info(f"Pose received: {msg.pose.position}")
        self.current_pose = msg
        # quaternion to euler
        euler = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.get_logger().info(f"Euler: {euler}")
        self.uae_cost_planner.params['x0'] =  self.uae.curr_x = np.array([msg.pose.position.x, msg.pose.position.y, euler[2]], dtype=np.float32) # RPY
        # self.uae_cost_planner.params['x0'] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.z], dtype=np.float32)

    def twist_callback(self, msg: TwistStamped):
        # self.get_logger().info(f"Twist received: {msg.twist}")
        self.current_velocity = msg.twist.linear.x

    def goal_callback(self, msg: PoseStamped):
        # self.get_logger().info(f"Goal received: {msg.pose.position}")
        # selg.global_goal_point = msg.pose.position
        self.global_goal = msg
        self.uae_cost_planner.params['xgoal'] = np.array([msg.pose.position.x, msg.pose.position.y, 0.0], dtype=np.float32)

    def costmap_callback(self, msg: OccupancyGrid):
        # self.get_logger().info(f"Costmap received: {msg.info.width}x{msg.info.height}")
        self.latest_costmap_msg = msg
    # --- Main Control Loop Timer ---
    def control_loop(self):
        if not self.is_ready():
            # pass
            # Waiting for data, do not necessarily stop the robot yet.
            self.get_logger().info("Control loop not ready - waiting for goal, costmap, or SDF", throttle_duration_sec=5.0)
            return
        
        # 1. Transform Goal
        local_goal_np = self.transform_goal_to_local()
        self.get_logger().info(f"Local goal: {local_goal_np}")
        # if local_goal_np is None:
        #     # self.publish_stop_command()
        #     self.get_logger().error("Failed to transform goal. Stopping robot.", throttle_duration_sec=1.0)
        #     return
        # else:        
            
            # pass
        # 2. Process Costmap
        costmap_np = self.process_costmap(self.latest_costmap_msg)
        if costmap_np is None:
            # self.publish_stop_command()
            self.get_logger().error("Failed to process costmap. Stopping robot.", throttle_duration_sec=1.0)
            return
        else:
            self.uae_cost_planner.load_costmap(costmap_np)
            # pass
        # 3. Execute Planning by UGE-MPC
        '''
        It nees to take the following inputs:  the local goal and the costmap
        '''
        # 3.1 Run the UAE-TO (uae)
        self.uae.curr_x = self.uae_cost_planner.params['x0']
        # self.get_logger().info(f"Current x: {self.uae.curr_x}")
        # self.get_logger().info(f"Global goal: {self.uae_cost_planner.params['xgoal']}")
        start_time = time.monotonic()
        action_seqs, final_trajs = self.uae.optimize3D(
                                    self.uae.curr_x, self.base_actions, self.R, self.Sigma0, self.Q,
                                    num_trajs=self.trajs, iters=self.iters, candidates_per_traj=self.candidates_per_traj, step_interval=self.step_val,
                                    v_bounds=self.v_bounds, delta_bounds=self.delta_bounds, random_seed=self.seed,
                                    decay_sharpness=self.decay_sharpness)
        # calucalte the cost
        # self.get_logger().info(f"Final trajectories: {final_trajs}")
        self.uae_cost_planner.load_trajectories(final_trajs)
        min_cost_idx, total_cost_uae = self.uae_cost_planner.solve_uae(final_trajs)

        useq_uae = action_seqs[min_cost_idx]

        self.get_logger().info(f"Min cost index: {min_cost_idx}")
        # self.get_logger().info(f"First action: {action_seqs[min_cost_idx][0]}")

        rollout_actions, rollout_states = self.uae.generate_rollouts(
                                useq_uae, self.T, self.uae.curr_x, noise_covariance=self.R, v_bounds=self.v_bounds, delta_bounds=self.delta_bounds, num_rollouts=256)
                                # solve the mppi update
        self.uae_cost_planner.load_trajectories(rollout_states)
        min_cost_idx_uae_mppi, total_cost_mppi = self.uae_cost_planner.solve_uae(rollout_states)
                                
        useq_mppi, min_cost_idx_mppi = self.uae.solve_mppi_only_with_total_cost(
                                    total_cost_mppi, rollout_actions, rollout_actions[0], 
                                    lambda_weight=0.001, v_bounds=self.v_bounds, delta_bounds=self.delta_bounds)
        useq = copy.deepcopy(useq_mppi)
        self.get_logger().info(f"Useq: {useq[:2]}")
        self.get_logger().info(f"Useq 0 : {useq[0]}")
        # self.get_logger().info(f"Mean useq: {np.mean(useq[:5], axis=0)}")
        self.publish_control_command(np.mean(useq[:2], axis=0)) # first 5 actions

        # u_now = np.asarray(useq[0], dtype=np.float32)      # shape (2,) -> [v, delta]
        # self.control_buffer.append(u_now)

        # if len(self.control_buffer) < self.smoothing_window:
        #     # Not enough samples yet: hold still (or you could pass-through if you prefer)
        #     if not self._smoothing_warmed_up:
        #         self.get_logger().info(
        #             f"Smoother warming up: {len(self.control_buffer)}/{self.smoothing_window} controls collected; holding."
        #         )
        #     self.publish_stop_command()
        # else:
        #     if not self._smoothing_warmed_up:
        #         self._smoothing_warmed_up = True
        #         self.get_logger().info("Smoother ready: publishing averaged controls.")

        #     # Average the last 5 controls
        #     u_avg = np.mean(np.stack(self.control_buffer, axis=0), axis=0)
        #     # Optional: clamp to vehicle bounds for safety (recommended)
        #     u_avg[0] = float(np.clip(u_avg[0], self.v_bounds[0], self.v_bounds[1]))     # v
        #     u_avg[1] = float(np.clip(u_avg[1], self.delta_bounds[0], self.delta_bounds[1]))  # delta

        #     self.get_logger().info(f"Publishing smoothed useq: v={u_avg[0]:.3f}, delta={u_avg[1]:.3f}")
        #     self.publish_control_command(u_avg)

        total_time = time.monotonic() - start_time
        total_time_ms = total_time * 1000
        self.get_logger().info(f"Total time: {total_time_ms} ms")
        # update the base actions
        self.base_actions = useq

        viz_start = time.monotonic()
        if self.viz:
            # Visualize Trajectories
            self.visualize_trajectories(self.uae_cost_planner.trajectories_d.copy_to_host())
            self.visualize_local_goal(local_goal_np)
            
            # Visualize Goal

        # 6. Publish the Command
        # self.publish_command(useq_mpc)
    def is_ready(self):
        """Check if all required data is available."""
        # initialize the base actions
        if self.base_actions is None:
            self.base_actions = np.random.uniform(
                                    low=np.array([1.0,-0.02], dtype=np.float32),
                                    high=np.array([1.0, 0.02], dtype=np.float32),
                                    size=(self.T, 2)).astype(np.float32)
            self.base_actions[:, 0] = np.clip(self.base_actions[:, 0], self.v_bounds[0], self.v_bounds[1])
            self.base_actions[:, 1] = np.clip(self.base_actions[:, 1], self.delta_bounds[0], self.delta_bounds[1])
        return (
            self.global_goal is not None and
            self.latest_costmap_msg is not None
        )

    def transform_goal_to_local(self):
        """Transforms the global goal into the robot's local frame (base_link) using TF2."""
        if self.global_goal is None:
            return None
            
        try:
            # transform from the goal's frame to the robot frame
            transform = self.tf_buffer.lookup_transform(
                self.base_link_frame,             # Target frame (robot)
                self.global_goal.header.frame_id, # Source frame (e.g., map)
                rclpy.time.Time()                 # Use the latest available transform
            )
            
            # Apply the transform to the goal pose
            local_goal_pose = tf2_geometry_msgs.do_transform_pose(self.global_goal.pose, transform)
            
            return np.array([local_goal_pose.position.x, local_goal_pose.position.y], dtype=np.float32)

        except TransformException as e:
            self.get_logger().warn(f"Could not transform goal to robot frame: {e}", throttle_duration_sec=1.0)
            return None
            
    def process_costmap(self, msg: OccupancyGrid):
        """Converts OccupancyGrid to numpy array, handling normalization and orientation."""
        try:
            width = msg.info.width
            height = msg.info.height
            
            data = np.array(msg.data, dtype=np.int8).reshape(height, width)
            
            # Convert ROS standard (0-100) to controller's expectation (0.0-1.0).
            # Handle unknown space (-1). Treat as free space (0.0).
            costmap_np = np.clip(data, 0, 100).astype(np.float32) / 100.0
            return costmap_np
        except Exception as e:
            self.get_logger().error(f"Error processing costmap: {e}")
            return None

    def publish_control_command(self, control_action: np.ndarray):
            """Publishes the control action, converting steering angle to angular velocity."""
            twist = Twist()
            v = float(control_action[0])
            # The controller library outputs steering angle (delta)
            delta = float(control_action[1])
            omega = -delta # NOTE: directly publish the steering angle as command to servo, add - to correct transformation
            
            # # Convert steering angle to angular velocity (omega) for the Twist message
            # # omega = v * tan(delta) / L
            # L = self.controller.wheelbase
            # if L > 0 and not math.isnan(delta):
            #     omega = v * np.tan(delta) / L
            # else:
            #     omega = 0.0

            twist.linear.x = v
            twist.angular.z = omega
            self.cmd_vel_pub.publish(twist)

    def setup_visualization_publishers(self):
        """Sets up publishers for visualization."""
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.traj_marker_pub = self.create_publisher(MarkerArray, '/visualization/sampled_trajectories', vis_qos)
        self.local_goal_pub = self.create_publisher(Marker, '/visualization/local_goal', vis_qos)
        # self.nominal_path_pub = self.create_publisher(Path, '/visualization/nominal_trajectory', vis_qos)


    def load_configuration(self, path):
            """Loads the YAML configuration and detects velocity mode."""
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                # Determine velocity mode automatically (required by the library initialization)
                return config
            except yaml.YAMLError as e:
                self.get_logger().fatal(f"Error parsing YAML file: {e}")
                return None

    def visualize_trajectories(self, rollouts_data, is_hybrid=False):
        """Visualizes trajectories in the robot frame using MarkerArrays."""
        # Visualization is published in the robot frame (base_link_frame) 
        # as the trajectories are generated locally. RViz handles the transformation.
        
        self.get_logger().debug(f"Visualizing trajectories - Hybrid: {is_hybrid}, Data type: {type(rollouts_data)}")
        
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = self.base_link_frame
        clear_marker.header.stamp = now
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        if is_hybrid:
            # Hybrid Visualization (CU-MPPI) - 4 colors
            self.get_logger().debug(f"Hybrid mode - Keys available: {list(rollouts_data.keys()) if isinstance(rollouts_data, dict) else 'Not a dict'}")
            self._add_trajectory_marker(markers, rollouts_data.get('cu_samples'), "cu_samples", now, color=(0.5, 0.5, 0.5, 0.2), width=0.01) # Grey
            self._add_trajectory_marker(markers, rollouts_data.get('cu_best'), "cu_best", now, color=(0.0, 0.0, 1.0, 0.5), width=0.03)       # Blue
            self._add_trajectory_marker(markers, rollouts_data.get('mppi_samples'), "mppi_samples", now, color=(1.0, 0.65, 0.0, 0.3), width=0.01) # Orange
            self._add_trajectory_marker(markers, rollouts_data.get('mppi_nominal'), "mppi_nominal", now, color=(0.0, 1.0, 0.0, 0.8), width=0.05) # Green
        else:
            # Standard Visualization (MPPI/C-Uniform)
            self.get_logger().debug(f"Standard mode - Rollouts shape: {rollouts_data.shape if hasattr(rollouts_data, 'shape') else 'No shape attr'}")
            if hasattr(rollouts_data, 'shape') and len(rollouts_data.shape) >= 1 and rollouts_data.shape[0] > 0:
                # Best trajectory (Index 0)
                self._add_trajectory_marker(markers, rollouts_data[0:1], "best_trajectory", now, color=(0.0, 1.0, 0.0, 0.8), width=0.05) # Green
                # Samples (Index 1+)
                if rollouts_data.shape[0] > 1:
                    self._add_trajectory_marker(markers, rollouts_data[1:], "samples", now, color=(0.5, 0.5, 0.5, 0.3), width=0.01) # Grey

        self.get_logger().debug(f"Publishing {len(markers.markers)} markers")
        self.traj_marker_pub.publish(markers)

    def _add_trajectory_marker(self, markers, trajectories, ns, stamp, color, width):
        """Helper to add trajectories (Numpy arrays) to a MarkerArray."""
        if trajectories is None or (isinstance(trajectories, np.ndarray) and trajectories.size == 0):
            return
        
        # Ensure trajectories are 3D (N, T, 3) even if N=1
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]

        for i in range(trajectories.shape[0]):
            marker = Marker()
            marker.header.frame_id = self.base_link_frame
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0 # Identity pose (points are already in the correct frame)
            marker.scale.x = width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            
            # Add points
            for t in range(trajectories.shape[1]):
                p = Point()
                p.x = float(trajectories[i, t, 0])
                p.y = float(trajectories[i, t, 1])
                p.z = 0.05 # Visualize slightly above the ground
                marker.points.append(p)
            
            markers.markers.append(marker)


    def visualize_local_goal(self, local_goal_np):
        """Visualizes the local goal in the robot frame."""
        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(local_goal_np[0])
        marker.pose.position.y = float(local_goal_np[1])
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        # Use a visible scale (e.g., 0.3m diameter sphere)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        # Magenta color
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.0, 1.0, 0.8)
        self.local_goal_pub.publish(marker)



    def publish_stop_command(self):
        """Publishes zero commands (stop) while the smoother warms up or on failure."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    
        
    
        

def main(args=None):
    rclpy.init(args=args)
    node = UGEControllerNode()
    rclpy.spin(node)
    node.destroy_node()
