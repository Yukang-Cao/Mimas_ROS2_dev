#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters

import numpy as np
import yaml
import cv2
import torch
import traceback
from cv_bridge import CvBridge

# ROS Message Types
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# For SDF and inflation processing
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation

class SemanticCostmapProcessor(Node):
    def __init__(self):
        super().__init__('semantic_costmap_processor')

        # --- Parameters ---
        self.declare_parameter('config_file_path', '')
        self.load_config()

        # --- Publishers ---
        self.binary_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_binary_raw', 10)
        self.inflated_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_inflated', 10)
        self.sdf_costmap_pub = self.create_publisher(
            OccupancyGrid, '/local_costmap_sdf', 10)
        
        # --- Subscribers ---
        # 1. Camera Info for Intrinsics
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)
        self.intrinsics = None # [fx, fy, cx, cy]

        # 2. Synchronized RGB + Depth
        rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        # --- Setup CV & Model ---
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading CLIPSeg on {self.device}...")
        
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.model.eval()

        self.get_logger().info(f"Semantic Costmap Node initialized. Resolution: {self.resolution}")

    def load_config(self):
        """Load configuration from the centralized experiment config file."""
        config_path = self.get_parameter('config_file_path').value
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Core Grid Params (Keeping these identical to LiDAR version)
        self.local_costmap_size = config['local_costmap_size']
        self.resolution = config['local_costmap_resolution']
        self.inflation_radius = config['inflation_radius']
        self.max_inflation_value = config['max_inflation_value']
        self.sdf_inflation_cells = config['sdf_inflation_cells']
        
        # Camera Specific Params
        self.max_depth = config.get('lidar_max_range', 6.0) # Reuse lidar range for consistency
        self.target_class = config.get('target_class', 'pavement')
        self.conf_threshold = config.get('conf_threshold', 0.4)
        self.robot_body_radius = config.get('robot_body_filter_radius', 0.3)

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]] # fx, fy, cx, cy

    def image_callback(self, rgb_msg, depth_msg):
        """Replaces the lidar_callback logic."""
        if self.intrinsics is None:
            return

        try:
            # 1. Image Preparation
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            depth_m = cv_depth.astype(np.float32) / 1000.0 if cv_depth.dtype == np.uint16 else cv_depth

            # 2. Semantic Inference
            rgb_input = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)
            inputs = self.processor(text=[self.target_class], images=[rgb_input], 
                                    padding="max_length", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 3. Generate Binary Mask
            mask_raw = torch.sigmoid(outputs.logits.unsqueeze(1))[0][0].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (cv_rgb.shape[1], cv_rgb.shape[0]))
            traversable_mask = (mask_resized > self.conf_threshold)

            # 4. Project Camera Data to Binary Costmap
            binary_costmap = self.create_binary_costmap_from_camera(depth_m, traversable_mask)
            
            # 5. Process Costmaps (Logic Identical to Lidar Node)
            inflated_costmap = self.create_inflated_costmap(binary_costmap)
            sdf_costmap = self.create_sdf_costmap(binary_costmap)
            
            # 6. Publish
            header = rgb_msg.header
            header.frame_id = "base_link" 
            
            self.publish_occupancy_grid(inflated_costmap, header, self.inflated_costmap_pub)
            self.publish_sdf_grid(sdf_costmap, header, self.sdf_costmap_pub)
            
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            traceback.print_exc()

    def create_binary_costmap_from_camera(self, depth_img, mask):
        """
        Calculates obstacle positions from depth and mask.
        Everything NOT in the mask is an obstacle.
        """
        binary_costmap = np.zeros((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)
        center = self.local_costmap_size // 2
        fx, fy, cx, cy = self.intrinsics

        # Identify non-traversable pixels within range
        obstacle_pixels = (~mask) & (depth_img > 0.1) & (depth_img < self.max_depth)
        v, u = np.where(obstacle_pixels)
        z_camera = depth_img[v, u]

        # Back-project to 3D
        x_camera = (u - cx) * z_camera / fx
        
        # Transform to Robot Base (Camera is forward along Z-axis in optical frame)
        # Note: Alignment matches the +X forward robot frame
        obs_x_robot = z_camera
        obs_y_robot = -x_camera

        # Filter Robot Body
        dist_from_robot = np.sqrt(obs_x_robot**2 + obs_y_robot**2)
        valid_points = dist_from_robot > self.robot_body_radius
        
        # Apply the same 90-degree rotation 'hack' from your Lidar script 
        # to match your existing visualization/system orientation
        rotated_x = obs_y_robot[valid_points]
        rotated_y = -obs_x_robot[valid_points]

        # Map to Grid
        grid_x = (rotated_x / self.resolution).astype(np.int32) + center
        grid_y = (rotated_y / self.resolution).astype(np.int32) + center

        # Bounds Check
        in_bounds = (grid_x >= 0) & (grid_x < self.local_costmap_size) & \
                    (grid_y >= 0) & (grid_y < self.local_costmap_size)
        
        binary_costmap[grid_y[in_bounds], grid_x[in_bounds]] = 1.0
        return binary_costmap

    # --- IDENTICAL COSTMAP PROCESSING LOGIC ---
    def create_inflated_costmap(self, binary_costmap):
        obstacle_mask = binary_costmap > 0.5
        distance_map = distance_transform_edt(~obstacle_mask)
        if self.inflation_radius > 0:
            inflated_costmap = np.clip((self.inflation_radius - distance_map) / self.inflation_radius, 0, 1) * self.max_inflation_value
        else:
            inflated_costmap = np.zeros_like(distance_map)
        return np.maximum(binary_costmap, inflated_costmap).astype(np.float32)

    def create_sdf_costmap(self, binary_costmap):
        sdf_obstacle_map = binary_costmap > 0.5
        if self.sdf_inflation_cells > 0:
            sdf_obstacle_map = binary_dilation(sdf_obstacle_map, iterations=self.sdf_inflation_cells)
        dist_out = distance_transform_edt(~sdf_obstacle_map) * self.resolution
        dist_in = distance_transform_edt(sdf_obstacle_map) * self.resolution
        sdf = dist_out - dist_in
        return np.flip(sdf, axis=0).copy().astype(np.float32)

    def _set_occupancy_grid_info(self, occupancy_grid):
        occupancy_grid.info.resolution = self.resolution
        occupancy_grid.info.width = self.local_costmap_size
        occupancy_grid.info.height = self.local_costmap_size
        map_extent = self.local_costmap_size * self.resolution / 2.0
        occupancy_grid.info.origin.position.x = -map_extent
        occupancy_grid.info.origin.position.y = -map_extent
        occupancy_grid.info.origin.orientation.w = 1.0

    def publish_occupancy_grid(self, costmap_data, header, publisher):
        occupancy_grid = OccupancyGrid(header=header)
        self._set_occupancy_grid_info(occupancy_grid)
        data = (costmap_data * 100).astype(np.int8)
        occupancy_grid.data = data.flatten().tolist()
        publisher.publish(occupancy_grid)
    
    def publish_sdf_grid(self, sdf_data, header, publisher):
        occupancy_grid = OccupancyGrid(header=header)
        self._set_occupancy_grid_info(occupancy_grid)
        scaled_data = (sdf_data * 10.0)
        clipped_data = np.clip(scaled_data, -128, 127).astype(np.int8)
        occupancy_grid.data = clipped_data.flatten().tolist()
        publisher.publish(occupancy_grid)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticCostmapProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()