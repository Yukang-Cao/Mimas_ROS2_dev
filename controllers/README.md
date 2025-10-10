# Controllers Package

PyTorch-based trajectory planners (MPPI, CU-MPPI, UGE-MPC) for autonomous vehicle control.

## Quick Start

1. **Build**: `colcon build --packages-select controllers`
2. **Run**: 
   ```bash
   # Option 1: Direct node execution
   ros2 run controllers local_planner_node
   
   # Option 2: Using launch file
   ros2 launch controllers controllers.launch.py
   
   # Option 3: With custom parameters
   ros2 launch controllers controllers.launch.py controller_type:=mppi_pytorch control_frequency:=30.0
   ```

## What It Does

- **Input**: Goal pose, local costmap, odometry
- **Output**: Velocity commands (`/cmd_vel`)
- **Purpose**: Plans collision-free trajectories to reach goals

## Required Topics

**Subscribed:**
- `/goal_pose` (geometry_msgs/PoseStamped) - Target goal
- `/local_costmap_inflated` (nav_msgs/OccupancyGrid) - Obstacle map
- `/odom` (nav_msgs/Odometry) - Robot pose/velocity

**Published:**
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands

## Configuration

Edit `config/experiment_config.yaml` to customize:
- Controller type (`cu_mppi_unsupervised_std`, `mppi_pytorch`, etc.)
- Vehicle dynamics parameters
- Control limits and constraints
- Visualization settings

## Controller Types

- **MPPI**: Model Predictive Path Integral control, Log-MPPI also included
- **C-Uniform**: Neural network-based controller
- **CU-MPPI**: C-Uniform/C-Free-Uniform + MPPI/Log-MPPI, 4 combinations here
- **UGE-MPC**: Uncertainty Guided Exploration MPC

## Requirements

- CUDA-capable GPU
- PyTorch
- ROS2 (tested with Humble)
- Standard ROS2 packages (nav_msgs, geometry_msgs, tf2_ros)