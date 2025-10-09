# ESC to Odometry (esc_to_odom)

Converts ESC telemetry data to wheel odometry for wheeled vehicles using xmaxx_msgs.

## Usage

```bash
# ESC to odometry only
ros2 launch esc_to_odom esc_to_odom_only.launch.py

# With EKF sensor fusion
ros2 launch esc_to_odom esc_to_odom.launch.py
```

## Configuration

Edit `params/xmaxx_to_odom.yaml` for vehicle-specific parameters:
- Throttle-to-speed mapping
- Steering angle conversion  
- Vehicle geometry (wheelbase)

## Topics

**Subscribed:** `/telem` (xmaxx_msgs/XmaxxTelem)  
**Published:** `odom` (nav_msgs/Odometry)

## Dependencies

- xmaxx_msgs
- robot_localization (optional, for EKF)
- Standard ROS2 packages (nav_msgs, geometry_msgs, tf2_ros)

## Maintainer

Yukang (mikasa.cyk@gmail.com)
