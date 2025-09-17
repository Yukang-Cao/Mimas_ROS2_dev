# uge_mpc_ros2/subscriber_node.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
# from nav_msgs.msg import Odometry   # Uncomment if you want to use odometry later


class UgeSubscriber(Node):
    def __init__(self):
        super().__init__('uge_subscriber')

        # QoS profile for sensor-like topics
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,depth=1)

        # Subscriptions
        # self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, sensor_qos)
        self.create_subscription(TwistStamped, '/vrpn_mocap/titan_alphatruck/twist', self.twist_callback, sensor_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap_inflated', self.costmap_callback, sensor_qos)
        

    # --- Callbacks ---
    def twist_callback(self, msg: TwistStamped):
        self.get_logger().info(f"Twist received: {msg.twist}")

    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Goal received: {msg.pose.position}")

    def costmap_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Costmap received: {msg.info.width}x{msg.info.height}")

    # def odom_callback(self, msg: Odometry):
    #     self.get_logger().info("Odometry received")


def main(args=None):
    rclpy.init(args=args)
    node = UgeSubscriber()
    rclpy.spin(node)
    node.destroy_node()
