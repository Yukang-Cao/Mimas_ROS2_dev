#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math

class TestGoalPublisher(Node):
    def __init__(self):
        super().__init__('test_goal_publisher')
        
        # Publisher for goal pose
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Timer to publish goals periodically
        self.timer = self.create_timer(5.0, self.publish_goal)  # Publish every 5 seconds
        
        self.goal_counter = 0
        
        self.get_logger().info("Test Goal Publisher initialized")

    def publish_goal(self):
        """Publish a test goal pose in base_link frame for testing."""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        # goal_msg.header.frame_id = "world"
        goal_msg.header.frame_id = "base_link"
        
        # Create different goal positions for testing
        goals = [
            (4.0, -1.0)
        ]
        
        goal_x, goal_y = goals[self.goal_counter % len(goals)]
        
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = 0.0
        
        # Set orientation (facing forward)
        goal_msg.pose.orientation.w = 1.0
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"Published goal #{self.goal_counter}: ({goal_x:.1f}, {goal_y:.1f})")
        
        self.goal_counter += 1


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TestGoalPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
