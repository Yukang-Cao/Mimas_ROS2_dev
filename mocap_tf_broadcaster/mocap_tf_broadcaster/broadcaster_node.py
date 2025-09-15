import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PoseToTfNode(Node):
    def __init__(self):
        super().__init__('pose_to_tf_broadcaster')
        
        # The name of the robot, used for the child frame ID
        self.robot_name = 'titan_alphatruck' 
        
        # The target child frame ID for the transform (e.g., base_link)
        self.child_frame_id = 'base_link' 
        
        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # 1. Define the QoS profile to match the publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. Apply the QoS profile to the subscription
        self.subscription = self.create_subscription(
            PoseStamped,
            f'/vrpn_mocap/{self.robot_name}/pose',
            self.pose_callback,
            qos_profile
        )

        self.get_logger().info(f"Subscribed to /vrpn_mocap/{self.robot_name}/pose")
        self.get_logger().info(f"Broadcasting transform from 'world' to '{self.child_frame_id}'")

    def pose_callback(self, msg):
        t = TransformStamped()

        # Read message content and assign it to the transform
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = msg.header.frame_id  # This should be 'world'
        t.child_frame_id = self.child_frame_id

        # Copy the position and orientation from the PoseStamped message
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        # Send the transform
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = PoseToTfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
