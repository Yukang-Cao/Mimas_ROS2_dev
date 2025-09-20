#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from xmaxx_msgs.msg import XmaxxTelem


class XmaxxToOdom(Node):
    """
    - Subscribes: /telem (xmaxx_msgs/XmaxxTelem)
    - Publishes : /odom (nav_msgs/Odometry) [+ /tf if publish_tf]

    Params:
      odom_frame: "odom"
      base_frame: "base_link"
      publish_tf: true

      # Throttle (raw ~1000..2000) -> speed (m/s)
      throttle_to_speed_gain:   0.0352
      throttle_to_speed_offset: -54.15
      throttle_deadband_speed:  0.05
      switch_a_threshold:       1500.0   # if rc_switch_a > threshold -> use ac_throttle, else rc_throttle

      # Steering raw -> wheel angle (rad)
      steering_max_right: 1950
      steering_max_left: 1000
      steering_straight: 1482
      steering_max_angle: 0.4014

      # Geometry
      wheelbase: 0.48
    """

    def __init__(self):
        super().__init__("xmaxx_to_odom_node")

        # Frames & TF
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", False)

        # Throttle→speed (shared mapping)
        self.declare_parameter("positive_throttle_to_speed_gain", 0.0352)
        self.declare_parameter("negative_throttle_to_speed_gain", 0.0092)
        self.declare_parameter("positive_throttle_to_speed_offset", -54.15)
        self.declare_parameter("negative_throttle_to_speed_offset", -13.07)
        self.declare_parameter("positive_throttle_deadband_value", 1539.0)
        self.declare_parameter("negative_throttle_deadband_value", 1415.0)
        self.declare_parameter("switch_a_threshold", 1500.0)

        # Steering & geometry
        self.declare_parameter("steering_max_right", 1950.0)
        self.declare_parameter("steering_max_left", 1000.0)
        self.declare_parameter("steering_straight", 1482.0)
        self.declare_parameter("steering_max_angle", 0.4014)
        self.declare_parameter("wheelbase", 0.48)

        self.odom_frame = self.get_parameter("odom_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value

        self.th_pos_gain = self.get_parameter("positive_throttle_to_speed_gain").get_parameter_value().double_value
        self.th_pos_off  = self.get_parameter("positive_throttle_to_speed_offset").get_parameter_value().double_value
        self.th_pos_dead  = self.get_parameter("positive_throttle_deadband_value").get_parameter_value().double_value

        self.th_neg_gain = self.get_parameter("negative_throttle_to_speed_gain").get_parameter_value().double_value
        self.th_neg_off  = self.get_parameter("negative_throttle_to_speed_offset").get_parameter_value().double_value
        self.th_neg_dead  = self.get_parameter("negative_throttle_deadband_value").get_parameter_value().double_value
        
        self.sw_thr  = self.get_parameter("switch_a_threshold").get_parameter_value().double_value

        self.steer_max_right = self.get_parameter("steering_max_right").get_parameter_value().double_value
        self.steer_max_left = self.get_parameter("steering_max_left").get_parameter_value().double_value
        self.steer_straight = self.get_parameter("steering_straight").get_parameter_value().double_value
        self.steer_max_angle = self.get_parameter("steering_max_angle").get_parameter_value().double_value

        self.wheelbase = self.get_parameter("wheelbase").get_parameter_value().double_value
        self.lr = self.lf = self.wheelbase / 2

        # debug
        self.get_logger().info(f"steer_max_right: {self.steer_max_right}")
        self.get_logger().info(f"steer_max_left: {self.steer_max_left}")
        self.get_logger().info(f"steer_straight: {self.steer_straight}")
        self.get_logger().info(f"steer_max_angle: {self.steer_max_angle}")
        self.get_logger().info(f"wheelbase: {self.wheelbase}")

        self.get_logger().info(f"th_pos_gain: {self.th_pos_gain}")
        self.get_logger().info(f"th_pos_off: {self.th_pos_off}")
        self.get_logger().info(f"th_pos_dead: {self.th_pos_dead}")
        self.get_logger().info(f"th_neg_gain: {self.th_neg_gain}")
        self.get_logger().info(f"th_neg_off: {self.th_neg_off}")
        self.get_logger().info(f"th_neg_dead: {self.th_neg_dead}")
        self.get_logger().info(f"sw_thr: {self.sw_thr}")
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_telem: Optional[XmaxxTelem] = None

        # IO
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None
        self.sub = self.create_subscription(
            XmaxxTelem, "/telem", self.telem_callback, rclpy.qos.qos_profile_sensor_data
        )
        
        self.get_logger().info(f"esc_to_odom node initialized")
    # ---------- helpers ----------
    def throttle_to_speed(self, raw: float) -> float:
        """Shared linear map: v = gain * raw + offset."""
        # check if the raw value is in deadband
        v = 0.0
        if (raw < self.th_pos_dead) and (raw > self.th_neg_dead): # val > 1415
            v = 0.0
        else:
            if raw > self.th_pos_dead:
                v = self.th_pos_gain * raw + self.th_pos_off
            else:
                v = self.th_neg_gain * raw + self.th_neg_off
        return v

    def choose_throttle_raw(self, msg: XmaxxTelem) -> float:
        """If rc_switch_a > threshold -> use AC throttle; else RC throttle."""
        return float(msg.ac_throttle if float(msg.rc_switch_a) > self.sw_thr else msg.rc_throttle)

    def choose_steering_raw(self, msg: XmaxxTelem) -> float:
        """If rc_switch_a > threshold -> use AC steering; else RC steering."""
        return float(msg.ac_steering if float(msg.rc_switch_a) > self.sw_thr else msg.rc_steering)

    def raw_to_steering_angle(self, raw_val: float) -> float:
        """Raw steering → wheel angle (rad) with optional raw deadband around straight."""

        # clamp the raw value to the max and min values
        if raw_val > self.steer_max_right:
            raw_val = self.steer_max_right
        elif raw_val < self.steer_max_left:
            raw_val = self.steer_max_left

        # calculate the angle
        # prevent division by zero
        if raw_val > self.steer_straight:
            angle = (raw_val - self.steer_straight) * self.steer_max_angle / (self.steer_max_right - self.steer_straight)
        else:
            angle = ((raw_val - self.steer_max_left) * self.steer_max_angle / (self.steer_straight - self.steer_max_left)) - self.steer_max_angle

        return angle

    # ---------- callback ----------
    def telem_callback(self, msg: XmaxxTelem):
        if self.last_telem is None:
            self.last_telem = msg
            return

        t_now = Time.from_msg(msg.header.stamp)
        t_prev = Time.from_msg(self.last_telem.header.stamp)
        dt = (t_now - t_prev).nanoseconds * 1e-9
        if dt <= 0.0:
            self.last_telem = msg
            return

        # Linear speed from selected throttle
        # between rc_throttle and ac_throttle
        raw_throttle = self.choose_throttle_raw(msg)
        v = self.throttle_to_speed(raw_throttle)
        self.get_logger().info(f"v: {v}")
        # Steering angle from feedback (use ac_steering)
        raw_steering = self.choose_steering_raw(msg)
        steer_angle = self.raw_to_steering_angle(raw_steering)

        # # Angular velocity (bicycle model)
        omega = 0.0
        if abs(self.wheelbase) > 1e-12:
            omega = v * math.tan(steer_angle) / self.wheelbase
        # invert the omega
        omega = -omega
        self.get_logger().info(f"omega: {omega}")

        # Integrate pose
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += omega * dt
        # self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))  # normalize

        # Odometry msg
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        half = 0.5 * self.yaw
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)

        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = 0.2
        odom.pose.covariance[7] = 0.2
        odom.pose.covariance[35] = 0.4

        odom.twist.twist.linear.x = v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = omega

        self.odom_pub.publish(odom)

        if self.publish_tf and self.tf_pub is not None:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf_pub.sendTransform(tf)

        self.last_telem = msg


def main(args=None):
    rclpy.init(args=args)
    node = XmaxxToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
