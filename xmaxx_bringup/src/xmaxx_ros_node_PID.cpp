// xmaxx_ros_node.cpp
#include "xmaxx_bringup/xmaxx_ros_node.h"

// Standard C++ includes 
#include <algorithm>    // For std::clamp (C++17)
#include <cmath>        // For std::round
#include <string>
#include <functional>   // For std::bind, std::placeholders

// constructor implementation
XmaxxRosNode::XmaxxRosNode(const rclcpp::NodeOptions & options)
    : Node("xmaxx_interface_node", options), last_control_update_time_(this->now())
{
    RCLCPP_INFO(this->get_logger(), "Initializing Xmaxx ROS Node...");

    // use kDefaultSerialDev and kDefaultBaud defined in the external xmaxx.h
    this->declare_parameter<std::string>("serial_port", kDefaultSerialDev);
    this->declare_parameter<int>("baud_rate", kDefaultBaud);

    std::string serial_port = this->get_parameter("serial_port").as_string();
    int baud_rate = this->get_parameter("baud_rate").as_int();

    // publisher on a topic /telem; SensorDataQoS for high-frequency telemetry data
    telem_publisher_ = this->create_publisher<xmaxx_msgs::msg::XmaxxTelem>("/telem", rclcpp::SensorDataQoS());
    
    // subscriber for /cmd_vel commands
    cmd_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&XmaxxRosNode::cmdVelCallback, this, std::placeholders::_1)
    );

    vel_bf_subscriber_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/vrpn_mocap/titan_alphatruck/velocity_body_frame_x", 10,
        std::bind(&XmaxxRosNode::velBfCallback, this, std::placeholders::_1)
    );

    // instantiate hardware interface
    xmaxx_interface_ = std::make_unique<Xmaxx>(serial_port, baud_rate);

    // wire up the telemetry callback
    xmaxx_interface_->setTelemetryCallback(
        std::bind(&XmaxxRosNode::telemetryCallback, this, std::placeholders::_1)
    );

    // last, call start() to open the serial port and begin listening
    if (!xmaxx_interface_->start()) {
        RCLCPP_FATAL(this->get_logger(), "Failed to start Xmaxx interface. Check connection/permissions (see stdout/stderr for library details). Shutting down.");
        rclcpp::shutdown();
    } else {
        RCLCPP_INFO(this->get_logger(), "Xmaxx interface ready and running on %s @ %d baud.", serial_port.c_str(), baud_rate);
    }
    
    // start the PID control timer
    using namespace std::chrono_literals;
    auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / CONTROL_LOOP_HZ)
    );
    // control_timer_ = this->create_wall_timer(
    //     period, std::bind(&XmaxxRosNode::updatePidControl, this)
    // );
    control_timer_ = this->create_wall_timer(
        period, std::bind(&XmaxxRosNode::PidControl, this)
    );
    
    RCLCPP_INFO(this->get_logger(), "PID control loop started at %.1f Hz", CONTROL_LOOP_HZ);
}

XmaxxRosNode::~XmaxxRosNode()
{
    // make sure the interface is stopped cleanly when the node shuts down
    if (xmaxx_interface_) {
        RCLCPP_INFO(this->get_logger(), "Shutting down Xmaxx ROS Node...");
        xmaxx_interface_->stop();
    }
}

// NOTE: this function is called by the Xmaxx object's background thread.
void XmaxxRosNode::telemetryCallback(const Telemetry& telem)
{
    // create an instance of the xmaxx_msgs::msg::XmaxxTelem message
    auto msg = xmaxx_msgs::msg::XmaxxTelem();
    msg.header.stamp = this->now();
    // msg.header.frame_id = "base_link"; // Optional: set appropriate frame_id

    // copy every field from the incoming const Telemetry& telem struct into the ROS2 message
    msg.counter = telem.counter;
    msg.state = telem.state;
    msg.rc_throttle = telem.rcThrottle;
    msg.rc_steering = telem.rcSteering;
    msg.rc_switch_a = telem.rcSwitchA;
    msg.rc_switch_b = telem.rcSwitchB;
    msg.rc_switch_c = telem.rcSwitchC;
    msg.rc_switch_d = telem.rcSwitchD;
    msg.ac_throttle = telem.acThrottle;
    msg.ac_steering = telem.acSteering;
    msg.up_rssi = telem.upRssi;
    msg.up_lqi = telem.upLqi;
    msg.down_rssi = telem.downRssi;
    msg.down_lqi = telem.downLqi;
    msg.esc_voltage_raw = telem.escVoltageRaw;
    msg.esc_current_raw = telem.escCurrentRaw;
    msg.esc_rpm_raw = telem.escRpmRaw;
    msg.esc_temp_raw = telem.escTempRaw;

    // Update current velocity estimates for PID control
    // Convert ESC RPM to linear velocity
    // TODO: [PID_TUNING] Calibrate conversion based on wheel circumference and gear ratio
    // constexpr double RPM_TO_LINEAR_VEL = 0.001; // Placeholder conversion factor
    // double estimated_linear_vel = static_cast<double>(telem.escRpmRaw) * RPM_TO_LINEAR_VEL;
    
    // Convert steering feedback to angular velocity using actual steering position
    // TODO: [PID_TUNING] Calibrate conversion based on steering geometry and vehicle dynamics
    // constexpr double STEERING_TO_ANGULAR_VEL = 0.002; // Placeholder conversion factor
    // double steering_normalized = (static_cast<double>(telem.acSteering) - RC_CENTER) / RC_RANGE;
    // double estimated_angular_vel = steering_normalized * STEERING_TO_ANGULAR_VEL;
    
    // // Update current velocity atomically (thread-safe)
    // current_linear_vel_.store(estimated_linear_vel);
    // current_angular_vel_.store(estimated_angular_vel);

    telem_publisher_->publish(msg); // publish the fully populated ROS2 message
}

void XmaxxRosNode::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    // Update target velocities atomically (thread-safe)
    target_linear_vel_.store(msg->linear.x);
    target_angular_vel_.store(msg->angular.z);

    // RCLCPP_INFO(this->get_logger(), "Received cmd_vel: linear=%.3f, angular=%.3f", 
    //              msg->linear.x, msg->angular.z);
    RCLCPP_DEBUG(this->get_logger(), "Received cmd_vel: linear=%.3f, angular=%.3f", 
                 msg->linear.x, msg->angular.z);
}

void XmaxxRosNode::velBfCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
       vel_bf_x_.store(msg->twist.linear.x);
       vel_bf_y_.store(msg->twist.linear.y);
       vel_bf_z_.store(msg->twist.linear.z);

       RCLCPP_DEBUG(this->get_logger(), "Received vel_bf: x=%.3f, y=%.3f, z=%.3f (stamp: %d.%09d)", 
                    msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z,
                    msg->header.stamp.sec, msg->header.stamp.nanosec);
}
// NOTE: this PID control is not implemented
// initial implementation not verified, kp ki kd not tuned(yet)
// USE FEEDFORWARD CONTROL INSTEAD
void XmaxxRosNode::PidControl()
{
    // Get current time and calculate dt
    auto current_time = this->now();
    double dt = (current_time - last_control_update_time_).seconds();
    last_control_update_time_ = current_time;

    // Get current setpoints and feedback (atomic read)
    double target_linear = target_linear_vel_.load();
    double target_angular = target_angular_vel_.load();
    double current_linear = vel_bf_x_.load();

    double ff_throttle_ = feedforwardVelToThrottleCmd(target_linear);
    double linear_error_vel_bf_x_   = - (current_linear - target_linear);
    std::cout << "curr bf x vel : " << current_linear << " tgt vel : " << target_linear << "error : " << linear_error_vel_bf_x_ << std::endl;
    double linear_errorDot_vel_bf_x = (linear_error_vel_bf_x_ - linear_error_vel_bf_x_prev_) / dt;
    linear_errorIntegral_vel_bf_x_  =  (linear_errorIntegral_gamma_ * linear_errorIntegral_vel_bf_x_) + linear_error_vel_bf_x_ * dt;

    double throttle = ff_throttle_ + KP_vel_bf_x * linear_error_vel_bf_x_ + KI_vel_bf_x * linear_errorIntegral_vel_bf_x_ + KD_vel_bf_x * linear_errorDot_vel_bf_x;
    uint16_t throttle_cmd = static_cast<uint16_t>(std::round(throttle));
    uint16_t steering_cmd = convertAngularVelToSteeringCmd(target_angular);

    
    // Send command to hardware (this is thread-safe as per Xmaxx design)
    if (xmaxx_interface_ && xmaxx_interface_->isRunning()) {
        xmaxx_interface_->sendDriveCmd(throttle_cmd, steering_cmd);
        // xmaxx_interface_->sendDriveCmd(1550, 1800);
    }

}


double XmaxxRosNode::feedforwardVelToThrottleCmd(double linear_vel)
{
    // Use the inverse of the piecewise model we derived.
    // Default to the neutral command for zero speed.
    double throttle_cmd = RC_CENTER; 

    if (linear_vel > 0.0) {
        // Invert the positive motion equation: Speed = 0.0352 * Input - 54.15
        throttle_cmd = (linear_vel + 54.15) / 0.0352;
    } 
    else if (linear_vel < 0.0) {
        // Handle negative saturation first
        if (linear_vel <= -2.5) {
            throttle_cmd = 1150.0;
        } else {
            // Invert the negative motion equation: Speed = 0.0092 * Input - 13.07
            throttle_cmd = (linear_vel + 13.07) / 0.0092;
        }
    }

    return throttle_cmd;
}

// A simple linear mapping for steering (can be improved with characterization)
uint16_t XmaxxRosNode::convertAngularVelToSteeringCmd(double angular_vel)
{
    /*
     * Converts a desired steering angle in radians to a corresponding integer
     * value for a vehicle's steering controller.
     *
     * This function performs a piecewise linear interpolation for left and right
     * turns, as the neutral steering value is not centered.
     *
     * Args:
     *     angle_rad: The desired steering angle in radians. A positive value
     *                indicates a right turn, and a negative value indicates a
     *                left turn.
     *
     * Returns:
     *     An integer value between 1000 and 2000, suitable for the steering
     *     controller.
     */
    // Vehicle Coordinate System Convention:
    // x-axis: forward
    // y-axis: right
    // z-axis: downward
    // A positive steering angle (+y rotation) corresponds to a right turn.

    // --- Constants based on controller specification ---
    constexpr double MAX_ANGLE_RAD = 0.4014;  // 23.7 degrees
    constexpr double STEERING_MAX_RIGHT = 1950;  // Corresponds to +23.7 deg
    constexpr double STEERING_MAX_LEFT = 1000;   // Corresponds to -23.7 deg
    constexpr double STEERING_STRAIGHT = 1482;   // Corresponds to 0 deg

    // --- Input Validation and Clamping ---
    // Ensure the input angle does not exceed the physical limits
    double clamped_angle = std::clamp(angular_vel, -MAX_ANGLE_RAD, MAX_ANGLE_RAD);

    // --- Piecewise Linear Interpolation ---
    double output;
    if (clamped_angle > 0) {
        // Interpolate for a right turn: maps the range [0, MAX_ANGLE_RAD]
        // to the output range [STEERING_STRAIGHT, STEERING_MAX_RIGHT].
        double slope = (STEERING_MAX_RIGHT - STEERING_STRAIGHT) / MAX_ANGLE_RAD;
        output = STEERING_STRAIGHT + clamped_angle * slope;
    } else if (clamped_angle < 0) {
        // Interpolate for a left turn: maps the range [-MAX_ANGLE_RAD, 0]
        // to the output range [STEERING_MAX_LEFT, STEERING_STRAIGHT].
        double slope = (STEERING_STRAIGHT - STEERING_MAX_LEFT) / MAX_ANGLE_RAD;
        output = STEERING_STRAIGHT + clamped_angle * slope; // angle is negative, so this is a subtraction
    } else { // angle is 0
        output = STEERING_STRAIGHT;
    }

    // Return the final controller value as a rounded integer
    return static_cast<uint16_t>(std::round(output));
}

// ============================================================================
// main function
// ============================================================================
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    try {
        auto node = std::make_shared<XmaxxRosNode>(options);

        // spin the node
        if (rclcpp::ok()) {
            rclcpp::spin(node);
        }

    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("XmaxxRosNodeMain"), "Unhandled exception: %s", e.what());
        rclcpp::shutdown();
        return 1;
    } catch (...) {
        RCLCPP_FATAL(rclcpp::get_logger("XmaxxRosNodeMain"), "Unknown unhandled exception");
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}