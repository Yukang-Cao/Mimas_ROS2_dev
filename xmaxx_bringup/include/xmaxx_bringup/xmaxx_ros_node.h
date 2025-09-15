// xmaxx_ros_node.h
#pragma once

// ROS2 includes
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <xmaxx_msgs/msg/xmaxx_telem.hpp>

// Standard C++ includes 
#include <memory>       // For std::unique_ptr
#include <cstdint>
#include <mutex>        // For thread safety
#include <atomic>       // For atomic variables

// include the Xmaxx library headers (External files in workspace root)
#include <xmaxx_bringup/xmaxx.h>

class XmaxxRosNode : public rclcpp::Node
{
public:
    // constructor declaration
    explicit XmaxxRosNode(const rclcpp::NodeOptions & options);
    ~XmaxxRosNode() override;

private:
    // callback declarations
    void telemetryCallback(const Telemetry& telem);
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void velBfCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    
    // control methods
    // void PidControl();
    void feedforwardControl();
    // double computePidOutput(double setpoint, double current_value, double& integral, double& previous_error, double dt);
    double feedforwardVelToThrottleCmd(double linear_vel);
    uint16_t convertLinearVelToThrottleCmd(double linear_vel);
    uint16_t convertAngularVelToSteeringCmd(double angular_vel);

    // member variable declarations
    rclcpp::Publisher<xmaxx_msgs::msg::XmaxxTelem>::SharedPtr telem_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_bf_subscriber_;

    // This now uses the externally defined Xmaxx class.
    std::unique_ptr<Xmaxx> xmaxx_interface_;

    // PID control state variables (protected by mutex)
    std::mutex control_mutex_;
    std::atomic<double> target_linear_vel_{0.0};
    std::atomic<double> target_angular_vel_{0.0};
    std::atomic<double> current_linear_vel_{0.0};   // derived from telemetry
    std::atomic<double> current_angular_vel_{0.0};  // derived from telemetry

    std::atomic<double> vel_bf_x_{0.0};
    std::atomic<double> vel_bf_y_{0.0};
    std::atomic<double> vel_bf_z_{0.0};

    
    // PID state variables (protected by mutex in updatePidControl)
    double linear_integral_{0.0};
    double linear_previous_error_{0.0};

    double linear_error_vel_bf_x_prev_{0.0};
    double linear_errorIntegral_vel_bf_x_{0.0};
    double linear_errorIntegral_gamma_{0.98};

    double angular_integral_{0.0};
    double angular_previous_error_{0.0};
    rclcpp::Time last_control_update_time_;
    
    // Timer for PID control loop
    rclcpp::TimerBase::SharedPtr control_timer_;

    // constants for RC signal conversion
    static constexpr double RC_CENTER = 1500.0;
    static constexpr double RC_RANGE = 500.0;
    static constexpr uint16_t RC_MIN_VAL = 1000;
    static constexpr uint16_t RC_MAX_VAL = 2000;
    
    // PID constants
    static constexpr double KP_vel_bf_x = 30.0;
    static constexpr double KI_vel_bf_x = 0.0;
    static constexpr double KD_vel_bf_x = 0.0;
    static constexpr double KP_steer = 0.0;
    static constexpr double KI_steer = 0.0;
    static constexpr double KD_steer = 0.0;
    static constexpr double CONTROL_LOOP_HZ = 50.0;  // 50Hz control loop
};