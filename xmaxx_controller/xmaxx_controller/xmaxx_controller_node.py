import threading
import sys, os
import serial
import math
import numpy as np
import struct
from struct import Struct

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from phx_nav_msgs.msg import OptimizedBicycleSequence
from phx_nav_msgs.msg import BicycleKinematicsStamped
from phx_nav_msgs.msg import BicycleKinematics
from nav_msgs.msg import Odometry
from xmaxx_msgs.msg import XmaxxTelem

#-------------------------------------------
"""
// Defines
//----------------------------------------------------
// APIs
#define AUTONOMY_API_MAGIC         (0x5A5A5A5A)
#define AUTONOMY_API_VERSION       (0x01)
#define AUTONOMY_CMD_TYPE_DRIVE    (0x01)
#define AUTONOMY_DATA_TYPE_TELEM   (0x81)

// Types
//----------------------------------------------------
typedef struct __attribute__((packed)) {
    int16_t     throttle;       // 1000-1500-2000
    int16_t     steering;       // 1000-1500-2000
} AutonomyDriveCmd_t;

typedef struct __attribute__((packed)) {
    uint32_t    magic;
    uint8_t     version;
    uint8_t     type;
    uint8_t     crc;
    union {
        AutonomyDriveCmd_t drive;
    } msg;
} AutonomyCmd_t;

typedef struct __attribute__((packed)) {
    uint64_t    counter;
    uint8_t     state;
    uint16_t    rcThrottle;     // 1000-1500-2000
    uint16_t    rcSteering;
    uint16_t    rcSwitchA;
    uint16_t    rcSwitchB;
    uint16_t    rcSwitchC;
    uint16_t    rcSwitchD;
    uint16_t    acThrottle;
    uint16_t    acSteering;
    uint8_t     upRssi;
    uint8_t     upLqi;
    uint8_t     downRssi;
    uint8_t     downLqi;
    uint16_t    escVoltageRaw;
    uint16_t    escCurrentRaw;
    uint16_t    escRpmRaw;
    uint16_t    escTempRaw;
} AutonomyTelemetryData_t;


// note - these are close enough that we won't
// bother with variable length structs
typedef struct __attribute__((packed)) {
    uint32_t    magic;
    uint8_t     version;
    uint8_t     type;
    uint8_t     crc;
    union {
        AutonomyTelemetryData_t telem;
    } msg;
} AutonomyData_t;
"""

#-------------------------------------------
MsgHeader = Struct('<LBBB')
MsgTelem  = Struct('<QBHHHHHHHHBBBBHHHH')

MsgDrive  = Struct('<LBBBHH')

#-------------------------------------------
CONTROL_DT_SECS = 0.1
CONTROL_MAX_STEERING_RADS = 0.454
#CONTROL_WHEELBASE_LEN_M = 0.50
CONTROL_FRONTBACK_LEN_M = 0.508

CONTROL_RC_CENTER_PT = 1500.0
CONTROL_RC_RANGE = 500.0
CONTROL_RC_MIN = 1000.0
CONTROL_RC_DEAD_ZONE = 50.0

M_PI = 3.14159265358979323846264338327950288

#-----------------------------
def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

#-----------------------------
class XmaxxController(Node):
    def __init__(self):
        super().__init__('xmaxx_controller')

        # control seq and integrated odom
        self.control_mutex = threading.Lock()
        self.controls = []

        self.odom = Odometry()
        
        # set fixed covar
        self.odom.pose.covariance = [1.00,0.00,0.00,0.00,0.00,0.00,
                                     0.00,1.00,0.00,0.00,0.00,0.00,
                                     0.00,0.00,1.00,0.00,0.00,0.00,
                                     0.00,0.00,0.00,1.00,0.00,0.00,
                                     0.00,0.00,0.00,0.00,1.00,0.00,
                                     0.00,0.00,0.00,0.00,0.00,1.00]
        self.odom.twist.covariance = [0.10,0.00,0.00,0.00,0.00,0.00,
                                     0.00,0.10,0.00,0.00,0.00,0.00,
                                     0.00,0.00,0.10,0.00,0.00,0.00,
                                     0.00,0.00,0.00,0.10,0.00,0.00,
                                     0.00,0.00,0.00,0.00,0.10,0.00,
                                     0.00,0.00,0.00,0.00,0.00,0.10]

        self.odom_theta = 0.0
        self.last_odom_update = self.get_clock().now()

        # setup publishers
        self.odom_pub = self.create_publisher(Odometry, 'platform/odom', 10)
        self.telem_pub = self.create_publisher(XmaxxTelem, 'platform/telem', 10)

        # setup subscribers
        self.seq_sub = self.create_subscription(
            OptimizedBicycleSequence,
            'optimized_bicycle_sequence',
            self.seq_callback,
            10)
        self.seq_sub  # prevent unused variable warning

        # setup timer
        timer_period = CONTROL_DT_SECS
        self.timer = self.create_timer(timer_period, self.send_rc_cmd)

        # setup serial connection to base
        self.ser = self.setupSerial()

        # start read thread
        if self.ser is not None:
            self.active = True
            self.rt = threading.Thread(target=self.readThread, args=())
            self.rt.start()    

    #-------------------------------------------
    def seq_callback(self, msg):
        # get local copy of control seq
        with self.control_mutex:
            self.controls = msg.controls

    #-------------------------------------------
    def send_rc_cmd(self):
        # this is called at action freq by a timer callback
        with self.control_mutex:
            if len(self.controls) > 0:
                ctrl = self.controls.pop(0).input
            else:
                ctrl = BicycleKinematics()

        #self.get_logger().info(f"{ctrl}")

        # map desired vel to throttle
        # from tachometer datapoints - best cubic fit
        x = abs(ctrl.velocity)
        cubic_est = 2.27772*pow(x,3) - 26.3869*pow(x,2) + 116.178*x - 3.87089

        # give min to get out of rc dead zone
        throttle_delta = int(max(CONTROL_RC_DEAD_ZONE+50, cubic_est))

        # NOTE - if the platform is configured to have a slower max speed in reverse,
        # then this will be slower than desired.  for example, if we are half-speed in rev,
        # then we should multiply this by 2.
        if ctrl.velocity == 0:
            throttle = int(CONTROL_RC_CENTER_PT)
        elif ctrl.velocity > 0.0:
            throttle = int(throttle_delta + CONTROL_RC_CENTER_PT)
        else:
            throttle = int((-throttle_delta*2) + CONTROL_RC_CENTER_PT)

        # set rc steer from angle (cap to range)
        clipped_steer_rads = min(CONTROL_MAX_STEERING_RADS,max(-CONTROL_MAX_STEERING_RADS,-ctrl.steering))
        steering = int(CONTROL_RC_CENTER_PT + clipped_steer_rads/CONTROL_MAX_STEERING_RADS*CONTROL_RC_RANGE)

        #self.get_logger().info(f"th:{throttle} ctrl.steering:{ctrl.steering} st:{steering}")

        # send drive message
        msg = MsgDrive.pack(0x5A5A5A5A, 0x01, 0x01, 0x0, throttle, steering)
        calculated_crc = self.crc8(msg)
        msg = MsgDrive.pack(0x5A5A5A5A, 0x01, 0x01, calculated_crc, throttle, steering)
        self.ser.write(msg)
        self.ser.flush()

    #-------------------------------------------
    def integrate_odom(self, telemMsg):
        # (1144309, 3, 1489, 1483, 988, 988, 2000, 1500, 0, 0, 26, 100, 0, 0, 1592, 472, 0, 0)
        mode = telemMsg[1]
        rcThrottle = telemMsg[2]
        acThrottle = telemMsg[8]
        rcSteering = telemMsg[3]
        acSteering = telemMsg[9]
        rpmRaw = telemMsg[16]

        # if we're in autonomy, use the autonomy values
        if mode == 5:
            throttle = acThrottle
            steering = acSteering    
        else:
            throttle = rcThrottle
            steering = rcSteering        

        #self.get_logger().info(f"rpmRaw:{rpmRaw} throttle:{throttle} steering:{steering}")

        # OLD METHOD
        # convert rpm to vel in ms
        #rpms = rpmRaw / 2042.0 * 20416.66
        #rps = rpms / 60.0
        #vel_ms = (rps / 31.3046 * 2.0 * 3.1415 * 0.1016)
        
        # NEW METHOD
        # linear fit from tachometer data
        axle_rpm = (0.20416 * rpmRaw) - 0.404002
        vel_ms = 0.147 * 0.103 * axle_rpm * 0.80    # 0.8 slip fudge factor

        # filter bad read - real reads start ~245
        # we seem to get spurious reads over 3k which are bad
        if rpmRaw < 10 or rpmRaw > 3000:
            vel_ms = 0.0

        # assume if we're negative that we're going backwards
        # active braking will mess with this...
        # also if we were negative last iteration, keep it so
        # we cover the spin down after a reverse
        if (throttle-CONTROL_RC_CENTER_PT) < -CONTROL_RC_DEAD_ZONE:
            vel_ms *= -1.0
        elif (self.odom.twist.twist.linear.x < 0):
            vel_ms *= -1.0

        #self.get_logger().info(f"throttle:{throttle} vel_ms:{vel_ms}")

        # get integration time
        now = self.get_clock().now()
        dT = now - self.last_odom_update
        dT = dT.nanoseconds * 1e-9
        #dT = CONTROL_DT_SECS

        # convert steering to angle
        phi = -(steering-CONTROL_RC_CENTER_PT)/CONTROL_RC_RANGE * CONTROL_MAX_STEERING_RADS

        # estimate changes
        vel_ms_f = vel_ms * math.cos(phi)
        vel_ms_w = vel_ms * math.sin(phi)/CONTROL_FRONTBACK_LEN_M

        x_dot = vel_ms_f * dT * math.cos(self.odom_theta)
        y_dot = vel_ms_f * dT * math.sin(self.odom_theta)

        theta_dot = vel_ms_w * dT

        # filter dead zone
        if abs(steering-CONTROL_RC_CENTER_PT) < CONTROL_RC_DEAD_ZONE:
            theta_dot = 0.0

        # update theta
        new_theta = self.odom_theta + theta_dot
        if(new_theta < -M_PI):
            new_theta += 2.0*M_PI
        elif(new_theta > M_PI):
            new_theta -= 2.0*M_PI
        self.odom_theta = new_theta

        # integrate
        self.odom.pose.pose.position.x += x_dot
        self.odom.pose.pose.position.y += y_dot

        x, y, z, w = quaternion_from_euler(0, 0, self.odom_theta)
        self.odom.pose.pose.orientation.x = x
        self.odom.pose.pose.orientation.y = y
        self.odom.pose.pose.orientation.z = z
        self.odom.pose.pose.orientation.w = w

        self.odom.twist.twist.linear.x = vel_ms_f
        self.odom.twist.twist.angular.z = vel_ms_w

        self.last_odom_update = now

        # publish telem so we can bag it
        xm_telem = XmaxxTelem()
        xm_telem.header.stamp = self.get_clock().now().to_msg()

        # fill out
        xm_telem.counter = telemMsg[0]
        xm_telem.state = telemMsg[1]
        xm_telem.rc_throttle = telemMsg[2]
        xm_telem.rc_steering = telemMsg[3]
        xm_telem.rc_switch_a = telemMsg[4]
        xm_telem.rc_switch_b = telemMsg[5]
        xm_telem.rc_switch_c = telemMsg[6]
        xm_telem.rc_switch_d = telemMsg[7]
        xm_telem.ac_throttle = telemMsg[8]
        xm_telem.ac_steering = telemMsg[9]
        xm_telem.up_rssi = telemMsg[10]
        xm_telem.up_lqi = telemMsg[11]
        xm_telem.down_rssi = telemMsg[12]
        xm_telem.down_lqi = telemMsg[13]
        xm_telem.esc_voltage_raw = telemMsg[14]
        xm_telem.esc_current_raw = telemMsg[15]
        xm_telem.esc_rpm_raw = telemMsg[16]
        xm_telem.esc_temp_raw = telemMsg[17]

        self.telem_pub.publish(xm_telem)

    #-------------------------------------------
    def publish_odom(self):        
        self.odom.header.stamp = self.get_clock().now().to_msg()
        self.odom.header.frame_id = 'groot/odom'    # TODO make this configurable
        self.odom.child_frame_id = 'groot/base_link'

        #self.get_logger().info(f"odom: x:{self.odom.pose.pose.position.x}, y:{self.odom.pose.pose.position.y}, th:{self.odom.pose.pose.orientation.w} th:{self.odom_theta}")
        self.odom_pub.publish(self.odom)

    #-------------------------------------------
    def crc8(self, data):
        c = 0
        for i in range(len(data)):
            c ^= data[i]
            for j in range(8):
                if(c & 0x80):
                    c = ((c << 1) ^ (0x1e7))
                else:
                    c = (c << 1)
        return c

    #-------------------------------------------
    def join(self):
        self.active = False
        self.rt.join()
        self.ser.close()

    #-------------------------------------------
    def readThread(self):
        sync = False

        try:
            while self.active:
                header = bytearray()
                if sync:
                    header = self.ser.read(MsgHeader.size)
                else:
                    message = ''
                    while not sync:
                        b = self.ser.read(1)
                        if(b == b'Z'):
                            header.extend(b)
                            if(header == bytearray(b'ZZZZ')):
                                header.extend(self.ser.read(MsgHeader.size-4))
                                self.get_logger().info("[-] Got sync...")
                                sync = True
                        else:
                            try:
                                message += b.decode()
                            except:
                                pass
                            header = bytearray()

                # unpack header
                msg = MsgHeader.unpack(header)
                msg_magic = msg[0]
                msg_version = msg[1]
                msg_type = msg[2]
                msg_crc = msg[3]

                if msg_magic != 0x5A5A5A5A:
                    self.get_logger().warning("[!] Bad magic")
                    sync = False
                    continue

                if msg_version != 0x01:
                    self.get_logger().error("[!] Bad version")
                    sync = False
                    continue

                # read body
                # all messages are the size of the largest (union)
                body = bytearray()
                body.extend(self.ser.read(MsgTelem.size))

                # check CRC
                msg = MsgHeader.pack(msg_magic, msg_version, msg_type, 0)
                computed_crc = self.crc8(msg+body)

                if msg_crc != computed_crc:
                    self.get_logger().error("[!] Bad CRC")
                    sync = False
                    continue

                # telem
                if msg_type == 0x81:
                    telemMsg = MsgTelem.unpack(body)
                    #self.get_logger().info("[-] telem : {}".format(telemMsg))
                    self.integrate_odom(telemMsg)
                    self.publish_odom()                

        except Exception as e1:
            self.get_logger().error("[!] Error: " + str(e1))
    
    #-------------------------------------------
    def setupSerial(self):
        ser = serial.Serial()
        ser.port = "/dev/ttyTHS1"
        ser.baudrate = 460800
        ser.bytesize = serial.EIGHTBITS
        ser.parity = serial.PARITY_NONE
        ser.stopbits = serial.STOPBITS_ONE
        ser.timeout = None
        ser.xonxoff = False
        ser.rtscts = False
        ser.dsrdtr = False
        ser.writeTimeout = None

        try: 
            ser.open()
        except Exception as e:
            self.get_logger().error("error opening serial port: " + str(e))
            return None

        return ser        

#-----------------------------
def main(args=None):
    aborted = False

    rclpy.init(args=args)

    # create and start node
    xm_controller = XmaxxController()

    try:
        rclpy.spin(xm_controller)
    except Exception as e:
        aborted = True
        xm_controller.get_logger().error("excecption: " + str(e))
        print("exception: " + str(e))

    # cleanup
    xm_controller.join()
    xm_controller.destroy_node()

    if not aborted:
        rclpy.shutdown()


if __name__ == '__main__':
    main()