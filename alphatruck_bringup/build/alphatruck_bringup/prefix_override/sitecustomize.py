import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/alphatruck/ros2_ws/src/alphatruck_bringup/install/alphatruck_bringup'
