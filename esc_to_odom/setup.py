from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'esc_to_odom'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yukang',
    maintainer_email='mikasa.cyk@gmail.com',
    description='ROS2 package for converting ESC telemetry data to odometry for wheeled vehicles',
    license='MIT',
    entry_points={
        'console_scripts': [
            'esc_to_odom = esc_to_odom.esc_to_odom_node:main',
        ],
    },
)
