from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'uge_mpc_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')), # Install launch files
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml'))), # Install parameter file

    ],
    install_requires=['setuptools', 'rclpy', 'numpy', 'torch', 'scipy', 'pyyaml', 'matplotlib', 'numba'],
    zip_safe=True,
    maintainer='alphatruck',
    maintainer_email='mikasa.cyk@gmail.com',
    description='UGE-MPC ROS2 package',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'subscriber_node = uge_mpc_ros2.subscriber_node:main',
            'uge_controller_node = uge_mpc_ros2.uge_controller_node:main',
        ],
    },
)
