from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alphatruck',
    maintainer_email='alphatruck@todo.todo',
    description='Perception package for AlphaTruck with LiDAR processing and costmap generation',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'perception_node = perception.perception_node:main',
            'costmap_processor_node = perception.costmap_processor_node:main',
            'test_goal_publisher = perception.test_goal_publisher:main',
            'dummy_odom_publisher = perception.dummy_odom_publisher:main',
        ],
    },
)
