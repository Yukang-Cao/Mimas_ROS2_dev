from setuptools import find_packages, setup

package_name = 'controllers'

setup(
    name=package_name,
    version='1.0.0',
    # Automatically discover packages including 'controllers' and 'controllers.lib'
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy', 'torch', 'scipy', 'pyyaml', 'matplotlib', 'numba'],
    zip_safe=True,
    maintainer='alphatruck',
    maintainer_email='cao00125@umn.edu',
    description='High-performance PyTorch-based trajectory planners (MPPI, CU-MPPI)',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # Define the entry point for the ROS node executable
            'local_planner_node = controllers.local_planner_node:main',
        ],
    },
)
