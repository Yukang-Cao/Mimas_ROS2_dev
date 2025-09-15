from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'xmaxx_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam P. Uccello',
    maintainer_email='adam@metastablelabs.com',
    description='XMAXX Controller',
    license='ARL',
    tests_require=[''],
    entry_points={
        'console_scripts': [
            'xmaxx_controller = xmaxx_controller.xmaxx_controller_node:main'
        ],
    },
)
