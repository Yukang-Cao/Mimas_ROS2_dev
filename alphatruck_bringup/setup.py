# src/alphatruck_bringup/setup.py
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'alphatruck_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')), # Install launch files
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml'))), # Install parameter files
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))), # Install URDF files
        (os.path.join('share', package_name, 'resource'), ['resource/' + package_name]),
        (os.path.join('share', package_name, 'resource', 'map_conditioned_cuniform_models_v2.5'), 
         glob(os.path.join('resource', 'map_conditioned_cuniform_models_v2.5', '*'))),
        (os.path.join('share', package_name, 'resource', 'unsupervised_cuniform_model_v2.5'), 
         glob(os.path.join('resource', 'unsupervised_cuniform_model_v2.5', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alphatruck',
    maintainer_email='cao00125@umn.edu',
    description='Bringup package for Alphatruck',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
        ],
    },
)
