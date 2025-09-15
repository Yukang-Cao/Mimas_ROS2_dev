from setuptools import find_packages, setup

package_name = 'mocap_tf_broadcaster'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alphatruck',
    maintainer_email='mikasa.cyk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
    'console_scripts': [
        'broadcaster_node = mocap_tf_broadcaster.broadcaster_node:main',
    ],
  },
)
