import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'Simulasi-1'


def get_all_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ]

    for root, dirs, files in os.walk('models'):
        if files:
            dest_dir = os.path.join('share', package_name, root)
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((dest_dir, file_paths))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=get_all_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a.putra',
    maintainer_email='a.putra@todo.todo',
    description='Package for multi-robot simulation and Cerdas UWB Tracker',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'uwb_tracker_node = Simulasi_1.uwb_tracker_node:main',
            'decentralized_agent_node = Simulasi_1.DecentralizedAgentNode:main',
            'front_end_transformation_node = Simulasi_1.front_end_transformation_node:main',
            'bicycle_kinematic_inversion_node = Simulasi_1.bicycle_kinematic_inversion_node:main',
            'train_phmarl = Simulasi_1.train_phmarl:main',
        ],
    },
)
