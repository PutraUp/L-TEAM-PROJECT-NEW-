#!/usr/bin/env python3
"""
Launch file: front_end.launch.py
Menjalankan FrontEndTransformationNode untuk ke-5 agen robot secara bersamaan.
Masing-masing instance menghitung dan mempublikasikan titik kontrol hidung (zx, zy)
ke topik /agent_{i}/front_point, /agent_{i}/front_pose, dan /agent_{i}/front_state.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    d_val = LaunchConfiguration('d', default='0.25')
    total_agents_val = 5

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Gunakan clock simulasi (Gazebo) jika true'
    )

    declare_d = DeclareLaunchArgument(
        'd',
        default_value='0.25',
        description='Jarak pergeseran ke hidung robot dalam meter (default: 0.25)'
    )

    # Buat 5 node FrontEndTransformationNode terdesentralisasi (agent_id 0..4)
    nodes = []
    for i in range(total_agents_val):
        node = Node(
            package='Simulasi-1',
            executable='front_end_transformation_node',
            name=f'front_end_agent_{i}',
            output='screen',
            parameters=[{
                'agent_id': i,
                'total_agents': total_agents_val,
                'd': d_val,
                'all_agents': False,
                'use_sim_time': use_sim_time
            }]
        )
        nodes.append(node)

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_d)
    for n in nodes:
        ld.add_action(n)

    return ld
