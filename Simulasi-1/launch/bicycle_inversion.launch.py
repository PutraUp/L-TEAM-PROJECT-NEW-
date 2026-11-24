#!/usr/bin/env python3
"""
Launch file: bicycle_inversion.launch.py
Menjalankan bicycle_kinematic_inversion_node untuk ke-5 agen robot secara bersamaan.
Masing-masing menerima tarikan (Fx, Fy) dan orientasi theta, lalu mengirimkan
gas (linear.x = v_target) dan setir (angular.z = delta) ke /agent_i/cmd_vel.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    d_val = LaunchConfiguration('d', default='0.25')
    L_val = LaunchConfiguration('L', default='0.5')
    max_steer_val = LaunchConfiguration('max_steer_rad', default='0.523')
    total_agents_val = 5

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Gunakan clock simulasi (Gazebo) jika true'
    )

    declare_d = DeclareLaunchArgument(
        'd',
        default_value='0.25',
        description='Jarak ke hidung robot dalam meter (default: 0.25)'
    )

    declare_L = DeclareLaunchArgument(
        'L',
        default_value='0.5',
        description='Wheelbase / jarak antar-poros roda dalam meter (default: 0.5)'
    )

    declare_max_steer = DeclareLaunchArgument(
        'max_steer_rad',
        default_value='0.523',
        description='Batas maksimal sudut setir dalam radian (default: 0.523 rad / 30 deg)'
    )

    nodes = []
    for i in range(total_agents_val):
        node = Node(
            package='Simulasi-1',
            executable='bicycle_kinematic_inversion_node',
            name=f'bicycle_inversion_agent_{i}',
            output='screen',
            parameters=[{
                'agent_id': i,
                'total_agents': total_agents_val,
                'd': d_val,
                'L': L_val,
                'max_steer_rad': max_steer_val,
                'all_agents': False,
                'control_rate_hz': 50.0,
                'use_sim_time': use_sim_time
            }]
        )
        nodes.append(node)

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_d)
    ld.add_action(declare_L)
    ld.add_action(declare_max_steer)
    for n in nodes:
        ld.add_action(n)

    return ld
