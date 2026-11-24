from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    total_agents_arg = LaunchConfiguration('total_agents', default='5')
    use_ai_arg = LaunchConfiguration('use_ai', default='true')
    use_virtual_spring_arg = LaunchConfiguration('use_virtual_spring', default='false')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )

    declare_total_agents = DeclareLaunchArgument(
        'total_agents',
        default_value='5',
        description='Total number of decentralized agents'
    )

    declare_use_ai = DeclareLaunchArgument(
        'use_ai',
        default_value='true',
        description='Gunakan Otak AI IDAPBC_Actor jika true, atau rumus pegas jika false'
    )

    declare_use_virtual_spring = DeclareLaunchArgument(
        'use_virtual_spring',
        default_value='true',
        description='Gunakan rumus pegas virtual jika true (otomatis aktif saat use_ai:=false)'
    )

    total_agents = 5
    agent_nodes = []
    for i in range(total_agents):
        node = Node(
            package='Simulasi-1',
            executable='decentralized_agent_node',
            name=f'decentralized_agent_{i}',
            output='screen',
            parameters=[{
                'agent_id': i,
                'total_agents': total_agents,
                'control_rate_hz': 50.0,
                'use_sim_time': use_sim_time,
                'use_ai': use_ai_arg,
                'use_virtual_spring': use_virtual_spring_arg,
            }]
        )
        agent_nodes.append(node)

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_total_agents)
    ld.add_action(declare_use_ai)
    ld.add_action(declare_use_virtual_spring)
    for node in agent_nodes:
        ld.add_action(node)

    return ld

