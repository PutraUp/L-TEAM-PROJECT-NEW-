import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Cari path direktori package (baik setelah colcon build maupun langsung di src)
    try:
        pkg_share = get_package_share_directory('Simulasi-1')
        models_dir = os.path.join(pkg_share, 'models')
        world_file = os.path.join(pkg_share, 'worlds', 'default.sdf')
        if not os.path.exists(models_dir):
            raise FileNotFoundError()
    except (PackageNotFoundError, FileNotFoundError):
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(src_dir, 'models')
        world_file = os.path.join(src_dir, 'worlds', 'default.sdf')

    # Deteksi otomatis DISPLAY (jika tidak ada di env, gunakan virtual display VNC :1 jika ada)
    display_env = os.environ.get('DISPLAY')
    if not display_env and os.path.exists('/tmp/.X11-unix/X1'):
        display_env = ':1'
        os.environ['DISPLAY'] = ':1'

    has_display = bool(display_env or os.environ.get('WAYLAND_DISPLAY'))
    default_headless = 'false' if has_display else 'true'

    # Argumen launch
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default=default_headless)
    start_agents = LaunchConfiguration('start_agents', default='true')
    use_ai = LaunchConfiguration('use_ai', default='true')
    use_virtual_spring = LaunchConfiguration('use_virtual_spring', default='false')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value=default_headless,
        description='Run Gazebo Sim in headless mode (server only) without GUI'
    )

    declare_start_agents = DeclareLaunchArgument(
        'start_agents',
        default_value='true',
        description='Jalankan juga 5 node agen robot terdesentralisasi jika true (default: true)'
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

    # Environment variables untuk Gazebo Sim
    current_gz_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_env = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=f"{models_dir}:{current_gz_resource_path}" if current_gz_resource_path else models_dir
    )

    set_display_env = SetEnvironmentVariable(
        name='DISPLAY',
        value=display_env if display_env else ':1'
    )

    # Launch Gazebo Sim: dengan GUI
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ),
        launch_arguments={'gz_args': f"-r {world_file}"}.items(),
        condition=UnlessCondition(headless)
    )

    # Launch Gazebo Sim: tanpa GUI (headless mode: -s)
    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ),
        launch_arguments={'gz_args': f"-s -r {world_file}"}.items(),
        condition=IfCondition(headless)
    )

    # Formasi Belah Ketupat (Rhombus/Diamond):
    #
    #                 x1_robot-1 (x=3.0, y=0.0) [Depan / Apex]
    #                             /\
    #                            /  \
    #    x1_robot-2 (0.0, 2.5)   \  /   x1_robot-4 (0.0, -2.5) [Kanan]
    #     [Kiri]             x1_robot-3 (0.0, 0.0) [Pusat]
    #                            \  /
    #                             \/
    #                 x1_robot-5 (x=-3.0, y=0.0) [Belakang]
    #
    robots = [
        {'name': 'x1_robot-1', 'x': '3.0',  'y': '0.0',  'z': '0.2', 'Y': '0.0', 'id': '0'},  # Apex / Depan (Agent 0)
        {'name': 'x1_robot-2', 'x': '0.0',  'y': '2.5',  'z': '0.2', 'Y': '0.0', 'id': '1'},  # Kiri (Agent 1)
        {'name': 'x1_robot-3', 'x': '0.0',  'y': '0.0',  'z': '0.2', 'Y': '0.0', 'id': '2'},  # Tengah (Agent 2)
        {'name': 'x1_robot-4', 'x': '0.0',  'y': '-2.5', 'z': '0.2', 'Y': '0.0', 'id': '3'},  # Kanan (Agent 3)
        {'name': 'x1_robot-5', 'x': '-3.0', 'y': '0.0',  'z': '0.2', 'Y': '0.0', 'id': '4'},  # Belakang (Agent 4)
    ]

    spawn_robot_nodes = []
    bridge_arguments = [
        # Bridge /clock dari Gazebo ke ROS 2
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # Bridge dynamic_pose dari Gazebo ke ROS 2 untuk sistem tracking Cerdas UWB
        '/world/default/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
    ]

    for bot in robots:
        sdf_path = os.path.join(models_dir, bot['name'], 'model.sdf')
        spawn_node = Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=[
                '-world', 'default',
                '-file', sdf_path,
                '-name', bot['name'],
                '-x', bot['x'],
                '-y', bot['y'],
                '-z', bot['z'],
                '-Y', bot['Y'],
            ],
            parameters=[{'use_sim_time': use_sim_time}]
        )
        spawn_robot_nodes.append(spawn_node)

        # Bridge topik LiDAR dari Gazebo Sim ke ROS 2 (LaserScan 2D & PointCloud 3D)
        ros_name = bot['name'].replace('-', '_')
        bridge_arguments.append(
            f"/{ros_name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
        )
        bridge_arguments.append(
            f"/{ros_name}/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
        )

        # Bridge Differential Drive: cmd_vel (ROS 2 -> Gazebo) dan odom (Gazebo -> ROS 2)
        agent_id = bot['id']
        bridge_arguments.append(
            f"/agent_{agent_id}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
        )
        bridge_arguments.append(
            f"/agent_{agent_id}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry"
        )

    # Parameter bridge untuk Gazebo Transport <-> ROS 2
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        output='screen',
        arguments=bridge_arguments,
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Node Sensor Cerdas UWB Tracker (Trilaterasi, Range, JSON format asli)
    uwb_tracker_node = Node(
        package='Simulasi-1',
        executable='uwb_tracker_node',
        name='cerdas_uwb_tracker_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 5 Node Agen Terdesentralisasi: dipusatkan melalui pemanggilan agents.launch.py
    agents_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('Simulasi-1'),
                'launch',
                'agents.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'total_agents': '5',
            'use_ai': use_ai,
            'use_virtual_spring': use_virtual_spring,
        }.items(),
        condition=IfCondition(start_agents)
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_headless)
    ld.add_action(declare_start_agents)
    ld.add_action(declare_use_ai)
    ld.add_action(declare_use_virtual_spring)
    ld.add_action(gz_resource_env)
    ld.add_action(set_display_env)
    ld.add_action(gz_sim_gui)
    ld.add_action(gz_sim_server)
    for node in spawn_robot_nodes:
        ld.add_action(node)
    ld.add_action(ros_gz_bridge)
    ld.add_action(uwb_tracker_node)
    ld.add_action(agents_launch)

    return ld
