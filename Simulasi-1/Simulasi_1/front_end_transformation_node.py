#!/usr/bin/env python3
"""
Node ROS 2: FrontEndTransformationNode
Implementasi Teorema Brockett: Geser Titik Kontrol ke "Hidung" Robot (front_end_transformation).

Menggeser titik kontrol dari poros roda belakang (x, y) ke titik bayangan di depan
hidung robot (zx, zy) sejauh jarak d (default: 0.25 meter).

Rumus Trigonometri:
    zx = x + d * cos(theta)
    zy = y + d * sin(theta)

Dengan transformasi ini, robot diferensial (nonholonomic unicycle) dapat dikendalikan
seperti sistem linear 2D holonomic tanpa mengalami singularitas Brockett.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    rclpy = None
    Node = object
    Odometry = PointStamped = PoseStamped = String = object
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = object
    HAS_ROS2 = False


def extract_yaw_from_quaternion(q) -> float:
    """
    Mengubah orientasi Quaternion (x, y, z, w) dari topik /odom menjadi 1 sudut biasa (theta / yaw) dalam radian.
    Mendukung objek Quaternion ROS maupun tuple/list.
    """
    if hasattr(q, 'x'):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def front_end_transformation(x: float, y: float, theta: float, d: float = 0.25) -> Tuple[float, float]:
    """
    Geser Titik Kontrol ke "Hidung" Robot (front_end_transformation).
    
    zx = x + d * cos(theta)
    zy = y + d * sin(theta)

    Parameters:
        x (float): Posisi x dari poros roda / odom (meter)
        y (float): Posisi y dari poros roda / odom (meter)
        theta (float): Arah hadap / yaw robot (radian)
        d (float): Jarak pergeseran ke depan hidung robot (meter, default: 0.25)

    Returns:
        Tuple[float, float]: Koordinat titik kontrol depan (zx, zy)
    """
    zx = float(x + d * math.cos(theta))
    zy = float(y + d * math.sin(theta))
    return zx, zy


def inverse_front_end_velocities(vx_z: float, vy_z: float, theta: float, d: float = 0.25) -> Tuple[float, float]:
    """
    Invers Matriks Transformasi Hidung Robot:
    Mengubah kecepatan virtual titik kontrol depan [dot_zx, dot_zy] menjadi
    kecepatan unicycle [v (linear), omega (angular)]:

    [ v     ] = [  cos(theta)          sin(theta)      ] [ dot_zx ]
    [ omega ]   [ -sin(theta)/d        cos(theta)/d    ] [ dot_zy ]
    """
    if d <= 0.0:
        raise ValueError("Jarak d harus positif dan lebih besar dari 0.")
    v = vx_z * math.cos(theta) + vy_z * math.sin(theta)
    omega = (-vx_z * math.sin(theta) + vy_z * math.cos(theta)) / d
    return float(v), float(omega)


class FrontEndTransformationNode(Node):
    """
    Node ROS 2 untuk menghitung dan mempublikasikan titik kontrol depan (zx, zy) untuk setiap robot.
    Mendukung mode per-agen (agent_id: 0..4) maupun mode gabungan semua agen (all_agents: True).
    """

    def __init__(self):
        super().__init__('front_end_transformation_node')

        # 1. Deklarasi Parameter ROS 2
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('total_agents', 5)
        self.declare_parameter('d', 0.25)               # Jarak ke hidung robot (meter)
        self.declare_parameter('all_agents', False)      # True jika satu node mengurus semua robot
        self.declare_parameter('publish_rate_hz', 50.0)

        # 2. Pembacaan Parameter
        self.agent_id: int = self.get_parameter('agent_id').value
        self.total_agents: int = self.get_parameter('total_agents').value
        self.d: float = self.get_parameter('d').value
        self.all_agents: bool = self.get_parameter('all_agents').value
        self.publish_rate: float = self.get_parameter('publish_rate_hz').value

        # 3. QoS Profile Realtime (Live Streaming)
        self.qos_realtime = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # 4. Inisialisasi Publisher & Subscriber
        self.publishers_point: Dict[int, rclpy.publisher.Publisher] = {}
        self.publishers_pose: Dict[int, rclpy.publisher.Publisher] = {}
        self.publishers_state: Dict[int, rclpy.publisher.Publisher] = {}
        self.subscriptions_odom: List[rclpy.subscription.Subscription] = []

        # Data Cache
        # agent_id -> {"x": float, "y": float, "yaw": float, "zx": float, "zy": float}
        self.transformed_states: Dict[int, Dict[str, float]] = {}

        target_agent_ids = list(range(self.total_agents)) if self.all_agents else [self.agent_id]

        for aid in target_agent_ids:
            self.transformed_states[aid] = {
                "x": 0.0, "y": 0.0, "yaw": 0.0,
                "zx": 0.0, "zy": 0.0
            }

            # Publisher PointStamped (/agent_i/front_point)
            pub_point = self.create_publisher(
                PointStamped,
                f"/agent_{aid}/front_point",
                self.qos_realtime
            )
            self.publishers_point[aid] = pub_point

            # Publisher PoseStamped (/agent_i/front_pose) untuk visualisasi orientasi di RViz
            pub_pose = self.create_publisher(
                PoseStamped,
                f"/agent_{aid}/front_pose",
                self.qos_realtime
            )
            self.publishers_pose[aid] = pub_pose

            # Publisher JSON (/agent_i/front_state)
            pub_state = self.create_publisher(
                String,
                f"/agent_{aid}/front_state",
                self.qos_realtime
            )
            self.publishers_state[aid] = pub_state

            # Subscriber Odometry (/agent_i/odom)
            sub = self.create_subscription(
                Odometry,
                f"/agent_{aid}/odom",
                lambda msg, target_id=aid: self.odom_callback(msg, target_id),
                self.qos_realtime
            )
            self.subscriptions_odom.append(sub)

        mode_str = "SEMUA AGEN (0..{})".format(self.total_agents - 1) if self.all_agents else f"AGENT {self.agent_id}"
        self.get_logger().info(
            f"=====================================================\n"
            f" [FrontEndTransformationNode AKTIF]\n"
            f"  - Mode Operasi    : {mode_str}\n"
            f"  - Jarak Hidung (d): {self.d} meter\n"
            f"  - Topik Output    : /agent_i/front_point\n"
            f"                      /agent_i/front_pose\n"
            f"                      /agent_i/front_state\n"
            f"====================================================="
        )

    def odom_callback(self, msg: Odometry, aid: int):
        """Menerima odometri robot, menghitung titik kontrol hidung (zx, zy), lalu mempublikasikannya."""
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        theta = extract_yaw_from_quaternion(msg.pose.pose.orientation)

        # Hitung titik depan (front_end_transformation)
        # zx = x + d * cos(theta)
        # zy = y + d * sin(theta)
        zx, zy = front_end_transformation(x, y, theta, self.d)

        # Simpan state internal
        self.transformed_states[aid] = {
            "x": x, "y": y, "yaw": theta,
            "zx": zx, "zy": zy
        }

        # 1. Publikasikan PointStamped (/agent_i/front_point)
        pt_msg = PointStamped()
        pt_msg.header = msg.header
        pt_msg.header.frame_id = f"agent_{aid}/odom"
        pt_msg.point.x = zx
        pt_msg.point.y = zy
        pt_msg.point.z = float(msg.pose.pose.position.z)
        if aid in self.publishers_point:
            self.publishers_point[aid].publish(pt_msg)

        # 2. Publikasikan PoseStamped (/agent_i/front_pose)
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = f"agent_{aid}/odom"
        pose_msg.pose.position.x = zx
        pose_msg.pose.position.y = zy
        pose_msg.pose.position.z = float(msg.pose.pose.position.z)
        pose_msg.pose.orientation = msg.pose.pose.orientation
        if aid in self.publishers_pose:
            self.publishers_pose[aid].publish(pose_msg)

        # 3. Publikasikan JSON State (/agent_i/front_state)
        state_dict = {
            "agent_id": aid,
            "d": self.d,
            "odom_pose": {
                "x": round(x, 4),
                "y": round(y, 4),
                "yaw": round(theta, 4)
            },
            "front_point": {
                "zx": round(zx, 4),
                "zy": round(zy, 4)
            }
        }
        str_msg = String()
        str_msg.data = json.dumps(state_dict)
        if aid in self.publishers_state:
            self.publishers_state[aid].publish(str_msg)

        # Log periodik dengan throttle (setiap 1.0 detik)
        self.get_logger().info(
            f"[Agent {aid} Front-End] odom=({x:.2f}, {y:.2f}, theta={math.degrees(theta):.1f} deg) "
            f"-> front_point (zx, zy) = ({zx:.2f}, {zy:.2f}) [d={self.d}m]",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = FrontEndTransformationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
