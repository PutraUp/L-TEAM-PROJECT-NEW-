#!/usr/bin/env python3
"""
Node ROS 2: bicycle_kinematic_inversion_node
Implementasi: Invers Kinematika Sepeda (bicycle_kinematic_inversion).

Menerjemahkan "tarikan" arah gerak horizontal (Fx) dan vertikal (Fy) pada titik depan robot
menjadi perintah setir (delta) dan gas (v_target) untuk Gazebo:

1. Hitung kecepatan maju (v) dan kecepatan putar (theta_dot):
   v = Fx * cos(theta) + Fy * sin(theta)
   theta_dot = (-Fx * sin(theta) + Fy * cos(theta)) / d

2. Ubah kecepatan putar jadi sudut setir (delta):
   delta = atan2(theta_dot * L, v + eps)
   di mana L = wheelbase (0.5 m), eps = 1e-5 (mencegah division by zero)

3. Batasi sudut setir (clamping):
   Maksimal +-30 derajat (+-0.523 radian)

4. Kirim ke Gazebo:
   Twist(linear.x = v_target, angular.z = delta) -> dipublish ke /agent_i/cmd_vel
"""

import json
import math
from typing import Dict, List, Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist, PointStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    rclpy = None
    Node = object
    Twist = PointStamped = Vector3Stamped = Odometry = String = object
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = object
    HAS_ROS2 = False


def extract_yaw_from_quaternion(q) -> float:
    """Mengubah Quaternion (x, y, z, w) dari topik /odom menjadi 1 sudut biasa (theta / yaw) dalam radian."""
    if hasattr(q, 'x'):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def bicycle_kinematic_inversion(
    Fx: float,
    Fy: float,
    theta: float,
    d: float = 0.25,
    L: float = 0.5,
    max_steer: float = 0.523,
    eps: float = 1e-5
) -> Tuple[float, float, float, float]:
    """
    Fungsi bicycle_kinematic_inversion:
    Menerjemahkan tarikan virtual (Fx, Fy) pada titik depan robot menjadi
    perintah kecepatan maju (v_target) dan sudut setir (delta) untuk Gazebo.

    Parameters:
        Fx (float): Tarikan / gaya gerak arah horizontal
        Fy (float): Tarikan / gaya gerak arah vertikal
        theta (float): Arah hadap / yaw robot (radian)
        d (float): Jarak titik kontrol ke hidung robot (default: 0.25 meter)
        L (float): Jarak antar-poros roda depan-belakang / wheelbase (default: 0.5 meter)
        max_steer (float): Batas fisik sudut setir dalam radian (default: 0.523 rad / 30 deg)
        eps (float): Epsilon pembagi pencegah division by zero saat v=0 (default: 1e-5)

    Returns:
        Tuple[float, float, float, float]:
            (v_target, delta_clamped, theta_dot, delta_raw)
    """
    if d <= 0.0:
        d = 0.25

    # 1. Hitung kecepatan maju (v) dan kecepatan putar (theta_dot) via matriks invers
    v = Fx * math.cos(theta) + Fy * math.sin(theta)
    theta_dot = (-Fx * math.sin(theta) + Fy * math.cos(theta)) / d

    # 2. Ubah kecepatan putar jadi sudut setir (delta)
    # Gunakan rumus atan2 dari kecepatan putar dikali L, lalu dibagi kecepatan maju (+ eps)
    denom = v + (eps if v >= 0.0 else -eps)
    delta = math.atan2(theta_dot * L, denom)

    # 3. Batasi sudut setir (clamping) maksimal +-30 derajat (0.523 radian)
    delta_clamped = max(-max_steer, min(max_steer, delta))

    return float(v), float(delta_clamped), float(theta_dot), float(delta)


class BicycleKinematicInversionNode(Node):
    """
    Node ROS 2 terpisah untuk invers kinematika sepeda (bicycle model inversion).
    Menerima tarikan (Fx, Fy) dari topik /agent_i/virtual_force dan orientasi /agent_i/odom,
    lalu mengirimkan gas (linear.x = v) dan setir (angular.z = delta) ke /agent_i/cmd_vel.
    """

    def __init__(self):
        super().__init__('bicycle_kinematic_inversion_node')

        # 1. Deklarasi Parameter ROS 2
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('total_agents', 5)
        self.declare_parameter('d', 0.25)               # Jarak titik hidung (m)
        self.declare_parameter('L', 0.5)                # Wheelbase (m)
        self.declare_parameter('max_steer_rad', 0.523)  # Batas setir +-30 derajat
        self.declare_parameter('eps', 1e-5)             # Epsilon anti division-by-zero
        self.declare_parameter('all_agents', False)     # True jika 1 node melayani semua 5 robot
        self.declare_parameter('control_rate_hz', 50.0) # Detak loop kontrol

        # 2. Pembacaan Parameter
        self.agent_id: int = self.get_parameter('agent_id').value
        self.total_agents: int = self.get_parameter('total_agents').value
        self.d: float = self.get_parameter('d').value
        self.L: float = self.get_parameter('L').value
        self.max_steer: float = self.get_parameter('max_steer_rad').value
        self.eps: float = self.get_parameter('eps').value
        self.all_agents: bool = self.get_parameter('all_agents').value
        self.control_rate: float = self.get_parameter('control_rate_hz').value

        # 3. Konfigurasi QoS
        self.qos_realtime = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # 4. State Management
        # agent_id -> {"theta": float, "Fx": float, "Fy": float, "v": float, "delta": float}
        self.agent_data: Dict[int, Dict[str, float]] = {}
        target_agent_ids = list(range(self.total_agents)) if self.all_agents else [self.agent_id]

        self.cmd_vel_publishers: Dict[int, rclpy.publisher.Publisher] = {}
        self.state_publishers: Dict[int, rclpy.publisher.Publisher] = {}
        self.odom_subscribers: List[rclpy.subscription.Subscription] = []
        self.force_subscribers: List[rclpy.subscription.Subscription] = []

        for aid in target_agent_ids:
            self.agent_data[aid] = {
                "theta": 0.0,
                "Fx": 0.0,
                "Fy": 0.0,
                "v": 0.0,
                "delta": 0.0,
                "theta_dot": 0.0
            }

            # Publisher /agent_i/cmd_vel (Kirim gas & setir ke Gazebo)
            pub_cmd = self.create_publisher(
                Twist,
                f"/agent_{aid}/cmd_vel",
                qos_cmd_vel
            )
            self.cmd_vel_publishers[aid] = pub_cmd

            # Publisher /agent_i/kinematic_state (Monitoring JSON)
            pub_state = self.create_publisher(
                String,
                f"/agent_{aid}/kinematic_state",
                self.qos_realtime
            )
            self.state_publishers[aid] = pub_state

            # Subscriber /agent_i/odom (Membaca sudut hadap theta)
            sub_odom = self.create_subscription(
                Odometry,
                f"/agent_{aid}/odom",
                lambda msg, target_id=aid: self.odom_callback(msg, target_id),
                self.qos_realtime
            )
            self.odom_subscribers.append(sub_odom)

            # Subscriber /agent_i/virtual_force (Menerima input tarikan Fx, Fy dari PointStamped / Twist)
            sub_force = self.create_subscription(
                PointStamped,
                f"/agent_{aid}/virtual_force",
                lambda msg, target_id=aid: self.force_point_callback(msg, target_id),
                self.qos_realtime
            )
            self.force_subscribers.append(sub_force)

            # Tambahan subscriber input fleksibel menggunakan Twist (/agent_i/target_pull)
            sub_twist_pull = self.create_subscription(
                Twist,
                f"/agent_{aid}/target_pull",
                lambda msg, target_id=aid: self.force_twist_callback(msg, target_id),
                self.qos_realtime
            )
            self.force_subscribers.append(sub_twist_pull)

        # 5. Timer Loop Kontrol 50 Hz (0.02 detik)
        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.control_step)

        mode_str = f"SEMUA AGEN (0..{self.total_agents - 1})" if self.all_agents else f"AGENT {self.agent_id}"
        self.get_logger().info(
            f"=====================================================\n"
            f" [BicycleKinematicInversionNode AKTIF]\n"
            f"  - Mode Operasi       : {mode_str}\n"
            f"  - Parameter d (hidung): {self.d} meter\n"
            f"  - Parameter L (wheel): {self.L} meter\n"
            f"  - Max Steer (+/-)    : {self.max_steer:.3f} rad ({math.degrees(self.max_steer):.1f} deg)\n"
            f"  - Epsilon (anti-zero): {self.eps}\n"
            f"  - Input Tarikan      : /agent_i/virtual_force, /agent_i/target_pull\n"
            f"  - Output Gazebo      : /agent_i/cmd_vel (Twist)\n"
            f"  - Status Topic       : /agent_i/kinematic_state\n"
            f"====================================================="
        )

    def odom_callback(self, msg: Odometry, aid: int):
        """Membaca arah hadap theta dari orientasi odometri."""
        theta = extract_yaw_from_quaternion(msg.pose.pose.orientation)
        if aid in self.agent_data:
            self.agent_data[aid]["theta"] = theta

    def force_point_callback(self, msg: PointStamped, aid: int):
        """Menerima tarikan Fx, Fy dari pesan PointStamped."""
        if aid in self.agent_data:
            self.agent_data[aid]["Fx"] = float(msg.point.x)
            self.agent_data[aid]["Fy"] = float(msg.point.y)

    def force_twist_callback(self, msg: Twist, aid: int):
        """Menerima tarikan Fx, Fy dari pesan Twist (linear.x = Fx, linear.y = Fy)."""
        if aid in self.agent_data:
            self.agent_data[aid]["Fx"] = float(msg.linear.x)
            self.agent_data[aid]["Fy"] = float(msg.linear.y)

    def control_step(self):
        """
        Detak loop kontrol 50 Hz:
        Menghitung invers kinematika sepeda untuk tiap agen dan mengirimkan Twist ke Gazebo.
        """
        for aid, data in self.agent_data.items():
            Fx = data["Fx"]
            Fy = data["Fy"]
            theta = data["theta"]

            # Eksekusi fungsi bicycle_kinematic_inversion
            v_target, delta, theta_dot, raw_delta = bicycle_kinematic_inversion(
                Fx=Fx,
                Fy=Fy,
                theta=theta,
                d=self.d,
                L=self.L,
                max_steer=self.max_steer,
                eps=self.eps
            )

            data["v"] = v_target
            data["delta"] = delta
            data["theta_dot"] = theta_dot

            # Kirim ke Gazebo melalui Twist:
            # linear.x = gas (kecepatan maju), angular.z = sudut setir (delta)
            cmd = Twist()
            cmd.linear.x = float(v_target)
            cmd.angular.z = float(delta)

            if aid in self.cmd_vel_publishers:
                self.cmd_vel_publishers[aid].publish(cmd)

            # Publikasikan status kinematika
            state_dict = {
                "agent_id": aid,
                "Fx": round(Fx, 3),
                "Fy": round(Fy, 3),
                "theta_deg": round(math.degrees(theta), 2),
                "v_target": round(v_target, 3),
                "delta_deg": round(math.degrees(delta), 2),
                "delta_rad": round(delta, 3),
                "theta_dot": round(theta_dot, 3),
            }
            str_msg = String()
            str_msg.data = json.dumps(state_dict)
            if aid in self.state_publishers:
                self.state_publishers[aid].publish(str_msg)

        # Log periodik dengan throttle agar terminal tetap bersih
        if self.agent_id in self.agent_data:
            d_curr = self.agent_data[self.agent_id]
            self.get_logger().info(
                f"[Agent {self.agent_id} Bicycle Inversion] F=({d_curr['Fx']:.2f}, {d_curr['Fy']:.2f}) "
                f"-> gas (v)={d_curr['v']:.2f} m/s, setir (delta)={math.degrees(d_curr['delta']):.1f} deg",
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = BicycleKinematicInversionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
