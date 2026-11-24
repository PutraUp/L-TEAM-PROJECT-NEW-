#!/usr/bin/env python3
"""
Node ROS 2: Cerdas UWB Tracker - Swarm Dual Mode (Anchor + Tag)
Emulasi penuh hardware & firmware Cerdas-UWB-Tracker (Decawave DWM1000):
- Mode Dual: Setiap robot bertindak sekaligus sebagai Tag dan Mobile Anchor
- Inter-Robot Ranging (TWR) antar 5 robot kawanan
- Noise Gaussian DWM1000 (std dev = 2.5 cm)
- Filter moving average 3 sampel (sesuai firmware fresh_link())
- Format payload JSON persis repositori https://github.com/geraicerdas/Cerdas-UWB-Tracker
- Penghitungan posisi relatif terhadap pusat formasi (Robot 3)
- Publikasi topik ROS 2 per-robot & matriks jarak kawanan
"""

import json
import math
import random
import threading
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Range

from gz.msgs.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode


def quat_rotate(q: Tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotasi vektor 3D dengan kuaternion (qx, qy, qz, qw)."""
    qx, qy, qz, qw = q
    # Formula Rodrigues / Quat multiplication
    # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + qw * v)
    q_vec = np.array([qx, qy, qz])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (qw * uv + uuv)
    

class CerdasUWBSwarmNode(Node):
    """
    Node ROS 2 untuk emulasi Cerdas UWB Tracker Swarm Inter-Robot Ranging.
    Setiap robot memiliki ID Anchor virtual dan mengukur jarak ke robot lainnya.
    """

    def __init__(self):
        super().__init__('cerdas_uwb_tracker_node')

        # Definisi 5 Robot & Alokasi ID UWB Anchor
        # Formasi Belah Ketupat:
        #             Robot 1 (Apex / Depan) [ID 1001]
        # Robot 2 (Kiri) [1002]  Robot 3 (Pusat) [1003]  Robot 4 (Kanan) [1004]
        #             Robot 5 (Belakang)     [1005]
        self.robots = [
            {"gz_name": "x1_robot-1", "ros_name": "x1_robot_1", "anchor_id": "1001", "role": "Apex/Depan"},
            {"gz_name": "x1_robot-2", "ros_name": "x1_robot_2", "anchor_id": "1002", "role": "Kiri"},
            {"gz_name": "x1_robot-3", "ros_name": "x1_robot_3", "anchor_id": "1003", "role": "Pusat Acuan"},
            {"gz_name": "x1_robot-4", "ros_name": "x1_robot_4", "anchor_id": "1004", "role": "Kanan"},
            {"gz_name": "x1_robot-5", "ros_name": "x1_robot_5", "anchor_id": "1005", "role": "Belakang"},
        ]

        # Offset fisik modul UWB Tag pada top-plate robot (meter)
        self.tag_offset_body = np.array([-0.06, 0.0, 0.30])

        # Riwayat filter moving average 3 sampel: [raw, prev1, prev2]
        # self.range_history[src_id][dst_id] = [r0, r1, r2]
        self.range_history: Dict[str, Dict[str, List[float]]] = {}
        for r_src in self.robots:
            src_id = r_src["anchor_id"]
            self.range_history[src_id] = {}
            for r_dst in self.robots:
                if r_src["anchor_id"] != r_dst["anchor_id"]:
                    self.range_history[src_id][r_dst["anchor_id"]] = [0.0, 0.0, 0.0]

        # Cache pose terakhir dari Gazebo (thread-safe)
        self.pose_lock = threading.Lock()
        self.latest_poses = {}

        # Setup ROS 2 Publishers dengan QoS Real-Time (Live Streaming agar tidak ngelag):
        # - BEST_EFFORT: Tidak kirim ulang data lama
        # - KEEP_LAST depth=1: Hanya simpan 1 data posisi terbaru
        # - VOLATILE: Tidak menyimpan riwayat masa lalu
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pubs_dict = {}
        for idx, r in enumerate(self.robots, start=1):
            r_name = r["ros_name"]
            gz_name = r["gz_name"]
            a_name = f"agent_{idx}"
            self.pubs_dict[gz_name] = {
                "json": [
                    self.create_publisher(String, f"/{r_name}/uwb/links_json", qos),
                    self.create_publisher(String, f"/{a_name}/uwb/links_json", qos),
                ],
                "rel_pos": [
                    self.create_publisher(PointStamped, f"/{r_name}/uwb/relative_position", qos),
                    self.create_publisher(PointStamped, f"/{a_name}/uwb/relative_position", qos),
                ],
                "rel_pose": [
                    self.create_publisher(PoseWithCovarianceStamped, f"/{r_name}/uwb/relative_pose", qos),
                    self.create_publisher(PoseWithCovarianceStamped, f"/{a_name}/uwb/relative_pose", qos),
                ],
                "ranges": {}
            }
            # Publisher sensor_msgs/Range untuk setiap pasangan robot
            for jdx, other in enumerate(self.robots, start=1):
                if other["gz_name"] != gz_name:
                    o_name = other["ros_name"]
                    oa_name = f"agent_{jdx}"
                    self.pubs_dict[gz_name]["ranges"][o_name] = [
                        self.create_publisher(Range, f"/{r_name}/uwb/range_to_{o_name}", qos),
                        self.create_publisher(Range, f"/{a_name}/uwb/range_to_{oa_name}", qos),
                    ]

        # Matriks jarak global swarm (seluruh pasangan robot)
        self.swarm_matrix_pub = self.create_publisher(String, "/swarm/distance_matrix", qos)

        # Subscribe ke Gazebo Sim pose topic via C++ Gazebo Transport
        self.gz_node = GzNode()
        self.gz_node.subscribe(Pose_V, "/world/default/pose/info", self.gz_pose_callback)

        # Timer 10 Hz (100 ms) sesuai rate update firmware Cerdas UWB Tracker
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            "=====================================================\n"
            "  Cerdas UWB Tracker: Swarm Dual Mode (Anchor + Tag)\n"
            "  - Mode: 100% Infrastructure-Free (Tanpa Tiang Statis)\n"
            "  - Inter-Robot Ranging: Antar 5 Robot Kawanan\n"
            "  - Alokasi ID Anchor: "
            + ", ".join([f"{r['ros_name']}: {r['anchor_id']}" for r in self.robots]) + "\n"
            "  - Acuan Pusat Formasi: x1_robot_3 (ID 1003)\n"
            "  - Update Rate: 10 Hz (100 ms)\n"
            "====================================================="
        )

    def gz_pose_callback(self, msg: Pose_V):
        """Menerima posisi robot dari Gazebo worker thread dan menyimpan ke cache."""
        with self.pose_lock:
            for p in msg.pose:
                for r in self.robots:
                    if p.name == r["gz_name"]:
                        self.latest_poses[p.name] = (
                            p.position.x, p.position.y, p.position.z,
                            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
                        )

    def timer_callback(self):
        """Loop kalkulasi jarak UWB TWR dan publikasi 10 Hz di main thread."""
        with self.pose_lock:
            current_poses = dict(self.latest_poses)

        # Butuh minimal 2 robot terdeteksi untuk ranging
        if len(current_poses) < 2:
            return

        now = self.get_clock().now().to_msg()

        # Hitung posisi absolut Tag UWB untuk setiap robot yang ada di cache
        tag_world_positions = {}
        for r in self.robots:
            gz_n = r["gz_name"]
            if gz_n in current_poses:
                px, py, pz, qx, qy, qz, qw = current_poses[gz_n]
                # Rotasi tag offset sesuai orientasi robot
                rotated_offset = quat_rotate((qx, qy, qz, qw), self.tag_offset_body)
                tag_world_positions[gz_n] = np.array([px, py, pz]) + rotated_offset

        # Posisi acuan pusat formasi (Robot 3) jika tersedia
        ref_gz_name = "x1_robot-3"
        ref_tag_pos = tag_world_positions.get(ref_gz_name, None)

        swarm_matrix_dict = {}

        # Loop setiap robot sebagai Tag (Initiator)
        for r_self in self.robots:
            gz_self = r_self["gz_name"]
            ros_self = r_self["ros_name"]
            id_self = r_self["anchor_id"]

            if gz_self not in tag_world_positions:
                continue

            pos_self = tag_world_positions[gz_self]
            pubs = self.pubs_dict[gz_self]

            links_list = []
            robot_matrix_row = {}

            # Ranging ke setiap robot lain sebagai Anchor (Responder)
            for r_other in self.robots:
                gz_other = r_other["gz_name"]
                ros_other = r_other["ros_name"]
                id_other = r_other["anchor_id"]

                if gz_other == gz_self or gz_other not in tag_world_positions:
                    continue

                pos_other = tag_world_positions[gz_other]
                true_dist = float(np.linalg.norm(pos_self - pos_other))

                # Noise Gaussian modul DWM1000 (std dev = 2.5 cm)
                noise = random.gauss(0.0, 0.025)
                raw_range = max(0.05, true_dist + noise)

                # Filter moving average 3 sampel (sesuai firmware fresh_link())
                hist = self.range_history[id_self][id_other]
                hist[2] = hist[1]
                hist[1] = hist[0]
                hist[0] = (raw_range + hist[1] + hist[2]) / 3.0 if hist[1] > 0 else raw_range

                current_range = hist[0]

                # Format payload JSON asli Cerdas UWB Tracker: {"A": "<ID>", "R": "<jarak>"}
                links_list.append({
                    "A": id_other,
                    "R": f"{current_range:.2f}"
                })
                robot_matrix_row[ros_other] = round(current_range, 2)

                # Publikasikan sensor_msgs/Range
                rng = Range()
                rng.header.stamp = now
                rng.header.frame_id = f"{ros_self}/base_link"
                rng.radiation_type = Range.ULTRASOUND
                rng.field_of_view = 6.28  # 360 derajat omnidirectional
                rng.min_range = 0.1
                rng.max_range = 50.0
                rng.range = float(current_range)
                for pub in pubs["ranges"][ros_other]:
                    pub.publish(rng)

            # 1. Publikasikan payload JSON Cerdas UWB Tracker
            json_payload = json.dumps({"links": links_list})
            str_msg = String()
            str_msg.data = json_payload
            for pub in pubs["json"]:
                pub.publish(str_msg)

            swarm_matrix_dict[ros_self] = robot_matrix_row

            # 2. Publikasikan posisi relatif terhadap Pusat Formasi (Robot 3)
            if ref_tag_pos is not None:
                rel_pos = pos_self - ref_tag_pos

                # PointStamped relatif terhadap Robot 3
                pt = PointStamped()
                pt.header.stamp = now
                pt.header.frame_id = "x1_robot_3/base_link"
                pt.point.x = float(rel_pos[0])
                pt.point.y = float(rel_pos[1])
                pt.point.z = float(rel_pos[2])
                for pub in pubs["rel_pos"]:
                    pub.publish(pt)

                # PoseWithCovarianceStamped
                px, py, pz, qx, qy, qz, qw = current_poses[gz_self]
                ps = PoseWithCovarianceStamped()
                ps.header.stamp = now
                ps.header.frame_id = "x1_robot_3/base_link"
                ps.pose.pose.position.x = float(rel_pos[0])
                ps.pose.pose.position.y = float(rel_pos[1])
                ps.pose.pose.position.z = float(rel_pos[2])
                ps.pose.pose.orientation.x = qx
                ps.pose.pose.orientation.y = qy
                ps.pose.pose.orientation.z = qz
                ps.pose.pose.orientation.w = qw

                cov = [0.0] * 36
                cov[0] = 0.0025  # Var(X) = (0.05m)^2
                cov[7] = 0.0025  # Var(Y) = (0.05m)^2
                cov[14] = 0.01   # Var(Z)
                ps.pose.covariance = cov
                for pub in pubs["rel_pose"]:
                    pub.publish(ps)

        # 3. Publikasikan Matriks Jarak Kawanan Global
        matrix_msg = String()
        matrix_msg.data = json.dumps(swarm_matrix_dict)
        self.swarm_matrix_pub.publish(matrix_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CerdasUWBSwarmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
