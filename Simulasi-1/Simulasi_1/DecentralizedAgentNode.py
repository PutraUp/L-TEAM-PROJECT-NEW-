#!/usr/bin/env python3
"""
Node ROS 2: DecentralizedAgentNode
Class pengendali terdesentralisasi (Decentralized Multi-Agent) untuk setiap robot:
- Menerima parameter ROS 2:
    - 'agent_id' (integer 0 s.d. total_agents - 1, default: 0)
    - 'total_agents' (integer, default: 5)
    - 'use_sim_time' (boolean, default: True)
- Memetakan ID 0..4 ke topik robot:
    - Topik Kecepatan: /agent_{id+1}/cmd_vel
    - Topik Odometri:  /agent_{id+1}/odom
    - Topik Sensor LiDAR: /x1_robot_{id+1}/scan (dan /agent_{id+1}/scan)
    - Topik UWB Ranging: /agent_{id+1}/uwb/links_json, /agent_{id+1}/uwb/range_to_agent_*
    - Topik Komunikasi Status Agen: /agent_{id+1}/status
- Memiliki modular control loop terdesentralisasi (obstacle avoidance, inter-agent coordination).
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import String

from Simulasi_1.front_end_transformation_node import front_end_transformation
from Simulasi_1.bicycle_kinematic_inversion_node import bicycle_kinematic_inversion
from Simulasi_1.idapbc_actor import IDAPBC_Actor


def extract_yaw_from_quaternion(q) -> float:
    """
    Mengubah orientasi Quaternion (x, y, z, w) dari topik /odom menjadi 1 sudut biasa (theta / yaw) dalam radian.
    Mendukung objek Quaternion ROS (dengan field x, y, z, w) maupun tuple/list.
    """
    if hasattr(q, 'x'):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class DecentralizedAgentNode(Node):
    """
    Class Node Pengendali Terdesentralisasi untuk Satu Agen Robot dalam Kawanan (Swarm).
    Didesain untuk dieksekusi secara instan berulang kali dengan parameter agent_id berbeda.
    """

    def __init__(self):
        super().__init__('decentralized_agent_node')

        # 1. Deklarasi Parameter ROS 2
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('total_agents', 5)
        self.declare_parameter('control_rate_hz', 50.0)    # Detak Jantung: Timer 50 Hz (0.02 detik)
        self.declare_parameter('safe_distance', 0.8)       # Batas aman halangan (meter)
        self.declare_parameter('max_linear_speed', 1.0)     # Kecepatan linear maks (m/s)
        self.declare_parameter('max_angular_speed', 1.5)    # Kecepatan putar maks (rad/s)
        self.declare_parameter('use_virtual_spring', True)  # Otomatis True sebagai penggerak saat use_ai=False
        self.declare_parameter('use_ai', True)              # Switch AI Otak Fisika IDAPBC_Actor (Fase 4)
        self.declare_parameter('target_distance', 2.0)      # Target jarak antar-titik depan (meter)
        self.declare_parameter('spring_k', 0.5)             # Konstanta pegas virtual
        self.declare_parameter('wheelbase', 0.5)            # Jarak antar-poros L (meter)
        self.declare_parameter('max_steer_rad', 0.523)      # Batas sudut setir (+-30 derajat)
        self.declare_parameter('model_weights_path', 'src/Simulasi-1/checkpoints/best_actor.pth') # Bobot model hasil training SAC

        # 2. Pembacaan Parameter
        self.agent_id: int = self.get_parameter('agent_id').value
        self.total_agents: int = self.get_parameter('total_agents').value
        self.control_rate: float = self.get_parameter('control_rate_hz').value
        self.safe_distance: float = self.get_parameter('safe_distance').value
        self.max_lin_speed: float = self.get_parameter('max_linear_speed').value
        self.max_ang_speed: float = self.get_parameter('max_angular_speed').value
        self.use_virtual_spring: bool = self.get_parameter('use_virtual_spring').value
        self.use_ai: bool = self.get_parameter('use_ai').value
        self.target_dist: float = self.get_parameter('target_distance').value
        self.spring_k: float = self.get_parameter('spring_k').value
        self.wheelbase: float = self.get_parameter('wheelbase').value
        self.max_steer: float = self.get_parameter('max_steer_rad').value

        # Tabel posisi spawn awal robot di dunia Gazebo (Diamond Formation)
        # Digunakan untuk mentransformasikan odometri lokal Gazebo DiffDrive (yang selalu mulai dari 0,0)
        # menjadi koordinat global dunia yang nyata.
        self.spawn_positions: Dict[int, Tuple[float, float, float]] = {
            0: (3.0, 0.0, 0.0),    # Agent 0 (Depan / Apex)
            1: (0.0, 2.5, 0.0),    # Agent 1 (Kiri)
            2: (0.0, 0.0, 0.0),    # Agent 2 (Pusat)
            3: (0.0, -2.5, 0.0),   # Agent 3 (Kanan)
            4: (-3.0, 0.0, 0.0),   # Agent 4 (Belakang)
        }

        # Parameter spawn posisi custom jika di-override via launch file
        self.declare_parameter('spawn_x', self.spawn_positions[self.agent_id][0])
        self.declare_parameter('spawn_y', self.spawn_positions[self.agent_id][1])
        self.declare_parameter('spawn_yaw', self.spawn_positions[self.agent_id][2])
        self.spawn_x = float(self.get_parameter('spawn_x').value)
        self.spawn_y = float(self.get_parameter('spawn_y').value)
        self.spawn_yaw = float(self.get_parameter('spawn_yaw').value)
        self.spawn_positions[self.agent_id] = (self.spawn_x, self.spawn_y, self.spawn_yaw)

        # Inisialisasi Otak Fisika AI (IDAPBC_Actor) dengan dukungan otomatis GPU/CUDA & Force Scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = IDAPBC_Actor(state_dim=4, force_scale=10.0).to(self.device)
        configured_path: str = str(self.get_parameter('model_weights_path').value)
        candidate_paths = [
            configured_path,
            os.path.join(os.path.expanduser('~'), 'l-team_ws', configured_path),
            os.path.join(os.path.expanduser('~'), 'l-team_ws/src/Simulasi-1/checkpoints/best_actor.pth'),
        ]
        loaded = False
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    self.policy.load_state_dict(torch.load(p, map_location=self.device))
                    self.get_logger().info(f"[Agent {self.agent_id}] Berhasil memuat bobot hasil latihan dari {p}")
                    loaded = True
                    break
                except Exception as e:
                    self.get_logger().warn(f"[Agent {self.agent_id}] Gagal memuat bobot dari {p}: {e}")
        if not loaded:
            self.get_logger().info(f"[Agent {self.agent_id}] Menggunakan bobot awal IDAPBC_Actor ({configured_path} belum ditemukan).")
        self.policy.eval()

        # Mapping penomoran topik (ID 0 -> /agent_0, ID 1 -> /agent_1, dst)
        self.prefix: str = f"/agent_{self.agent_id}"
        self.robot_name: str = f"x1_robot_{self.agent_id + 1}"

        # 3. State Internal & Posisi Sesuai Panduan:
        # Format 4 angka sederhana: [x, y, kecepatan, arah_hadap]
        self.ego_state: List[float] = [0.0, 0.0, 0.0, 0.0]
        self.neighbor_states: Dict[int, List[float]] = {}

        # Titik Kontrol Depan "Hidung" Robot (front_end_transformation, d = 0.25 meter)
        self.front_d: float = 0.25
        self.ego_front_point: List[float] = [0.0, 0.0]
        self.neighbor_front_points: Dict[int, List[float]] = {}
        self.last_fx: float = 0.0
        self.last_fy: float = 0.0
        self.last_v: float = 0.0
        self.last_delta: float = 0.0

        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_yaw: float = 0.0
        self.current_lin_vel: float = 0.0
        self.current_ang_vel: float = 0.0
        self.odom_received: bool = False

        # Sensor LiDAR Data
        self.min_front_dist: float = float('inf')
        self.min_left_dist: float = float('inf')
        self.min_right_dist: float = float('inf')
        self.scan_received: bool = False

        # Sensor UWB Data: neighbor_id -> measured_distance (m)
        self.uwb_neighbors_dist: Dict[int, float] = {}
        self.uwb_received: bool = False

        # 4. Konfigurasi QoS "Live Streaming" agar Tidak Ngelag (Sesuai Panduan Swarm):
        # - BEST_EFFORT : Tidak usah mengirim ulang paket data yang hilang di jalan.
        # - KEEP_LAST dengan depth=1 : Hanya simpan 1 data paling baru di memori, buang data lama agar tidak menumpuk.
        # - VOLATILE : Tidak perlu tahu riwayat masa lalu, cukup tahu data saat ini saja.
        self.qos_realtime = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # QoS Perintah Motor / cmd_vel (Reliable, depth=1, Volatile)
        # depth=1 & Volatile mencegah lag antrean kecepatan, Reliable menjamin kompatibilitas driver/bridge simulator
        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # 5. Publisher
        # Kontrol kecepatan ke driver motor robot
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f"{self.prefix}/cmd_vel",
            qos_cmd_vel
        )

        # Broadcast status lokal agen ke jaringan swarm ("Obrolan" antar-agen terdesentralisasi)
        self.status_pub = self.create_publisher(
            String,
            f"{self.prefix}/status",
            self.qos_realtime
        )

        # =====================================================================
        # 6. Pasang "Telinga" untuk Mendengar Posisi (Subscriber /odom)
        # =====================================================================

        # A. Dengar diri sendiri:
        # Robot ID 0 men-subscribe /agent_0/odom lalu menyimpannya di variabel ego_state
        self.ego_odom_sub = self.create_subscription(
            Odometry,
            f"/agent_{self.agent_id}/odom",
            self.ego_odom_callback,
            self.qos_realtime
        )

        # B. Dengar tetangga:
        # Pakai looping (for i in range(5)), jika i bukan ID dirinya sendiri,
        # maka robot akan men-subscribe /agent_i/odom dan menyimpannya di dalam dictionary neighbor_states
        self.neighbor_odom_subs = []
        for i in range(self.total_agents):
            if i != self.agent_id:
                self.neighbor_states[i] = [0.0, 0.0, 0.0, 0.0]
                sub = self.create_subscription(
                    Odometry,
                    f"/agent_{i}/odom",
                    lambda msg, neighbor_id=i: self.neighbor_odom_callback(msg, neighbor_id),
                    self.qos_realtime
                )
                self.neighbor_odom_subs.append(sub)

        # 7. Subscriber Sensor LiDAR & UWB
        # LiDAR Scan
        scan_topic = f"/{self.robot_name}/scan"
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            self.qos_realtime
        )

        # UWB Data Links JSON
        self.uwb_sub = self.create_subscription(
            String,
            f"{self.prefix}/uwb/links_json",
            self.uwb_callback,
            self.qos_realtime
        )

        # 8. Loop Kontrol Terdesentralisasi (Timer Callback)
        timer_period = 1.0 / self.control_rate
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            f"=====================================================\n"
            f" [DecentralizedAgentNode AKTIF]\n"
            f"  - Agent ID       : {self.agent_id} (Robot: {self.robot_name})\n"
            f"  - Total Agents   : {self.total_agents}\n"
            f"  - Topic cmd_vel  : {self.prefix}/cmd_vel\n"
            f"  - Topic odom     : {self.prefix}/odom\n"
            f"  - Topic LiDAR    : {scan_topic}\n"
            f"  - Topic UWB      : {self.prefix}/uwb/links_json\n"
            f"  - Status Topic   : {self.prefix}/status\n"
            f"  - Control Rate   : {self.control_rate} Hz\n"
            f"====================================================="
        )

    # =========================================================================
    # Callbacks Penerimaan Sensor & Odometri
    # =========================================================================

    def ego_odom_callback(self, msg: Odometry):
        """
        Dengar diri sendiri:
        Menerima data /agent_id/odom, menghitung posisi global di Gazebo dengan menambahkan titik spawn awal,
        lalu menyimpannya di variabel self.ego_state [x, y, kecepatan, arah_hadap].
        """
        init_x, init_y, init_yaw = self.spawn_positions.get(self.agent_id, (0.0, 0.0, 0.0))
        odom_x = float(msg.pose.pose.position.x)
        odom_y = float(msg.pose.pose.position.y)
        x = float(init_x + odom_x * math.cos(init_yaw) - odom_y * math.sin(init_yaw))
        y = float(init_y + odom_x * math.sin(init_yaw) + odom_y * math.cos(init_yaw))

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        kecepatan = float(math.sqrt(vx**2 + vy**2))
        odom_yaw = float(extract_yaw_from_quaternion(msg.pose.pose.orientation))
        arah_hadap = float(init_yaw + odom_yaw)

        # Simpan 4 angka sederhana sesuai panduan: [x, y, kecepatan, arah_hadap]
        self.ego_state = [x, y, kecepatan, arah_hadap]

        # Hitung titik depan "hidung" robot (front_end_transformation)
        # zx = x + d * cos(theta), zy = y + d * sin(theta)
        zx = float(x + self.front_d * math.cos(arah_hadap))
        zy = float(y + self.front_d * math.sin(arah_hadap))
        self.ego_front_point = [zx, zy]

        # Sinkronkan ke variabel internal
        self.current_x = x
        self.current_y = y
        self.current_lin_vel = kecepatan
        self.current_yaw = arah_hadap
        self.current_ang_vel = float(msg.twist.twist.angular.z)
        self.odom_received = True

    def neighbor_odom_callback(self, msg: Odometry, neighbor_id: int):
        """
        Dengar tetangga:
        Menerima data /agent_i/odom dari robot rekan (i != self.agent_id),
        menghitung posisi global rekan dengan menambahkan titik spawn awal rekan,
        lalu menyimpannya di dalam dictionary self.neighbor_states[neighbor_id].
        """
        init_x, init_y, init_yaw = self.spawn_positions.get(neighbor_id, (0.0, 0.0, 0.0))
        odom_x = float(msg.pose.pose.position.x)
        odom_y = float(msg.pose.pose.position.y)
        x = float(init_x + odom_x * math.cos(init_yaw) - odom_y * math.sin(init_yaw))
        y = float(init_y + odom_x * math.sin(init_yaw) + odom_y * math.cos(init_yaw))

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        kecepatan = float(math.sqrt(vx**2 + vy**2))
        odom_yaw = float(extract_yaw_from_quaternion(msg.pose.pose.orientation))
        arah_hadap = float(init_yaw + odom_yaw)

        # Simpan ke dictionary neighbor_states dengan key neighbor_id
        self.neighbor_states[neighbor_id] = [x, y, kecepatan, arah_hadap]

        # Hitung titik depan "hidung" robot tetangga (front_end_transformation)
        zx = float(x + self.front_d * math.cos(arah_hadap))
        zy = float(y + self.front_d * math.sin(arah_hadap))
        self.neighbor_front_points[neighbor_id] = [zx, zy]

    # Alias untuk kompatibilitas
    odom_callback = ego_odom_callback

    def scan_callback(self, msg: LaserScan):
        """Menerima data LiDAR untuk deteksi rintangan omnidirectional."""
        ranges = np.array(msg.ranges)
        # Filter nilai tidak valid (inf / nan) dengan max_range
        valid_ranges = np.where(
            np.isnan(ranges) | np.isinf(ranges) | (ranges < msg.range_min),
            msg.range_max,
            ranges
        )
        total_samples = len(valid_ranges)
        if total_samples == 0:
            return

        # Sektor depan (+/- 30 derajat dari sumbu x robot)
        # Sektor kiri (+30 s.d. +90 derajat)
        # Sektor kanan (-90 s.d. -30 derajat)
        deg_per_sample = 360.0 / total_samples

        front_idx = int(30.0 / deg_per_sample)
        front_left = valid_ranges[:front_idx]
        front_right = valid_ranges[-front_idx:]
        front_sector = np.concatenate([front_left, front_right])

        left_start = int(30.0 / deg_per_sample)
        left_end = int(90.0 / deg_per_sample)
        left_sector = valid_ranges[left_start:left_end]

        right_start = total_samples - int(90.0 / deg_per_sample)
        right_end = total_samples - int(30.0 / deg_per_sample)
        right_sector = valid_ranges[right_start:right_end]

        self.min_front_dist = float(np.min(front_sector)) if len(front_sector) > 0 else msg.range_max
        self.min_left_dist = float(np.min(left_sector)) if len(left_sector) > 0 else msg.range_max
        self.min_right_dist = float(np.min(right_sector)) if len(right_sector) > 0 else msg.range_max
        self.scan_received = True

    def uwb_callback(self, msg: String):
        """Menerima jarak UWB ke agen tetangga dalam format JSON Cerdas UWB Tracker."""
        try:
            data = json.loads(msg.data)
            # data: {"links": [{"A": "1002", "R": "3.89"}, ...]}
            if "links" in data:
                for link in data["links"]:
                    anchor_id = int(link["A"])
                    # Konversi Anchor ID 1001..1005 ke Agent ID 0..4
                    neighbor_id = (anchor_id - 1001)
                    dist = float(link["R"])
                    self.uwb_neighbors_dist[neighbor_id] = dist
                self.uwb_received = True
        except Exception as e:
            self.get_logger().warn(f"Gagal mem-parsing payload UWB: {e}")

    def peer_status_callback(self, msg: String, peer_id: int):
        """Menerima pesan status dari agen rekan di jaringan terdesentralisasi."""
        try:
            self.peers_state[peer_id] = json.loads(msg.data)
        except Exception:
            pass

    # =========================================================================
    # Loop Pengendalian Terdesentralisasi (Detak Jantung 50 Hz)
    # =========================================================================

    def control_loop(self):
        """
        Detak jantung robot (Timer 50 Hz / 0.02 detik):
        Setiap 0.02 detik sekali mengecek data terbaru di ego_state dan neighbor_states,
        mencetak ke terminal untuk verifikasi kelancaran komunikasi,
        lalu mengirim perintah gerak (Twist) melalui "mulut" publisher (/agent_i/cmd_vel).
        """
        # 1. Cek data terbaru & cetak status ke terminal
        mode_str = "AI IDAPBC" if self.use_ai else ("Pegas" if self.use_virtual_spring else "Diam")
        ego_str = f"[{self.ego_state[0]:.2f}, {self.ego_state[1]:.2f}, {self.ego_state[2]:.2f}, {self.ego_state[3]:.2f}]"
        self.get_logger().info(
            f"[Agent {self.agent_id} | {mode_str} 50Hz] ego: {ego_str} | F_pH=({self.last_fx:.3f}, {self.last_fy:.3f}) -> v={self.last_v:.3f}m/s, setir={math.degrees(self.last_delta):.1f}deg",
            throttle_duration_sec=0.5
        )

        # 2. Publikasikan status lokal agen ke jaringan swarm
        self.publish_agent_status()

        # 3. Hitung perintah gerak (Decentralized Control Law)
        cmd = self.compute_control_command()

        # 4. "Mulut" Robot: Kirimkan perintah Twist ke motor differential drive Gazebo
        self.cmd_vel_pub.publish(cmd)

    def publish_agent_status(self):
        """Mempublikasikan kondisi internal agen untuk dibaca oleh agen tetangga."""
        status_dict = {
            "agent_id": self.agent_id,
            "timestamp": self.get_clock().now().to_msg().sec,
            "ego_state": [round(val, 3) for val in self.ego_state],
            "ego_front_point": [round(val, 3) for val in self.ego_front_point],
            "neighbor_states": {
                str(k): [round(val, 3) for val in v] for k, v in self.neighbor_states.items()
            },
            "neighbor_front_points": {
                str(k): [round(val, 3) for val in v] for k, v in self.neighbor_front_points.items()
            },
            "uwb_neighbors": self.uwb_neighbors_dist,
            "min_front_dist": round(self.min_front_dist, 2),
        }
        msg = String()
        msg.data = json.dumps(status_dict)
        self.status_pub.publish(msg)

    def compute_control_command(self) -> Twist:
        """
        Kalkulasi perintah kecepatan (Linear & Angular).
        Mendukung Switch self.use_ai:
        - True  : Otak Fisika AI (IDAPBC_Actor)
        - False : Rumus Pegas Virtual Sederhana (Fase 3)
        """
        cmd = Twist()

        # Fitur Keamanan: Obstacle Avoidance menggunakan sensor LiDAR jika ada rintangan sangat dekat
        if self.scan_received and self.min_front_dist < (self.safe_distance * 0.5):
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 if self.min_left_dist > self.min_right_dist else -0.6
            return cmd

        if not self.odom_received:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        total_fx = 0.0
        total_fy = 0.0

        # ---------------------------------------------------------------------
        # 1. Komponen Pegas Virtual (Virtual Spring Baseline)
        # ---------------------------------------------------------------------
        if self.use_virtual_spring:
            zx_ego, zy_ego = self.ego_front_point
            v_spring_fx = 0.0
            v_spring_fy = 0.0
            valid_nbrs = 0

            for nid, n_pt in self.neighbor_front_points.items():
                if n_pt == [0.0, 0.0]:
                    continue

                dx = n_pt[0] - zx_ego
                dy = n_pt[1] - zy_ego
                dist = math.hypot(dx, dy)

                if dist < 0.05:
                    continue

                dist_error = dist - self.target_dist
                f_mag = self.spring_k * dist_error

                v_spring_fx += f_mag * (dx / dist)
                v_spring_fy += f_mag * (dy / dist)
                valid_nbrs += 1

            if valid_nbrs > 0:
                total_fx += v_spring_fx / math.sqrt(valid_nbrs)
                total_fy += v_spring_fy / math.sqrt(valid_nbrs)

        # ---------------------------------------------------------------------
        # 2. Komponen Otak Fisika AI (IDAPBC_Actor Port-Hamiltonian)
        # ---------------------------------------------------------------------
        if self.use_ai:
            ego_front = np.array([
                self.ego_front_point[0],
                self.ego_front_point[1],
                self.current_lin_vel,
                self.current_yaw
            ], dtype=np.float32)
            t_ego = torch.FloatTensor(ego_front).unsqueeze(0) # [1, 4]

            nbr_fronts_list = []
            for nid in sorted(self.neighbor_front_points.keys()):
                if nid in self.neighbor_states and nid != self.agent_id:
                    nbr_pt = self.neighbor_front_points[nid]
                    nbr_st = self.neighbor_states[nid]
                    nbr_fronts_list.append([nbr_pt[0], nbr_pt[1], nbr_st[2], nbr_st[3]])

            if len(nbr_fronts_list) > 0:
                neighbor_fronts = np.array(nbr_fronts_list, dtype=np.float32)
                t_neighbors = torch.FloatTensor(neighbor_fronts).unsqueeze(0).to(self.device) # [1, N, 4]
            else:
                t_neighbors = torch.empty((1, 0, 4), dtype=torch.float32, device=self.device)

            F_pH = self.policy(t_ego.to(self.device), t_neighbors)[0].detach().cpu() # [1, 2]
            ai_fx = float(F_pH[0, 0].item())
            ai_fy = float(F_pH[0, 1].item())

            total_fx += ai_fx
            total_fy += ai_fy

        # Jika kedua mode dinonaktifkan
        if not self.use_ai and not self.use_virtual_spring:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        fx = total_fx
        fy = total_fy

        # Batasi besar gaya tarikan agar kecepatan tidak melebihi max_lin_speed
        f_norm = math.hypot(fx, fy)
        if f_norm > self.max_lin_speed:
            fx = (fx / f_norm) * self.max_lin_speed
            fy = (fy / f_norm) * self.max_lin_speed

        # Deadband: jika formasi sangat presisi dan gaya sangat kecil (< 0.01 m/s), hentikan robot
        if f_norm < 0.01:
            fx = 0.0
            fy = 0.0

        # Teruskan ke Invers Kinematika Sepeda: (Fx, Fy) -> (v_target, delta)
        v_target, delta, theta_dot, raw_delta = bicycle_kinematic_inversion(
            Fx=fx,
            Fy=fy,
            theta=self.current_yaw,
            d=self.front_d,
            L=self.wheelbase,
            max_steer=self.max_steer,
            eps=1e-5
        )

        self.last_fx = fx
        self.last_fy = fy
        self.last_v = v_target
        self.last_delta = delta

        cmd.linear.x = float(v_target)
        cmd.angular.z = float(delta)
        return cmd

    def stop_robot(self):
        """Mengirimkan perintah berhenti ke motor robot."""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DecentralizedAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
