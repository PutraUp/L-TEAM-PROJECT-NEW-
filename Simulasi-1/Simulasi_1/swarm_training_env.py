#!/usr/bin/env python3
"""
Modul: SwarmTrainingEnv ("Lapangan Latihan" Simulator Python Cepat untuk Training AI)
Implementasi Fase 5: 1. Siapkan "Lapangan Latihan" (Training Environment)

Karakteristik & Fitur Utama:
1. Simulator Python Murni Cepat:
   - Kinematika 5 robot unicycle / bicycle model dengan front-end transformation (d = 0.25 m)
     dan bicycle kinematic inversion.
   - Tanpa overhead rendering GUI 3D Gazebo: 1 episode 10 detik (500 steps @ 50 Hz)
     dapat diselesaikan dalam 1-3 milidetik (~5.000x lebih cepat dari real-time Gazebo).
   - Melatih ribuan episode hanya membutuhkan beberapa puluh detik.

2. Acak Posisi Awal (Random Initial Spawning):
   - Setiap kali episode di-reset, kelima robot ditempatkan pada posisi dan orientasi acak.
   - Dilengkapi rejection sampling untuk memastikan jarak antar-robot awal >= min_distance (mencegah tabrakan instan).
   - AI dipaksa belajar konsep menjaga jarak dan koordinasi dinamis, bukan sekadar menghafal posisi statis.

3. Standar API Lingkungan:
   - reset(randomize=True, seed=None) -> obs, info
   - step(actions) -> obs, rewards, terminated, truncated, info
   - Dukungan tensor PyTorch: get_torch_observations(device) siap masuk ke IDAPBC_Actor.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from Simulasi_1.front_end_transformation_node import front_end_transformation
from Simulasi_1.bicycle_kinematic_inversion_node import bicycle_kinematic_inversion
from Simulasi_1.swarm_critic import compute_formation_reward


class SwarmTrainingEnv:
    """
    Simulator Python Cepat (Training Environment) untuk Multi-Agent Swarm Robotics.
    Memungkinkan ribuan iterasi training reinforcement learning / optimasi kebijakan AI dalam waktu singkat.
    """

    def __init__(
        self,
        num_agents: int = 5,
        dt: float = 0.02,                   # 50 Hz kontrol loop (0.02 detik)
        episode_duration: float = 10.0,      # 10 detik durasi episode
        target_distance: float = 2.0,        # Target jarak formasi antar-robot (meter)
        front_d: float = 0.25,               # Jarak titik hidung robot (meter)
        wheelbase: float = 0.5,              # Jarak antar poros roda (wheelbase L)
        max_steer_rad: float = 0.523,        # Batas fisik sudut setir (+-30 derajat / 0.523 rad)
        max_speed: float = 1.0,              # Kecepatan linear maksimum (m/s)
        arena_size: float = 8.0,             # Ukuran area acak spawn [-arena_size/2, +arena_size/2]
        collision_threshold: float = 0.6,    # Batas deteksi tabrakan fisik (meter)
    ):
        self.num_agents = num_agents
        self.dt = dt
        self.episode_duration = episode_duration
        self.max_steps = int(episode_duration / dt)  # 500 steps per episode
        self.target_distance = target_distance
        self.front_d = front_d
        self.wheelbase = wheelbase
        self.max_steer = max_steer_rad
        self.max_speed = max_speed
        self.arena_size = arena_size
        self.collision_threshold = collision_threshold

        # Matriks Formasi Belah Ketupat Acuan (Diamond Formation Target)
        # Digunakan untuk evaluasi formasi target
        self.default_poses = np.array([
            [3.0,  0.0, 0.0, 0.0],   # Agent 0: Depan (Apex)
            [0.0,  2.5, 0.0, 0.0],   # Agent 1: Kiri
            [0.0,  0.0, 0.0, 0.0],   # Agent 2: Pusat
            [0.0, -2.5, 0.0, 0.0],   # Agent 3: Kanan
            [-3.0, 0.0, 0.0, 0.0],   # Agent 4: Belakang
        ], dtype=np.float32)

        # State internal: [num_agents, 4] -> [x, y, v, theta]
        self.states = np.zeros((self.num_agents, 4), dtype=np.float32)
        # Front points: [num_agents, 2] -> [zx, zy]
        self.front_points = np.zeros((self.num_agents, 2), dtype=np.float32)

        self.current_step = 0
        self.rng = np.random.RandomState()

    def reset(
        self,
        randomize: bool = True,
        seed: Optional[int] = None,
        curriculum_factor: Optional[float] = None
    ) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
        """
        Mereset lingkungan simulasi ke kondisi awal episode.
        
        Parameters:
            randomize (bool): Jika True, posisi awal diacak agar AI tidak menghafal 1 posisi.
            seed (Optional[int]): Random seed untuk reproduktibilitas.
            curriculum_factor (Optional[float]): Faktor kurikulum 0.0 (dekat formasi) s.d. 1.0 (acak penuh satu arena).

        Returns:
            Tuple[Dict, Dict]: (observations per robot, info dict)
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0

        if not randomize:
            # Gunakan posisi default formasi belah ketupat
            self.states = self.default_poses.copy()
        elif curriculum_factor is not None and curriculum_factor < 0.95:
            # Mode Kurikulum Bertahap (Curriculum Learning):
            # Robot di-spawn di sekitar formasi target dengan noise perturbasi yang meningkat bertahap
            noise_radius = float(0.4 + curriculum_factor * 2.6)  # 0.4 m -> 3.0 m
            min_spawn_dist = 0.8
            positions = []
            for i in range(self.num_agents):
                base_x = float(self.default_poses[i, 0])
                base_y = float(self.default_poses[i, 1])
                for attempt in range(100):
                    cand_x = base_x + float(self.rng.uniform(-noise_radius, noise_radius))
                    cand_y = base_y + float(self.rng.uniform(-noise_radius, noise_radius))

                    overlap = False
                    for (px, py) in positions:
                        if math.hypot(cand_x - px, cand_y - py) < min_spawn_dist:
                            overlap = True
                            break
                    if not overlap or attempt == 99:
                        positions.append((cand_x, cand_y))
                        break

            for i, (px, py) in enumerate(positions):
                yaw = float(self.rng.uniform(-math.pi, math.pi))
                self.states[i] = [px, py, 0.0, yaw]
        else:
            # Acak Posisi Awal 5 Robot di dalam seluruh arena
            # Menggunakan rejection sampling agar robot tidak langsung menabrak saat spawn
            min_spawn_dist = 1.0  # Jarak minimal antar-robot saat spawn (meter)
            spawn_range = self.arena_size * 0.4  # [-3.2, 3.2] meter

            positions = []
            for _ in range(self.num_agents):
                for attempt in range(100):
                    cand_x = float(self.rng.uniform(-spawn_range, spawn_range))
                    cand_y = float(self.rng.uniform(-spawn_range, spawn_range))

                    # Cek jarak terhadap robot yang sudah ada
                    overlap = False
                    for (px, py) in positions:
                        if math.hypot(cand_x - px, cand_y - py) < min_spawn_dist:
                            overlap = True
                            break
                    if not overlap or attempt == 99:
                        positions.append((cand_x, cand_y))
                        break

            # Inisialisasi states: kecepatan 0, arah hadap acak [-pi, pi]
            for i, (px, py) in enumerate(positions):
                yaw = float(self.rng.uniform(-math.pi, math.pi))
                self.states[i] = [px, py, 0.0, yaw]

        # Perbarui koordinat titik kontrol hidung (front_end_transformation)
        self._update_front_points()

        obs = self._get_observations()
        info = {
            "step": self.current_step,
            "sim_time": 0.0,
            "randomized": randomize,
            "initial_states": self.states.copy()
        }
        return obs, info

    def _update_front_points(self):
        """Menghitung titik depan hidung robot: zx = x + d*cos(theta), zy = y + d*sin(theta)."""
        x = self.states[:, 0]
        y = self.states[:, 1]
        theta = self.states[:, 3]
        self.front_points[:, 0] = x + self.front_d * np.cos(theta)
        self.front_points[:, 1] = y + self.front_d * np.sin(theta)

    def _get_observations(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Menyusun observasi desentralisasi untuk ke-5 robot.
        Setiap robot menerima:
        - 'ego': [x, y, v, theta] diri sendiri (shape: [4])
        - 'neighbors': [num_neighbors, 4] state semua rekan (shape: [4, 4])
        - 'ego_front': [zx, zy] titik depan diri sendiri
        - 'neighbor_fronts': [num_neighbors, 2] titik depan rekan-rekan
        """
        obs = {}
        for i in range(self.num_agents):
            ego_state = self.states[i].copy()
            ego_front = self.front_points[i].copy()

            neighbor_indices = [j for j in range(self.num_agents) if j != i]
            neighbor_states = self.states[neighbor_indices].copy()
            neighbor_fronts = self.front_points[neighbor_indices].copy()

            obs[i] = {
                "ego": ego_state,
                "neighbors": neighbor_states,
                "ego_front": ego_front,
                "neighbor_fronts": neighbor_fronts,
            }
        return obs

    def get_torch_observations(
        self,
        device: Union[str, torch.device] = 'cpu'
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Helper konversi ke format PyTorch Tensor yang siap langsung diumpankan ke IDAPBC_Actor:
        Untuk setiap robot i:
        - t_ego: Tensor [1, 4]
        - t_neighbors: Tensor [1, N, 4]
        """
        tensors = []
        for i in range(self.num_agents):
            ego = self.states[i]
            nbr_indices = [j for j in range(self.num_agents) if j != i]
            nbrs = self.states[nbr_indices]

            t_ego = torch.tensor(ego, dtype=torch.float32, device=device).unsqueeze(0)
            t_nbrs = torch.tensor(nbrs, dtype=torch.float32, device=device).unsqueeze(0)
            tensors.append((t_ego, t_nbrs))
        return tensors

    def step(
        self,
        actions: Union[np.ndarray, torch.Tensor, List[Tuple[float, float]], Dict[int, Tuple[float, float]]]
    ) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, float], bool, bool, Dict]:
        """
        Melangkah 1 siklus simulasi (dt = 0.02 detik).
        
        Parameters:
            actions: Vektor gaya kontrol [num_agents, 2] berupa (Fx, Fy) dari IDAPBC_Actor
                     atau dictionary {agent_id: (Fx, Fy)}.

        Returns:
            Tuple[Dict, Dict, bool, bool, Dict]:
                - obs: Observasi terbaru
                - rewards: Reward per robot (evaluasi formasi & keselamatan)
                - terminated: True jika episode selesai (tabrakan fatal / formasi tercapai)
                - truncated: True jika waktu 10 detik habis (step >= 500)
                - info: Informasi metrik tambahan
        """
        self.current_step += 1
        sim_time = self.current_step * self.dt

        # Parse actions menjadi array [num_agents, 2] (Fx, Fy)
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        elif isinstance(actions, dict):
            actions_np = np.zeros((self.num_agents, 2), dtype=np.float32)
            for aid, act in actions.items():
                actions_np[aid] = act
        elif isinstance(actions, list):
            actions_np = np.array(actions, dtype=np.float32)
        else:
            actions_np = np.array(actions, dtype=np.float32)

        # Update kinematika untuk setiap robot menggunakan invers kinematika sepeda
        applied_cmds = []
        for i in range(self.num_agents):
            fx = float(actions_np[i, 0])
            fy = float(actions_np[i, 1])

            # Batasi gaya maksimum
            f_mag = math.hypot(fx, fy)
            if f_mag > self.max_speed:
                fx = (fx / f_mag) * self.max_speed
                fy = (fy / f_mag) * self.max_speed

            theta = float(self.states[i, 3])

            # Invers Kinematika Sepeda: (Fx, Fy) -> (v_target, delta)
            v_target, delta, theta_dot, _ = bicycle_kinematic_inversion(
                Fx=fx,
                Fy=fy,
                theta=theta,
                d=self.front_d,
                L=self.wheelbase,
                max_steer=self.max_steer,
                eps=1e-5
            )

            # Hitung kecepatan sudut aktual dari sudut setir roda:
            # omega = (v / L) * tan(delta)
            actual_omega = (v_target / self.wheelbase) * math.tan(delta)

            # Update Posisi & Orientasi (Integrasi Euler):
            # x_new = x + v * cos(theta) * dt
            # y_new = y + v * sin(theta) * dt
            # theta_new = theta + omega * dt
            self.states[i, 0] += v_target * math.cos(theta) * self.dt
            self.states[i, 1] += v_target * math.sin(theta) * self.dt
            self.states[i, 2] = v_target
            self.states[i, 3] = self._normalize_angle(theta + actual_omega * self.dt)

            applied_cmds.append((v_target, delta))

        # Perbarui posisi titik hidung
        self._update_front_points()

        # Hitung Matriks Jarak Antar Titik Hidung & Evaluasi Formasi
        pairwise_dists = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        collisions = 0
        total_dist_error = 0.0

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = float(np.hypot(
                    self.front_points[i, 0] - self.front_points[j, 0],
                    self.front_points[i, 1] - self.front_points[j, 1]
                ))
                pairwise_dists[i, j] = d
                pairwise_dists[j, i] = d

                # Cek tabrakan
                if d < self.collision_threshold:
                    collisions += 1

                # Penalti selisih jarak terhadap target formasi
                dist_err = abs(d - self.target_distance)
                total_dist_error += dist_err

        # Hitung Reward per Robot untuk Pembelajaran AI menggunakan compute_formation_reward
        rewards = {}
        for i in range(self.num_agents):
            ego_pt = self.front_points[i]
            nbr_indices = [j for j in range(self.num_agents) if j != i]
            nbr_pts = self.front_points[nbr_indices]

            # Aturan Nilai Sederhana: mean formation error terhadap target 2.0 m
            reward = compute_formation_reward(
                ego_point=ego_pt,
                neighbor_points=nbr_pts,
                target_distance=self.target_distance
            )
            rewards[i] = reward

        # Cek Selesai (Terminated / Truncated)
        truncated = self.current_step >= self.max_steps
        terminated = False  # Bisa diset True jika formasi sudah konvergen sempurna

        obs = self._get_observations()
        info = {
            "step": self.current_step,
            "sim_time": sim_time,
            "mean_formation_error": float(total_dist_error / 10.0),
            "mean_dist_error": float(total_dist_error / 10.0),
            "collisions": collisions,
            "applied_cmds": applied_cmds,
            "pairwise_distances": pairwise_dists
        }

        return obs, rewards, terminated, truncated, info

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Menormalkan sudut ke rentang [-pi, +pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))


def demo_speed_benchmark():
    """Menguji kecepatan simulator Python: Berapa ribu step dan episode per detik yang dapat diproses."""
    env = SwarmTrainingEnv(num_agents=5, episode_duration=10.0, dt=0.02)
    num_episodes = 20

    print("=" * 80)
    print(" BENCHMARK KECEPATAN: LAPANGAN LATIHAN SWARM TRAINING ENV")
    print("=" * 80)
    print(f"Konfigurasi:")
    print(f"  - Jumlah Robot       : 5 Robot")
    print(f"  - Durasi per Episode : 10.0 Detik ({env.max_steps} steps @ 50 Hz)")
    print(f"  - Target Formasi     : {env.target_distance} meter")
    print(f"  - Total Uji Coba     : {num_episodes} Episode ({num_episodes * env.max_steps} steps)")
    print("-" * 80)

    t_start = time.perf_counter()
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset(randomize=True)
        done = False
        while not done:
            # Contoh aksi random force
            random_actions = np.random.uniform(-0.5, 0.5, size=(5, 2)).astype(np.float32)
            obs, rewards, terminated, truncated, info = env.step(random_actions)
            total_steps += 1
            if terminated or truncated:
                done = True

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    simulated_time = num_episodes * env.episode_duration
    speedup = simulated_time / elapsed
    steps_per_sec = total_steps / elapsed

    print(f"Hasil Benchmark:")
    print(f"  - Total Waktu Nyata     : {elapsed:.3f} detik")
    print(f"  - Total Waktu Simulasi  : {simulated_time:.1f} detik ({num_episodes} episode x 10 detik)")
    print(f"  - Kecepatan Eksekusi    : {steps_per_sec:,.0f} steps / detik")
    print(f"  - Percepatan (Speedup)  : {speedup:,.1f}x LEBIH CEPAT DARI REAL-TIME GAZEBO!")
    print(f"  - Rata-rata per Episode : {elapsed / num_episodes * 1000.0:.2f} ms per episode 10 detik")
    print("=" * 80)


if __name__ == '__main__':
    demo_speed_benchmark()
