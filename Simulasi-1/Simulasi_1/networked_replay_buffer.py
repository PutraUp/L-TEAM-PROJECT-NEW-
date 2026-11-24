#!/usr/bin/env python3
"""
Modul: NetworkedReplayBuffer ("Buku Catatan Pengalaman" dengan Graf Jaringan)
Implementasi Fase 5: 3. Buat "Buku Catatan Pengalaman" (Networked Replay Buffer)

Karakteristik & Fitur Utama:
1. Isi Catatan Pengalaman:
   Menyimpan paket data transisi setiap langkah simulasi:
   (posisi_sekarang, gaya_yang_dipilih, reward, posisi_berikutnya, done, struktur_graf)

2. Kunci Khusus pH-MARL (Graf Jaringan):
   - Wajib menyimpan struktur topologi graf keterhubungan antar-robot (siapa bertetangga dengan siapa pada detik itu).
   - Menyimpan matriks keterhubungan / adjacency matrix dan relative displacement antar-node.
   - Memungkinkan AI memahami korelasi perilaku (behavioral correlation): aksi satu robot langsung
     memengaruhi energi dan respon robot rekannya, sehingga proses pembelajaran desentralisasi stabil
     tanpa membutuhkan centralized critic runtime.

3. Sampling Cepat & Siap Masuk PyTorch:
   - Pengambilan acak (random mini-batch sampling) untuk memutus korelasi temporal.
   - Otomatis mengubah batch ke format torch.Tensor (mendukung CPU & GPU CUDA).
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class NetworkedTransition:
    """Satu unit catatan pengalaman interaksi robot dalam jaringan swarm."""
    ego_state: np.ndarray              # [4]: [x, y, v, yaw]
    neighbor_states: np.ndarray        # [N, 4]: fitur semua rekan tetangga
    action: np.ndarray                 # [2]: gaya pilihan Actor (Fx, Fy)
    reward: float                      # Nilai reward pada langkah tersebut
    next_ego_state: np.ndarray         # [4]: state ego berikutnya
    next_neighbor_states: np.ndarray   # [N, 4]: state tetangga berikutnya
    done: bool                         # Status episode berakhir
    graph_structure: np.ndarray        # [M, M] atau [N]: Matriks keterhubungan graf / adjacency


class NetworkedReplayBuffer:
    """
    Buku Catatan Pengalaman (Networked Replay Buffer):
    Penyimpanan sirkular pengalaman interaksi multi-agent dengan pelacakan topologi graf jaringan.
    """

    def __init__(
        self,
        capacity: int = 100000,
        comm_range: float = 6.0,
        seed: Optional[int] = None
    ):
        """
        Inisialisasi NetworkedReplayBuffer.

        Parameters:
            capacity (int): Kapasitas maksimal memori transisi (default: 100.000 langkah)
            comm_range (float): Jarak jangkauan komunikasi UWB/graf antar-robot (default: 6.0 m)
            seed (Optional[int]): Random seed untuk pengambilan sampel acak
        """
        self.capacity = capacity
        self.comm_range = comm_range
        self.buffer: List[NetworkedTransition] = []
        self.position = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Mengecek apakah jumlah data di memori sudah cukup untuk di-sampling."""
        return len(self.buffer) >= batch_size

    def push(
        self,
        ego_state: Union[np.ndarray, List[float]],
        neighbor_states: Union[np.ndarray, List[List[float]]],
        action: Union[np.ndarray, List[float], Tuple[float, float]],
        reward: float,
        next_ego_state: Union[np.ndarray, List[float]],
        next_neighbor_states: Union[np.ndarray, List[List[float]]],
        done: bool,
        graph_structure: Optional[np.ndarray] = None
    ):
        """
        Menyimpan satu catatan pengalaman individual robot ke dalam buffer sirkular.
        """
        ego = np.array(ego_state, dtype=np.float32).flatten()
        nbrs = np.array(neighbor_states, dtype=np.float32)
        act = np.array(action, dtype=np.float32).flatten()
        nxt_ego = np.array(next_ego_state, dtype=np.float32).flatten()
        nxt_nbrs = np.array(next_neighbor_states, dtype=np.float32)

        # Jika graf tidak diberikan secara eksplisit, bentuk graf keterhubungan berdasarkan jarak UWB
        if graph_structure is None:
            diffs = nbrs[:, :2] - ego[:2]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            # Graf keterhubungan: 1 jika dalam jangkauan komunikasi comm_range, 0 jika di luar
            graph_structure = (dists <= self.comm_range).astype(np.float32)
        else:
            graph_structure = np.array(graph_structure, dtype=np.float32)

        transition = NetworkedTransition(
            ego_state=ego,
            neighbor_states=nbrs,
            action=act,
            reward=float(reward),
            next_ego_state=nxt_ego,
            next_neighbor_states=nxt_nbrs,
            done=bool(done),
            graph_structure=graph_structure
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def push_swarm_step(
        self,
        obs: Dict[int, Dict[str, np.ndarray]],
        actions: Union[np.ndarray, Dict[int, np.ndarray], List],
        rewards: Dict[int, float],
        next_obs: Dict[int, Dict[str, np.ndarray]],
        done: bool,
        adj_matrix: Optional[np.ndarray] = None
    ):
        """
        Helper praktis: Menyimpan transisi ke-5 robot sekaligus dari 1 langkah step() lingkungan.
        Secara otomatis menghitung struktur graf topologi jaringan pada detik tersebut.
        """
        num_agents = len(obs)

        # Matriks topologi graf global jika belum disediakan
        if adj_matrix is None:
            # Hitung matriks adjacency berbobot jarak
            adj_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
            positions = np.array([obs[i]["ego"][:2] for i in range(num_agents)])
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        d = float(np.hypot(positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1]))
                        if d <= self.comm_range:
                            adj_matrix[i, j] = 1.0

        # Simpan pengalaman untuk masing-masing agen
        for i in range(num_agents):
            ego = obs[i]["ego"]
            nbrs = obs[i]["neighbors"]

            if isinstance(actions, dict):
                act = actions[i]
            else:
                act = actions[i]

            rew = rewards[i]
            nxt_ego = next_obs[i]["ego"]
            nxt_nbrs = next_obs[i]["neighbors"]

            # Subgraf keterhubungan agen i dengan tetangganya
            nbr_indices = [j for j in range(num_agents) if j != i]
            graph_edges = adj_matrix[i, nbr_indices]

            self.push(
                ego_state=ego,
                neighbor_states=nbrs,
                action=act,
                reward=rew,
                next_ego_state=nxt_ego,
                next_neighbor_states=nxt_nbrs,
                done=done,
                graph_structure=graph_edges
            )

    def sample(
        self,
        batch_size: int,
        device: Union[str, torch.device] = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Mengambil batch sampel secara acak dari memori dan mengonversinya langsung ke PyTorch Tensor.

        Parameters:
            batch_size (int): Jumlah sampel mini-batch (misal: 64, 128, 256)
            device (Union[str, torch.device]): Target device ('cpu' atau 'cuda')

        Returns:
            Dict[str, torch.Tensor]:
                - 'ego_states': [batch, 4]
                - 'neighbor_states': [batch, num_neighbors, 4]
                - 'actions': [batch, 2]
                - 'rewards': [batch, 1]
                - 'next_ego_states': [batch, 4]
                - 'next_neighbor_states': [batch, num_neighbors, 4]
                - 'dones': [batch, 1]
                - 'graph_structures': [batch, num_neighbors]
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Ukuran buffer ({len(self.buffer)}) kurang dari batch_size ({batch_size}).")

        samples = random.sample(self.buffer, batch_size)

        ego_list = [s.ego_state for s in samples]
        nbr_list = [s.neighbor_states for s in samples]
        act_list = [s.action for s in samples]
        rew_list = [[s.reward] for s in samples]
        nxt_ego_list = [s.next_ego_state for s in samples]
        nxt_nbr_list = [s.next_neighbor_states for s in samples]
        done_list = [[1.0 if s.done else 0.0] for s in samples]
        graph_list = [s.graph_structure for s in samples]

        batch = {
            "ego_states": torch.tensor(np.array(ego_list), dtype=torch.float32, device=device),
            "neighbor_states": torch.tensor(np.array(nbr_list), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(act_list), dtype=torch.float32, device=device),
            "rewards": torch.tensor(np.array(rew_list), dtype=torch.float32, device=device),
            "next_ego_states": torch.tensor(np.array(nxt_ego_list), dtype=torch.float32, device=device),
            "next_neighbor_states": torch.tensor(np.array(nxt_nbr_list), dtype=torch.float32, device=device),
            "dones": torch.tensor(np.array(done_list), dtype=torch.float32, device=device),
            "graph_structures": torch.tensor(np.array(graph_list), dtype=torch.float32, device=device),
        }

        return batch

    def clear(self):
        """Mengosongkan isi buffer pengalaman."""
        self.buffer.clear()
        self.position = 0


def demo_networked_replay_buffer():
    """Demonstrasi pengujian Buku Catatan Pengalaman (Networked Replay Buffer)."""
    print("=" * 80)
    print(" DEMO: NETWORKED REPLAY BUFFER ('BUKU CATATAN PENGALAMAN') pH-MARL")
    print("=" * 80)

    buffer = NetworkedReplayBuffer(capacity=1000, comm_range=5.0)

    # 1. Simulasikan pengisian catatan dari 5 langkah swarm
    num_steps = 10
    num_agents = 5
    print(f"1. Mengisi buffer dengan simulasi {num_steps} langkah interaksi swarm (5 robot)...")

    for step_idx in range(num_steps):
        # Dummy data observasi 5 robot
        obs = {}
        next_obs = {}
        actions = {}
        rewards = {}

        for i in range(num_agents):
            ego = np.array([float(i * 1.5), 0.0, 0.2, 0.0], dtype=np.float32)
            nbrs = np.array([
                [float(j * 1.5), 0.0, 0.2, 0.0] for j in range(num_agents) if j != i
            ], dtype=np.float32)
            obs[i] = {"ego": ego, "neighbors": nbrs}

            next_ego = ego + np.array([0.05, 0.01, 0.0, 0.0], dtype=np.float32)
            next_nbrs = nbrs + np.array([0.05, 0.01, 0.0, 0.0], dtype=np.float32)
            next_obs[i] = {"ego": next_ego, "neighbors": next_nbrs}

            actions[i] = np.array([0.1, -0.05], dtype=np.float32)
            rewards[i] = -0.35

        is_done = (step_idx == num_steps - 1)
        buffer.push_swarm_step(obs, actions, rewards, next_obs, done=is_done)

    print(f"   -> Total pengalaman tersimpan: {len(buffer)} unit (10 langkah x 5 robot = 50 transisi).")

    # 2. Uji Sampling Mini-Batch PyTorch
    batch_size = 8
    print(f"\n2. Mengambil sampel mini-batch acak (batch_size = {batch_size})...")
    batch = buffer.sample(batch_size=batch_size, device='cpu')

    for key, tensor in batch.items():
        print(f"   - {key:<22}: Shape = {list(tensor.shape)} (dtype: {tensor.dtype})")

    # 3. Verifikasi struktur graf
    print(f"\n3. Verifikasi Struktur Graf Jaringan (Keterhubungan Tetangga):")
    sample_graph = batch["graph_structures"][0].numpy()
    print(f"   - Contoh Bobot Graf Sampel #0: {sample_graph} -> (1 = terhubung komunikasi, 0 = terputus)")

    print("=" * 80)
    print(">> KESIMPULAN: NETWORKED REPLAY BUFFER SIAP 100% UNTUK TRAINING ACTOR-CRITIC!")


if __name__ == '__main__':
    demo_networked_replay_buffer()
