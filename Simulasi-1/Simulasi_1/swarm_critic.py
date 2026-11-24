#!/usr/bin/env python3
"""
Modul: SwarmCritic & Reward Function ("Juri" & "Aturan Nilai" Pelatihan pH-MARL)
Implementasi Fase 5: 2. Buat "Juri" & "Aturan Nilai" (Critic & Reward Function)

Komponen:
1. Critic (Jaringan Penilai / "Juri"):
   - Berfungsi sebagai evaluator saat pelatihan Actor-Critic (hanya digunakan saat training, dibuang saat run Gazebo).
   - Menerima:
       a. Posisi & status robot ego serta tetangga-tetangganya (state dinamis via GraphAttentionLayer).
       b. Pilihan gaya kontrol (Fx, Fy) yang diajukan oleh Actor (IDAPBC_Actor).
   - Menghasilkan:
       Skor Q (skalar Q-value) yang menilai seberapa baik keputusan manuver tersebut terhadap masa depan formasi.

2. Aturan Nilai (Reward Function) Sederhana:
   - Pada pH-MARL (Port-Hamiltonian Multi-Agent Reinforcement Learning), kestabilan dan hukum kekekalan
     energi sudah dijamin oleh struktur pasif (J - R) * grad(H).
   - Rumus reward tidak memerlukan reward shaping rumit, melainkan cukup sparse & transparan:
       mean_formation_error = (1 / N) * sum(|dist_ij - target_distance|)
       reward = - mean_formation_error
     Semakin kecil selisih jaraknya terhadap 2.0 meter, semakin tinggi nilainya (mendekati 0).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Simulasi_1.graph_attention_layer import GraphAttentionLayer


def compute_formation_reward(
    ego_point: Union[np.ndarray, List[float], torch.Tensor],
    neighbor_points: Union[np.ndarray, List[List[float]], torch.Tensor],
    target_distance: float = 2.0,
    min_dist_penalty: float = 0.5,
    collision_penalty: float = 0.0
) -> float:
    """
    Aturan Nilai (Reward Function Sederhana) pH-MARL:
    Menghitung rata-rata selisih jarak antar-titik depan robot dibanding target 2.0 meter (mean formation error).
    
    Semakin kecil selisihnya, semakin besar reward-nya (maksimal 0 saat formasi sempurna).

    Parameters:
        ego_point: Titik depan robot ego [zx, zy] (meter)
        neighbor_points: Daftar titik depan robot tetangga [[zx_1, zy_1], ...] (meter)
        target_distance: Jarak formasi yang diinginkan (default: 2.0 meter)
        min_dist_penalty: Jarak batas tabrakan fisik jika ada penalti tabrakan tambahan (opsional)
        collision_penalty: Penalti jika melanggar min_dist_penalty (default: 0.0 untuk reward murni)

    Returns:
        float: Nilai reward skalar (makin dekat 0 makin bagus).
    """
    if isinstance(ego_point, torch.Tensor):
        ego = ego_point.detach().cpu().numpy().flatten()[:2]
    else:
        ego = np.array(ego_point, dtype=np.float32).flatten()[:2]

    if isinstance(neighbor_points, torch.Tensor):
        nbrs = neighbor_points.detach().cpu().numpy().reshape(-1, 2)
    else:
        nbrs = np.array(neighbor_points, dtype=np.float32).reshape(-1, 2)

    if len(nbrs) == 0:
        return 0.0

    # Hitung jarak Euclidean ke semua rekan
    diffs = nbrs - ego
    distances = np.hypot(diffs[:, 0], diffs[:, 1])

    # Hitung selisih jarak absolut terhadap target 2.0 meter
    errors = np.abs(distances - target_distance)
    mean_formation_error = float(np.mean(errors))

    # Reward dasar sederhana: - mean_formation_error
    reward = - mean_formation_error

    # Tambahan penalti keamanan jika sangat dekat (opsional, jika diset > 0)
    if collision_penalty > 0.0 and np.any(distances < min_dist_penalty):
        reward -= collision_penalty

    return reward


class SwarmCritic(nn.Module):
    """
    Jaringan Penilai (Critic / "Juri"):
    Menggabungkan representasi Graph Attention dari lingkungan multi-robot dengan aksi gaya kontrol (Fx, Fy)
    untuk memprediksi nilai Q(s, a).
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128
    ):
        """
        Inisialisasi SwarmCritic.

        Parameters:
            state_dim (int): Dimensi fitur per robot (default: 4 -> [x, y, kecepatan, arah_hadap])
            action_dim (int): Dimensi aksi gaya kontrol Actor (default: 2 -> [Fx, Fy])
            embed_dim (int): Dimensi embedding GraphAttentionLayer (default: 64)
            num_heads (int): Jumlah attention heads (default: 4)
            hidden_dim (int): Dimensi neuron layer tersembunyi MLP (default: 128)
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # 1. "Mata Pintar" Critic: Memproses observasi tetangga dinamis
        self.gat = GraphAttentionLayer(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        # 2. MLP Head Penilai Q(s, a):
        # Menerima gabungan embedding state (embed_dim) + aksi gaya kontrol Actor (action_dim)
        in_features = embed_dim + action_dim
        self.q_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: Skalar skor Q-value
        )

    def forward(
        self,
        ego_state: Union[torch.Tensor, List[float]],
        neighbor_states: Union[torch.Tensor, List[List[float]]],
        action: Union[torch.Tensor, List[float], Tuple[float, float]]
    ) -> torch.Tensor:
        """
        Forward Pass Critic:
        Mengevaluasi pasangan (state, action) dan menghasilkan skor Q.

        Parameters:
            ego_state: State robot ego [batch_size, 4] atau [4]
            neighbor_states: State tetangga [batch_size, num_neighbors, 4]
            action: Aksi gaya dari Actor [batch_size, 2] atau [2] -> (Fx, Fy)

        Returns:
            torch.Tensor: Skor Q [batch_size, 1]
        """
        device = next(self.parameters()).device

        # Standarisasi tensor ego_state: [batch, 4]
        if not isinstance(ego_state, torch.Tensor):
            ego_tensor = torch.tensor(ego_state, dtype=torch.float32, device=device)
        else:
            ego_tensor = ego_state.to(device)
        if ego_tensor.ndim == 1:
            ego_tensor = ego_tensor.unsqueeze(0)

        # Standarisasi tensor neighbor_states: [batch, num_neighbors, 4]
        if not isinstance(neighbor_states, torch.Tensor):
            nbr_tensor = torch.tensor(neighbor_states, dtype=torch.float32, device=device)
        else:
            nbr_tensor = neighbor_states.to(device)
        if nbr_tensor.ndim == 2:
            nbr_tensor = nbr_tensor.unsqueeze(0)

        # Standarisasi tensor action: [batch, 2]
        if not isinstance(action, torch.Tensor):
            act_tensor = torch.tensor(action, dtype=torch.float32, device=device)
        else:
            act_tensor = action.to(device)
        if act_tensor.ndim == 1:
            act_tensor = act_tensor.unsqueeze(0)

        # 1. Ekstrak representasi hubungan spasial formasi melalui Graph Attention
        state_embed, _ = self.gat(ego_tensor, nbr_tensor)  # [batch, embed_dim]

        # 2. Gabungkan state representation dengan gaya yang diajukan Actor
        sa_concat = torch.cat([state_embed, act_tensor], dim=-1)  # [batch, embed_dim + 2]

        # 3. Hitung skor evaluasi Q(s, a)
        q_value = self.q_net(sa_concat)  # [batch, 1]

        return q_value


def demo_critic_and_reward():
    """Demonstrasi pengujian Juri (Critic) & Aturan Nilai (Reward)."""
    print("=" * 80)
    print(" DEMO: CRITIC ('JURI') & REWARD FUNCTION ('ATURAN NILAI') pH-MARL")
    print("=" * 80)

    # Inisialisasi Critic
    critic = SwarmCritic(state_dim=4, action_dim=2, embed_dim=64, hidden_dim=128)
    critic.eval()

    # Contoh 1: Robot ego di (0, 0), 4 tetangga di jarak bervariasi
    ego = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
    neighbors = torch.FloatTensor([[[2.0, 0.0, 0.0, 0.0],
                                    [0.0, 2.0, 0.0, 0.0],
                                    [-2.0, 0.0, 0.0, 0.0],
                                    [0.0, -2.0, 0.0, 0.0]]])

    # Aksi 1: Gaya tepat mempertahankan formasi (0, 0)
    action_stay = torch.FloatTensor([[0.0, 0.0]])
    # Aksi 2: Gaya acak menjauh (0.8, -0.6)
    action_move = torch.FloatTensor([[0.8, -0.6]])

    q1 = critic(ego, neighbors, action_stay)
    q2 = critic(ego, neighbors, action_move)

    print(f"1. Input Ego State        : {list(ego.shape)} -> [x, y, v, yaw]")
    print(f"2. Input Neighbor States  : {list(neighbors.shape)} -> 4 tetangga x 4 fitur")
    print(f"3. Input Action (Fx, Fy)  : {list(action_stay.shape)} -> Gaya dorong/tarik")
    print(f"4. Output Skor Q (Aksi 1) : {q1.item():+.4f} (Evaluasi keputusan diam)")
    print(f"5. Output Skor Q (Aksi 2) : {q2.item():+.4f} (Evaluasi keputusan bergerak)")

    print("\n" + "-" * 80)
    print(" PENGUJIAN ATURAN NILAI (REWARD FUNCTION)")
    print("-" * 80)

    # Kasus A: Formasi Sempurna (Jarak tepat 2.0 meter ke semua tetangga)
    ego_pt_perfect = [0.0, 0.0]
    nbrs_pt_perfect = [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]]
    r_perfect = compute_formation_reward(ego_pt_perfect, nbrs_pt_perfect, target_distance=2.0)
    print(f"Kasus A (Formasi Tepat 2.0 m)   -> Reward: {r_perfect:+.3f} (Maksimal: 0.000)")

    # Kasus B: Formasi Rapat (Jarak rata-rata meleset 0.5 meter, misal jarak 1.5 m)
    nbrs_pt_close = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]
    r_close = compute_formation_reward(ego_pt_perfect, nbrs_pt_close, target_distance=2.0)
    print(f"Kasus B (Meleset Rata-rata 0.5 m)-> Reward: {r_close:+.3f} (Penalti sesuai selisih)")

    # Kasus C: Formasi Tersebar Jauh (Jarak rata-rata meleset 1.5 meter, misal jarak 3.5 m)
    nbrs_pt_far = [[3.5, 0.0], [0.0, 3.5], [-3.5, 0.0], [0.0, -3.5]]
    r_far = compute_formation_reward(ego_pt_perfect, nbrs_pt_far, target_distance=2.0)
    print(f"Kasus C (Meleset Rata-rata 1.5 m)-> Reward: {r_far:+.3f} (Penalti lebih besar)")

    print("=" * 80)
    print(">> KESIMPULAN: CRITIC & ATURAN NILAI REWARD SIAP 100% UNTUK PELATIHAN!")


if __name__ == '__main__':
    demo_critic_and_reward()
