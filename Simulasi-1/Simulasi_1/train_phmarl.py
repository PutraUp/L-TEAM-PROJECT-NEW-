#!/usr/bin/env python3
"""
Modul: train_phmarl.py (Alur Siklus Latihan / Training Loop SAC untuk Swarm Robotics)
Implementasi Fase 5: 4. Alur Siklus Latihan (Training Loop SAC)

Siklus Kerja Algoritma SAC (Soft Actor-Critic) dengan pH-MARL:
1. Eksplorasi (Coba-coba):
   - Robot bergerak menggunakan IDAPBC_Actor dengan penambahan unsur entropi (keacakan/rasa ingin tahu)
   - Kebijakan Squashed Gaussian (via tanh) memastikan gaya dorong (Fx, Fy) selalu berada dalam batas fisik.
   - Mencegah robot terjebak diam di tempat (local minima) dan mendorong pencarian formasi optimal.

2. Catat:
   - Setiap langkah simulasi dicatat ke NetworkedReplayBuffer:
     (posisi_sekarang, gaya_yang_dipilih, reward, posisi_berikutnya, done, struktur_graf).

3. Evaluasi & Koreksi (Backpropagation):
   - Secara berkala mengambil sekumpulan sampel acak (mini-batch) dari Replay Buffer.
   - Twin Critics (Q1, Q2) dilatih dengan soft target update (Polyak averaging tau = 0.005)
     agar tebakan nilai Q-nya terhadap reward masa depan semakin akurat.
   - Actor (IDAPBC_Actor) memperbarui bobot energi potensial H, matriks interkoneksi J (skew-symmetric),
     dan matriks redaman R (positif) untuk menghasilkan gaya yang mendapat skor setinggi mungkin dari Critic.

4. Pantau Grafik:
   - Mencatat rata-rata reward dan rata-rata formation error setiap 10 episode.
   - Menyimpan grafik perkembangan latihan (formation_training_curve.png).
   - Menyimpan bobot terbaik (best_actor.pth) ketika error jarak antar-robot mendekati 0 meter.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# Tambahkan direktori root package ke sys.path agar dapat dijalankan langsung
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif untuk server headless / Linux
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Simulasi_1.idapbc_actor import IDAPBC_Actor
from Simulasi_1.swarm_critic import SwarmCritic, compute_formation_reward
from Simulasi_1.networked_replay_buffer import NetworkedReplayBuffer
from Simulasi_1.swarm_training_env import SwarmTrainingEnv


# =============================================================================
# 1. SAC Policy Wrapper (Eksplorasi Entropi + IDAPBC_Actor)
# =============================================================================
class SACPolicy(nn.Module):
    """
    Pembungkus Kebijakan SAC berbasis IDAPBC_Actor:
    Menggabungkan gaya deterministik hukum fisika Port-Hamiltonian dengan eksplorasi
    stokastik Gaussian ber-entropi (Squashed Gaussian via tanh).
    """

    def __init__(
        self,
        state_dim: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 32,
        max_action: float = 1.0,
        force_scale: float = 10.0,
        log_std_min: float = -5.0,
        log_std_max: float = 0.5,
    ):
        super().__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Otak Fisika Deterministik IDAPBC dengan pengali gaya agar responsif
        self.actor = IDAPBC_Actor(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            force_scale=force_scale,
        )

        # Parameter eksplorasi log_std yang dapat dipelajari (atau disetel adaptif)
        self.log_std = nn.Parameter(torch.full((2,), -0.5, dtype=torch.float32))

    def sample(
        self,
        ego_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Mengambil sampel aksi gaya dorong (Fx, Fy):
        - Saat deterministic=True: aksi murni berbasis fisika port-Hamiltonian tanpa noise.
        - Saat deterministic=False: menambahkan rasa ingin tahu (entropi) SAC via reparameterization trick.
        """
        # Mu diperoleh langsung dari turunan energi potensial dan interkoneksi IDAPBC
        # create_graph=True agar gradien loss actor dapat merambat balik ke bobot H, J, R
        mu, info = self.actor(ego_state=ego_state, neighbor_states=neighbor_states, create_graph=True)

        if deterministic:
            action = torch.tanh(mu) * self.max_action
            return action, None, info

        # Reparameterization trick: u = mu + sigma * eps, eps ~ N(0, I)
        log_std_clamped = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std_clamped)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()

        # Squashed Gaussian via tanh
        action = torch.tanh(u) * self.max_action

        log_prob = None
        if with_logprob:
            # Koreksi Jacobian untuk transformasi fungsi nonlinear tanh
            # log pi(a|s) = log N(u; mu, sigma) - sum(log(1 - tanh(u)^2 + 1e-6))
            log_prob_raw = dist.log_prob(u)
            log_prob_correction = torch.log(self.max_action * (1.0 - torch.tanh(u).pow(2)) + 1e-6)
            log_prob = (log_prob_raw - log_prob_correction).sum(dim=-1, keepdim=True)

        return action, log_prob, info


# =============================================================================
# 2. SACTrainer (Twin Critics, Actor Loss, Backpropagation, & Target Update)
# =============================================================================
class SACTrainer:
    """
    Pengelola Pelatihan Algoritma SAC (Soft Actor-Critic) untuk pH-MARL:
    - Melatih Twin Critics (Q1, Q2) untuk mengatasi overestimasi nilai Q.
    - Melatih IDAPBC_Actor untuk memaksimalkan Q-value + bonus entropi.
    - Melakukan soft target updates secara berkala dengan Polyak averaging.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = False,
        target_entropy: Optional[float] = None,
        max_action: float = 1.0,
        force_scale: float = 10.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.max_action = max_action
        self.force_scale = force_scale

        # 1. Model Kebijakan Actor (IDAPBC_Actor + Entropi + Force Scaling)
        self.policy = SACPolicy(
            state_dim=state_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=32,
            max_action=max_action,
            force_scale=force_scale,
        ).to(self.device)

        # 2. Twin Critics (Q1 & Q2)
        self.critic1 = SwarmCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic2 = SwarmCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # 3. Target Critics (Q1_targ & Q2_targ)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # 4. Optimizers
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # 5. Pengaturan Entropi Otomatis (Auto-tuning Alpha)
        if self.auto_alpha:
            if target_entropy is None:
                self.target_entropy = -float(action_dim)  # -2.0 untuk 2D actions
            else:
                self.target_entropy = float(target_entropy)

            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = float(self.log_alpha.exp().item())

    def select_action(
        self,
        ego_state: Union[np.ndarray, List[float]],
        neighbor_states: Union[np.ndarray, List[List[float]]],
        deterministic: bool = False
    ) -> np.ndarray:
        """Memilih aksi kontrol (Fx, Fy) dari observasi saat ini."""
        self.policy.eval()
        t_ego = torch.tensor(ego_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        t_nbr = torch.tensor(neighbor_states, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = self.policy.sample(t_ego, t_nbr, deterministic=deterministic, with_logprob=False)

        return action.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Langkah 3: Evaluasi & Koreksi (Backpropagation).
        Menghitung loss critic dan actor, lalu memperbarui bobot neural network.
        """
        self.policy.train()
        self.critic1.train()
        self.critic2.train()

        ego = batch["ego_states"].to(self.device)                 # [B, 4]
        nbr = batch["neighbor_states"].to(self.device)           # [B, N, 4]
        actions = batch["actions"].to(self.device)               # [B, 2]
        rewards = batch["rewards"].to(self.device)               # [B, 1]
        next_ego = batch["next_ego_states"].to(self.device)       # [B, 4]
        next_nbr = batch["next_neighbor_states"].to(self.device) # [B, N, 4]
        dones = batch["dones"].to(self.device)                   # [B, 1]

        # ---------------------------------------------------------------------
        # 1. Update Critics (Juri Belajar Tebakan Reward)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # Aksi masa depan s' dari kebijakan saat ini
            next_actions, next_log_prob, _ = self.policy.sample(next_ego, next_nbr, deterministic=False)
            q1_target = self.target_critic1(next_ego, next_nbr, next_actions)
            q2_target = self.target_critic2(next_ego, next_nbr, next_actions)
            q_target_min = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * q_target_min

        # Evaluasi nilai Q saat ini
        q1 = self.critic1(ego, nbr, actions)
        q2 = self.critic2(ego, nbr, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        # ---------------------------------------------------------------------
        # 2. Update Actor (IDAPBC_Actor Memperbaiki Matriks H, J, dan R)
        # ---------------------------------------------------------------------
        # Sampel aksi baru dari state s dengan gradien tersambung ke actor
        new_actions, log_prob, _ = self.policy.sample(ego, nbr, deterministic=False)
        q1_new = self.critic1(ego, nbr, new_actions)
        q2_new = self.critic2(ego, nbr, new_actions)
        q_new_min = torch.min(q1_new, q2_new)

        # Actor Loss: Memaksimalkan Q_value dengan penalti entropi -> minimize (alpha * log_prob - Q)
        actor_loss = (self.alpha * log_prob - q_new_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------------------------------------------------------------
        # 3. Update Alpha (Jika Auto-tuning Entropi Aktif)
        # ---------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = float(self.log_alpha.exp().item())
            alpha_loss_val = float(alpha_loss.item())

        # ---------------------------------------------------------------------
        # 4. Soft Target Update untuk Critics (Polyak Averaging)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for p, p_targ in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)
            for p, p_targ in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": float(self.alpha),
            "q1_mean": float(q1.mean().item()),
            "q_target_mean": float(target_q.mean().item()),
        }

    def save_checkpoint(self, path_actor: str, path_critic: Optional[str] = None):
        """Menyimpan bobot IDAPBC_Actor dan Critic."""
        os.makedirs(os.path.dirname(os.path.abspath(path_actor)), exist_ok=True)
        # Simpan state_dict IDAPBC_Actor langsung agar kompatibel dengan DecentralizedAgentNode
        torch.save(self.policy.actor.state_dict(), path_actor)
        if path_critic is not None:
            torch.save({
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "policy_full": self.policy.state_dict(),
                "alpha": self.alpha,
            }, path_critic)

    def load_checkpoint(self, path_actor: str, path_critic: Optional[str] = None):
        """Memuat bobot IDAPBC_Actor."""
        if os.path.exists(path_actor):
            self.policy.actor.load_state_dict(torch.load(path_actor, map_location=self.device))
            print(f"[*] Berhasil memuat bobot Actor dari: {path_actor}")
        if path_critic and os.path.exists(path_critic):
            data = torch.load(path_critic, map_location=self.device)
            self.critic1.load_state_dict(data["critic1"])
            self.critic2.load_state_dict(data["critic2"])
            self.target_critic1.load_state_dict(data["critic1"])
            self.target_critic2.load_state_dict(data["critic2"])
            if "alpha" in data:
                self.alpha = data["alpha"]
            print(f"[*] Berhasil memuat bobot Critic dari: {path_critic}")


# =============================================================================
# 3. Fungsi Plot Grafik Pelatihan (Pantau Grafik)
# =============================================================================
def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    target_distance: float = 2.0
):
    """
    Langkah 4: Pantau Grafik
    Menyimpan kurva perkembangan pelatihan:
    - Kurva Formation Error vs Episode (menuju 0 meter).
    - Kurva Average Reward per Episode.
    - Kurva Actor & Critic Loss.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    episodes = history.get("episodes", [])
    if len(episodes) == 0:
        return

    plt.figure(figsize=(14, 10))

    # Subplot 1: Formation Error (Target: 0.0 meter)
    plt.subplot(2, 2, 1)
    errors = history.get("formation_errors", [])
    final_errors = history.get("final_formation_errors", [])
    plt.plot(episodes, errors, color="crimson", alpha=0.5, linewidth=1.5, label="Mean Error (Sepanjang Episode)")
    if len(final_errors) == len(episodes):
        plt.plot(episodes, final_errors, color="royalblue", linewidth=2.0, label="Final Error (Akhir Episode)")
        if len(final_errors) > 5:
            window = min(5, len(final_errors))
            ma = np.convolve(final_errors, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], ma, color="navy", linestyle="--", linewidth=2, label=f"Moving Avg Final (w={window})")
    elif len(errors) > 5:
        window = min(5, len(errors))
        ma = np.convolve(errors, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], ma, color="darkred", linestyle=":", linewidth=2, label=f"Moving Avg (w={window})")

    plt.axhline(0.0, color="green", linestyle="--", alpha=0.8, label="Target Formasi Sempurna (0.0 m)")
    plt.title("Perkembangan Kesalahan Formasi (Mean & Final Error)", fontsize=12, fontweight="bold")
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Error Jarak (meter)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 2: Average Reward
    plt.subplot(2, 2, 2)
    rewards = history.get("rewards", [])
    plt.plot(episodes, rewards, color="royalblue", linewidth=2, label="Episode Mean Reward")
    plt.axhline(0.0, color="green", linestyle="--", alpha=0.8, label="Reward Maksimum (0.0)")
    plt.title("Perkembangan Reward Formasi (Semakin Mendekati 0)", fontsize=12, fontweight="bold")
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Reward Rata-rata", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 3: Critic Loss & Actor Loss
    plt.subplot(2, 2, 3)
    c_losses = history.get("critic_losses", [])
    a_losses = history.get("actor_losses", [])
    if len(c_losses) > 0:
        plt.plot(episodes[:len(c_losses)], c_losses, color="orange", linewidth=1.8, label="Critic Loss")
    if len(a_losses) > 0:
        plt.plot(episodes[:len(a_losses)], a_losses, color="purple", linewidth=1.8, label="Actor Loss")
    plt.title("Loss Pelatihan Neural Network (Actor & Critic)", fontsize=12, fontweight="bold")
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Loss Value", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 4: Nilai Rata-rata Q (Q-Value Prediksi)
    plt.subplot(2, 2, 4)
    q_vals = history.get("q_values", [])
    if len(q_vals) > 0:
        plt.plot(episodes[:len(q_vals)], q_vals, color="teal", linewidth=2, label="Mean Q-Value")
    plt.title("Evolusi Estimasi Skor Q (Critic Confidence)", fontsize=12, fontweight="bold")
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Q-Value", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# 4. Alur Siklus Latihan Utama (Training Loop SAC)
# =============================================================================
def train(
    num_episodes: int = 200,
    steps_per_episode: int = 500,
    batch_size: int = 64,
    warmup_steps: int = 500,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    alpha: float = 0.2,
    auto_alpha: bool = False,
    force_scale: float = 10.0,
    curriculum: bool = True,
    curriculum_episodes: int = 60,
    target_distance: float = 2.0,
    buffer_capacity: int = 100000,
    save_dir: str = "src/Simulasi-1/checkpoints",
    plot_interval: int = 10,
    resume: bool = False,
    device: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Menjalankan Siklus Penuh Pelatihan SAC pH-MARL:
    1. Eksplorasi (dengan Force Scaling & Entropi)
    2. Pencatatan ke Networked Replay Buffer
    3. Evaluasi & Koreksi Backpropagation
    4. Pemantauan Grafik & Kurikulum Adaptif (Curriculum Learning)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 95)
    print(f" MEMULAI SIKLUS PELATIHAN SAC pH-MARL (Fase 5 - Tahap 4)")
    print(f" Device              : {device.upper()}")
    print(f" Mode                : {'MELANJUTKAN LATIHAN (--resume)' if resume else 'LATIHAN DARI AWAL (FRESH)'}")
    print(f" Force Scale Gain    : {force_scale}x (Robot lincah & responsif)")
    print(f" Curriculum Learning : {curriculum} (Pemanasan radius formasi bertahap)")
    print(f" Tambah Episode      : {num_episodes}")
    print(f" Langkah/Episode     : {steps_per_episode} langkah (10 detik @ 50 Hz)")
    print(f" Target Jarak        : {target_distance} meter")
    print(f" Batch Size          : {batch_size}")
    print(f" Direktori Model     : {save_dir}")
    print("=" * 95)

    os.makedirs(save_dir, exist_ok=True)
    best_actor_path = os.path.join(save_dir, "best_actor.pth")
    best_critic_path = os.path.join(save_dir, "best_critic.pth")
    latest_actor_path = os.path.join(save_dir, "latest_actor.pth")
    plot_path = os.path.join(save_dir, "formation_training_curve.png")
    history_path = os.path.join(save_dir, "training_history.json")

    # Inisialisasi Lapangan Latihan & Buku Catatan Pengalaman
    env = SwarmTrainingEnv(
        num_agents=5,
        dt=0.02,
        episode_duration=steps_per_episode * 0.02,
        target_distance=target_distance,
    )
    buffer = NetworkedReplayBuffer(capacity=buffer_capacity, comm_range=6.0)

    # Inisialisasi Trainer SAC dengan Force Scaling
    trainer = SACTrainer(
        state_dim=4,
        action_dim=2,
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        alpha=alpha,
        auto_alpha=auto_alpha,
        force_scale=force_scale,
        device=device,
    )

    history: Dict[str, List[float]] = {
        "episodes": [],
        "formation_errors": [],
        "final_formation_errors": [],
        "rewards": [],
        "critic_losses": [],
        "actor_losses": [],
        "q_values": [],
    }

    start_episode = 1
    best_formation_error = float("inf")
    total_env_steps = 0

    # Jika resume aktif dan ada bobot sebelumnya: muat bobot & riwayat lama
    if resume and os.path.exists(best_actor_path):
        trainer.load_checkpoint(best_actor_path, best_critic_path)
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    old_hist = json.load(f)
                    for k in history:
                        if k in old_hist:
                            history[k] = list(old_hist[k])
                if len(history["episodes"]) > 0:
                    start_episode = history["episodes"][-1] + 1
                    if "final_formation_errors" in old_hist and len(old_hist["final_formation_errors"]) > 0:
                        best_formation_error = min(old_hist["final_formation_errors"])
                    elif len(history["formation_errors"]) > 0:
                        best_formation_error = min(history["formation_errors"])
                    print(f"[*] Melanjutkan dari Episode {start_episode} (Best Error Sebelumnya: {best_formation_error:.4f} m)")
            except Exception as e:
                print(f"[!] Gagal membaca riwayat lama: {e}")
        warmup_steps = 0  # Tidak butuh warmup acak jika melanjutkan model yang sudah pintar

    start_train_time = time.time()

    print(f"\n{'Episode':^10} | {'Total Step':^10} | {'Avg Reward':^12} | {'Final / Avg Err':^19} | {'Critic Loss':^12} | {'Actor Loss':^12} | {'Durasi':^8}")
    print("-" * 105)

    end_episode = start_episode + num_episodes - 1
    for episode in range(start_episode, end_episode + 1):
        ep_start_time = time.time()
        # 1. Reset Lapangan Latihan dengan posisi acak (Rejection Sampling & Kurikulum)
        curr_factor = min(1.0, (episode - 1) / max(1, curriculum_episodes)) if curriculum else 1.0
        obs, _ = env.reset(randomize=True, curriculum_factor=curr_factor)

        ep_reward_accum = 0.0
        ep_error_accum = 0.0
        last_50_errors = []
        ep_c_loss = []
        ep_a_loss = []
        ep_q_val = []

        for step in range(steps_per_episode):
            total_env_steps += 1
            actions = {}

            # 1. Eksplorasi (Coba-coba): Setiap robot menentukan gaya dengan bantuan Actor + Entropi
            for i in range(env.num_agents):
                ego = obs[i]["ego"]
                nbrs = obs[i]["neighbors"]

                if total_env_steps < warmup_steps:
                    # Warmup awal: eksplorasi acak murni agar buffer terisi keragaman awal
                    act = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
                else:
                    # Eksplorasi SAC: IDAPBC_Actor + noise keacakan entropi
                    act = trainer.select_action(ego, nbrs, deterministic=False)

                actions[i] = act

            # Langkah simulasi lingkungan
            next_obs, rewards, done, _, info = env.step(actions)

            # 2. Catat ke Networked Replay Buffer lengkap dengan graf keterhubungan
            buffer.push_swarm_step(obs, actions, rewards, next_obs, done=done)

            obs = next_obs
            ep_reward_accum += sum(rewards.values()) / env.num_agents
            err = info.get("mean_formation_error", info.get("mean_dist_error", 0.0))
            ep_error_accum += err
            if step >= steps_per_episode - 50:
                last_50_errors.append(err)

            # 3. Evaluasi & Koreksi (Backpropagation)
            if buffer.is_ready(batch_size) and total_env_steps >= warmup_steps:
                batch = buffer.sample(batch_size=batch_size, device=device)
                metrics = trainer.update(batch)
                ep_c_loss.append(metrics["critic_loss"])
                ep_a_loss.append(metrics["actor_loss"])
                ep_q_val.append(metrics["q1_mean"])

            if done:
                break

        # Rata-rata metrik episode ini
        avg_reward = ep_reward_accum / steps_per_episode
        avg_error = ep_error_accum / steps_per_episode
        final_error = float(np.mean(last_50_errors)) if len(last_50_errors) > 0 else avg_error
        mean_c_loss = float(np.mean(ep_c_loss)) if len(ep_c_loss) > 0 else 0.0
        mean_a_loss = float(np.mean(ep_a_loss)) if len(ep_a_loss) > 0 else 0.0
        mean_q = float(np.mean(ep_q_val)) if len(ep_q_val) > 0 else 0.0
        ep_duration = time.time() - ep_start_time

        history["episodes"].append(episode)
        history["rewards"].append(avg_reward)
        history["formation_errors"].append(avg_error)
        history["final_formation_errors"].append(final_error)
        history["critic_losses"].append(mean_c_loss)
        history["actor_losses"].append(mean_a_loss)
        history["q_values"].append(mean_q)

        # Simpan checkpoint model terbaik (berdasarkan final_error konvergen di akhir episode)
        is_best = False
        if final_error < best_formation_error:
            best_formation_error = final_error
            trainer.save_checkpoint(best_actor_path, best_critic_path)
            is_best = True

        # Selalu simpan checkpoint terbaru
        trainer.save_checkpoint(latest_actor_path)

        # 4. Pantau Grafik & Tampilkan Rata-rata setiap plot_interval episode
        if episode % plot_interval == 0 or episode == end_episode:
            best_flag = " [*BEST]" if is_best else ""
            err_str = f"{final_error:.3f} / {avg_error:.3f} m"
            print(f"{episode:^10} | {total_env_steps:^10} | {avg_reward:^12.4f} | {err_str:^19}{best_flag:<4} | {mean_c_loss:^12.4f} | {mean_a_loss:^12.4f} | {ep_duration:^6.2f}s")
            
            # Perbarui grafik kurva latihan
            plot_training_curves(history, plot_path, target_distance=target_distance)

            # Simpan riwayat numerik JSON
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

    total_duration = time.time() - start_train_time
    print("=" * 105)
    print(f" PELATIHAN SELESAI DALAM {total_duration:.1f} DETIK!")
    print(f" -> Best Final Formation Error : {best_formation_error:.4f} meter")
    print(f" -> Bobot Actor Terbaik        : {best_actor_path}")
    print(f" -> Grafik Kurva Latihan       : {plot_path}")
    print("=" * 105)

    return history

    return history


def main():
    parser = argparse.ArgumentParser(description="Alur Siklus Latihan (Training Loop SAC) untuk Swarm Robotics pH-MARL")
    parser.add_argument("--episodes", type=int, default=100, help="Jumlah total episode latihan (default: 100)")
    parser.add_argument("--steps", type=int, default=500, help="Jumlah langkah simulasi per episode (default: 500 = 10 detik)")
    parser.add_argument("--batch-size", type=int, default=64, help="Ukuran mini-batch sampling Replay Buffer (default: 64)")
    parser.add_argument("--warmup", type=int, default=500, help="Langkah awal eksplorasi acak sebelum backprop (default: 500)")
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Learning rate IDAPBC_Actor (default: 3e-4)")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Learning rate SwarmCritic (default: 1e-3)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Koefisien entropi rasa ingin tahu SAC (default: 0.2)")
    parser.add_argument("--auto-alpha", action="store_true", help="Otomatis menyetel koefisien alpha SAC")
    parser.add_argument("--target-dist", type=float, default=2.0, help="Target jarak formasi antar-robot (default: 2.0 m)")
    parser.add_argument("--force-scale", type=float, default=10.0, help="Pengali gaya responsif robot (default: 10.0)")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Aktifkan kurikulum pemanasan radius formasi bertahap (default: True)")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false", help="Nonaktifkan kurikulum bertahap")
    parser.add_argument("--curriculum-episodes", type=int, default=60, help="Jumlah episode awal kurikulum (default: 60)")
    parser.add_argument("--save-dir", type=str, default="src/Simulasi-1/checkpoints", help="Folder penyimpanan model & plot")
    parser.add_argument("--plot-interval", type=int, default=10, help="Interval perbaruan grafik dan log terminal (default: 10 episode)")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan latihan dari bobot best_actor.pth sebelumnya")
    parser.add_argument("--device", type=str, default=None, help="Device komputasi: 'cpu' atau 'cuda' (default: otomatis)")

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        force_scale=args.force_scale,
        curriculum=args.curriculum,
        curriculum_episodes=args.curriculum_episodes,
        target_distance=args.target_dist,
        save_dir=args.save_dir,
        plot_interval=args.plot_interval,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
