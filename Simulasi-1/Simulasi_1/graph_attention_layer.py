#!/usr/bin/env python3
"""
Modul PyTorch: GraphAttentionLayer ("Mata Pintar" Swarm AI)
Implementasi Self-Attention (nn.MultiheadAttention) untuk koordinasi multi-robot terdesentralisasi:

1. Gabungkan data:
   Posisi diri sendiri (ego_state: [x, y, kecepatan, arah_hadap]) dan posisi semua tetangga
   (neighbor_states: list/dict dari [x, y, kecepatan, arah_hadap]) digabung menjadi satu sequence.

2. Beri bobot perhatian (Self-Attention):
   MultiheadAttention otomatis menilai tetangga mana yang paling penting saat ini
   (misalnya tetangga yang terlalu dekat berisiko tabrakan, atau terlalu jauh kehilangan formasi).

3. Rangkum hasil:
   Semua informasi dirata-rata (torch.mean) menjadi satu vektor ringkasan (summary vector)
   berdimensi tetap yang siap diolah oleh neural network berikutnya.
   Bersifat invarian ukuran: kompatibel dengan jumlah tetangga berapa pun (2, 4, 10, dst).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """
    Lapisan Graph Attention ("Mata Pintar") berbasis MultiheadAttention PyTorch.
    Mampu memproses jumlah tetangga dinamis (invarian skala) tanpa mengubah arsitektur.
    """

    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        """
        Inisialisasi GraphAttentionLayer.

        Parameters:
            input_dim (int): Dimensi fitur per robot (default: 4 -> [x, y, kecepatan, arah_hadap])
            embed_dim (int): Dimensi representasi tersembunyi / embedding (default: 64, harus habis dibagi num_heads)
            num_heads (int): Jumlah attention heads (default: 4)
            dropout (float): Dropout probability (default: 0.0)
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) harus habis dibagi num_heads ({num_heads})")

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Proyeksi fitur input ke dimensi embedding
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalisasi dan aktivasi
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def create_sequence_tensor(
        self,
        ego_state: Union[List[float], torch.Tensor],
        neighbor_states: Union[Dict[int, List[float]], List[List[float]], torch.Tensor],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Langkah 1: Gabungkan data diri (ego_state) dan semua tetangga (neighbor_states)
        menjadi satu tensor urutan (sequence) berdimensi [batch_size, seq_len, input_dim].
        Mendukung pemrosesan single-agent (batch_size=1) maupun mini-batch (batch_size=B).
        """
        if device is None:
            device = next(self.parameters()).device

        # Kasus A: ego_state berupa PyTorch Tensor (batch support)
        if isinstance(ego_state, torch.Tensor):
            ego_t = ego_state.to(device=device, dtype=torch.float32)
            if ego_t.ndim == 1:
                ego_t = ego_t.unsqueeze(0)  # [1, input_dim]
            # ego_t: [batch_size, input_dim]
            ego_seq = ego_t.unsqueeze(1)    # [batch_size, 1, input_dim]

            if isinstance(neighbor_states, torch.Tensor):
                nbr_t = neighbor_states.to(device=device, dtype=torch.float32)
                if nbr_t.ndim == 2:
                    nbr_t = nbr_t.unsqueeze(0)  # [1, num_neighbors, input_dim]
                # nbr_t: [batch_size, num_neighbors, input_dim]
                if nbr_t.shape[1] > 0:
                    return torch.cat([ego_seq, nbr_t], dim=1)
                return ego_seq
            elif isinstance(neighbor_states, (dict, list)):
                nbr_list = [list(val) for val in neighbor_states.values()] if isinstance(neighbor_states, dict) else [list(n) for n in neighbor_states]
                if len(nbr_list) > 0:
                    nbr_t = torch.tensor(nbr_list, dtype=torch.float32, device=device).unsqueeze(0)
                    if ego_seq.shape[0] > 1:
                        nbr_t = nbr_t.repeat(ego_seq.shape[0], 1, 1)
                    return torch.cat([ego_seq, nbr_t], dim=1)
                return ego_seq

        # Kasus B: ego_state berupa list (single sample)
        ego_list = list(ego_state)

        # Kumpulkan semua neighbor_states
        if isinstance(neighbor_states, dict):
            neighbors_list = [list(val) for val in neighbor_states.values()]
        elif isinstance(neighbor_states, torch.Tensor):
            neighbors_list = neighbor_states.detach().cpu().tolist()
            if len(neighbors_list) > 0 and isinstance(neighbors_list[0][0], list):
                neighbors_list = neighbors_list[0]
        else:
            neighbors_list = [list(n) for n in neighbor_states]

        # Gabungkan menjadi satu sequence: [ego, neighbor_1, neighbor_2, ...]
        full_seq = [ego_list] + neighbors_list
        tensor_seq = torch.tensor(full_seq, dtype=torch.float32, device=device)
        # Bentuk batch: [1, seq_len, input_dim]
        return tensor_seq.unsqueeze(0)

    def forward(
        self,
        ego_state: Optional[Union[List[float], torch.Tensor]] = None,
        neighbor_states: Optional[Union[Dict[int, List[float]], List[List[float]], torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward Pass "Mata Pintar":
        
        Parameters:
            ego_state: State diri sendiri [x, y, kecepatan, arah_hadap]
            neighbor_states: State tetangga-tetangga (dict atau list)
            x (Tensor opsional): Jika sudah dalam bentuk Tensor [batch_size, seq_len, input_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - summary_vector: Vektor ringkasan gabungan [batch_size, embed_dim]
                - attn_weights: Matriks bobot perhatian [batch_size, seq_len, seq_len]
        """
        # 1. Gabungkan data menjadi sequence jika diinput sebagai state terpisah
        if x is None:
            if ego_state is None:
                raise ValueError("Harap sediakan ego_state dan neighbor_states atau tensor x.")
            nbr_states = neighbor_states if neighbor_states is not None else {}
            device = next(self.parameters()).device
            x = self.create_sequence_tensor(ego_state, nbr_states, device=device)

        # Proyeksi ke ruang embedding: [batch, seq_len, embed_dim]
        embeds = self.activation(self.embedding(x))

        # 2. Beri bobot perhatian dengan MultiheadAttention:
        # MultiheadAttention menilai tetangga mana yang paling penting saat ini
        attn_out, attn_weights = self.self_attn(
            query=embeds,
            key=embeds,
            value=embeds,
            need_weights=True
        )

        # Residual connection + LayerNorm
        norm_out = self.layer_norm(embeds + attn_out)

        # 3. Rangkum hasil:
        # Semua informasi dirata-rata (torch.mean) sepanjang dimensi sequence (dim=1)
        # Menghasilkan vektor ringkasan berdimensi tetap [batch_size, embed_dim]
        summary_vector = torch.mean(norm_out, dim=1)

        return summary_vector, attn_weights


def main():
    """Uji mandiri GraphAttentionLayer dengan variasi jumlah tetangga."""
    print("=" * 65)
    print(" UJI MANDIRI: GraphAttentionLayer ('Mata Pintar' Swarm AI)")
    print("=" * 65)

    model = GraphAttentionLayer(input_dim=4, embed_dim=64, num_heads=4)
    model.eval()

    # Diri sendiri (Agent 0)
    ego = [3.0, 0.0, 0.0, 0.0]

    # Kasus 1: Kawanan Normal (5 robot total -> 1 ego + 4 tetangga)
    nbrs_4 = {
        1: [0.0, 2.5, 0.0, 0.0],
        2: [0.0, 0.0, 0.0, 0.0],
        3: [0.0, -2.5, 0.0, 0.0],
        4: [-3.0, 0.0, 0.0, 0.0],
    }
    with torch.no_grad():
        summary_4, weights_4 = model(ego_state=ego, neighbor_states=nbrs_4)
    print(f"\n[Kasus 1] Normal: 4 Tetangga (5 Robot Total)")
    print(f"  -> Input Sequence Size : [1, {1 + len(nbrs_4)}, 4]")
    print(f"  -> Output Summary Size : {list(summary_4.shape)} (Dimensi tetap: 64)")
    print(f"  -> Attention Matrix    : {list(weights_4.shape)} (5x5 attention)")

    # Kasus 2: 1 Robot Mati (sisa 3 tetangga)
    nbrs_3 = {
        1: [0.0, 2.5, 0.0, 0.0],
        2: [0.0, 0.0, 0.0, 0.0],
        3: [0.0, -2.5, 0.0, 0.0],
    }
    with torch.no_grad():
        summary_3, weights_3 = model(ego_state=ego, neighbor_states=nbrs_3)
    print(f"\n[Kasus 2] 1 Robot Mati: 3 Tetangga (4 Robot Total)")
    print(f"  -> Input Sequence Size : [1, {1 + len(nbrs_3)}, 4]")
    print(f"  -> Output Summary Size : {list(summary_3.shape)} (Tetap berjalan mulus!)")

    # Kasus 3: Kawanan Besar (10 tetangga)
    nbrs_10 = {i: [float(i), float(-i), 0.5, 0.1] for i in range(1, 11)}
    with torch.no_grad():
        summary_10, weights_10 = model(ego_state=ego, neighbor_states=nbrs_10)
    print(f"\n[Kasus 3] Skala Besar: 10 Tetangga (11 Robot Total)")
    print(f"  -> Input Sequence Size : [1, {1 + len(nbrs_10)}, 4]")
    print(f"  -> Output Summary Size : {list(summary_10.shape)} (Invarian skala terbukti!)")

    print("\n" + "=" * 65)
    print(" HASIL: GraphAttentionLayer 100% SUKSES MEMPROSES DYNAMIC INPUT!")
    print("=" * 65)


if __name__ == '__main__':
    main()
