# Simulasi-1: Multi-Agent Swarm Robotics & pH-MARL Controller

Package ROS 2 (Jazzy Jalisco / Ubuntu 24.04 LTS) untuk simulasi kawanan (*swarm*) multi-robot otonom terdesentralisasi menggunakan **Gazebo Sim (GZ Harmonic)** dan pengendali berbasis fisika **pH-MARL (Port-Hamiltonian Multi-Agent Reinforcement Learning)** yang diintegrasikan dengan **Graph Attention Network (GAT)** serta emulasi sensor **Cerdas UWB Tracker (Decawave DWM1000)**.

---

## Daftar Isi
- [Ringkasan Sistem](#ringkasan-sistem)
- [Arsitektur & Alur Kerja](#arsitektur--alur-kerja)
- [Komunikasi Terdesentralisasi](#komunikasi-terdesentralisasi)
- [Algoritma AI: pH-MARL (IDA-PBC + GAT + SAC)](#algoritma-ai-ph-marl-ida-pbc--gat--sac)
- [Lingkungan Latihan Cepat (SwarmTrainingEnv)](#lingkungan-latihan-cepat-swarmtrainingenv)
- [Struktur Direktori & File](#struktur-direktori--file)
- [Prasyarat & Instalasi](#prasyarat--instalasi)
- [Panduan Penggunaan](#panduan-penggunaan)
  - [1. Melatih Model AI (Training)](#1-melatih-model-ai-training)
  - [2. Menjalankan Simulasi di Gazebo](#2-menjalankan-simulasi-di-gazebo)
  - [3. Memantau Topik & Telemetri](#3-memantau-topik--telemetri)
- [Parameter Konfigurasi](#parameter-konfigurasi)

---

## Ringkasan Sistem

Sistem ini mensimulasikan **5 robot beroda bergerak (*differential/bicycle drive*)** yang bergerak secara terkoordinasi untuk membentuk dan mempertahankan **Formasi Belah Ketupat (*Diamond/Rhombus Formation*)** dengan jarak antar-robot target sebesar **2.0 meter**:

```text
                  Robot 1 (Apex / Depan) [ID 0 / Anchor 1001]
                             /\
                            /  \
     Robot 2 (Kiri)        /    \       Robot 4 (Kanan)
  [ID 1 / Anchor 1002]    \      /    [ID 3 / Anchor 1004]
                           \    /
                            \  /
                  Robot 3 (Pusat Acuan) [ID 2 / Anchor 1003]
                             /\
                            /  \
                  Robot 5 (Belakang) [ID 4 / Anchor 1005]
```

### Karakteristik Utama:
1. **Fully Decentralized**: Tidak ada node server pusat (*single point of failure*). Setiap robot mengeksekusi komputasi kontrol dan pengambilan keputusannya sendiri secara mandiri.
2. **Physics-Informed AI (IDA-PBC)**: Pengendali AI dibatasi oleh struktur fisika Port-Hamiltonian dengan disipasi energi, menjamin stabilitas pasif dan kestabilan Lyapunov (robot tidak akan bergerak liar atau berosilasi berlebih).
3. **Infrastructure-Free UWB Ranging**: Setiap robot bertindak ganda sebagai Tag dan Mobile Anchor menggunakan metode *Two-Way Ranging (TWR)* tanpa membutuhkan tiang pemancar statis.
4. **Fast Training Pipeline**: Pelatihan Reinforcement Learning (SAC) dilakukan di simulator kinematika Python murni dengan kecepatan **>5.000x real-time**, sebelum bobot model dieksekusi di Gazebo fisik.

---

## Arsitektur & Alur Kerja

Setiap agen robot mengeksekusi pipeline modular pada frekuensi kontrol **50 Hz**:

```text
+-----------------------------------------------------------------------------------+
|                        DECENTRALIZED AGENT PIPELINE                               |
|                                                                                   |
|  [Odometri Lokal]       [Topik Tetangga (/agent_j/odom)]   [UWB TWR (/links_json)]|
|         │                                │                            │           |
|         ▼                                ▼                            ▼           |
|  ┌─────────────┐                ┌─────────────────┐          ┌─────────────────┐  |
|  │ Ego State   │                │ Neighbor States │          │ Relative Range  │  |
|  │ [x,y,v,θ]   │                │ [x_j,y_j,v_j,θ] │          │ Matrix          │  |
|  └──────┬──────┘                └────────┬────────┘          └────────┬────────┘  |
|         │                                │                            │           |
|         └───────────────────────┬────────┴────────────────────────────┘           |
|                                 ▼                                                 |
|                 ┌───────────────────────────────┐                                 |
|                 │  Graph Attention Layer (GAT)  │                                 |
|                 │  Multi-Head Spatial Attention │                                 |
|                 └───────────────┬───────────────┘                                 |
|                                 ▼                                                 |
|                 ┌───────────────────────────────┐                                 |
|                 │  IDAPBC_Actor (Otak Fisika)   │                                 |
|                 │  F_pH = (J - R) * ∇H          │                                 |
|                 └───────────────┬───────────────┘                                 |
|                                 ▼                                                 |
|                         Gaya Hidung (Fx, Fy)                                      |
|                                 │                                                 |
|                                 ▼                                                 |
|                 ┌───────────────────────────────┐                                 |
|                 │  Bicycle Kinematic Inversion  │                                 |
|                 │  (Fx, Fy, θ) ──> (v, δ/ω)     │                                 |
|                 └───────────────┬───────────────┘                                 |
|                                 ▼                                                 |
|                       /agent_{i}/cmd_vel                                          |
|                     (Motor Roda Gazebo Sim)                                       |
+-----------------------------------------------------------------------------------+
```

---

## Komunikasi Terdesentralisasi

Sistem komunikasi antar-robot dirancang tanpa perantara server (*broker-less peer-to-peer*) melalui ROS 2 DDS dan protokol UWB realistis:

### 1. Lapisan Emulasi UWB (`uwb_tracker_node.py`)
* Mengemulasikan perangkat keras **Decawave DWM1000** berbasis firmware [Cerdas-UWB-Tracker](https://github.com/geraicerdas/Cerdas-UWB-Tracker).
* **Mode Dual (Anchor + Tag)** pada setiap robot.
* Metode pengukuran: **Two-Way Ranging (TWR)** dengan noise Gaussian ($\sigma = 2.5\text{ cm}$) dan filter *moving average* 3-sampel.
* **Format Payload JSON**:
  Dipublikasikan pada topik `/{ros_name}/uwb/links_json` dan `/agent_{i}/uwb/links_json`:
  ```json
  {
    "device_id": "1001",
    "role": "dual",
    "links": [
      {"anchor_id": "1002", "range_m": 2.52, "rssi_dbm": -79.4},
      {"anchor_id": "1003", "range_m": 3.01, "rssi_dbm": -81.2}
    ]
  }
  ```

### 2. Kualitas Layanan Jaringan (QoS Policy)
Topik inter-robot (`/agent_{i}/odom`, `/agent_{i}/status`, `/agent_{i}/uwb/links_json`) dikonfigurasi dengan profil **Live-Streaming**:
* `Reliability = BEST_EFFORT`: Menghindari latensi retransmisi paket usang.
* `History = KEEP_LAST (depth=1)`: Memori hanya menyimpan 1 pesan paling mutakhir.
* `Durability = VOLATILE`: Tidak membaca pesan lampau.

---

## Algoritma AI: pH-MARL (IDA-PBC + GAT + SAC)

Algoritma kecerdasan buatan mengkombinasikan teori kontrol sistem Port-Hamiltonian dengan Deep Reinforcement Learning.

### 1. Formulasi Port-Hamiltonian (IDA-PBC)
Gaya kendali dihitung berdasarkan persamaan:
$$F_{\text{pH}} = (J - R) \nabla H$$

* **$\nabla H = \left[\frac{\partial H}{\partial x}, \frac{\partial H}{\partial y}\right]^T$ (Gradien Energi Potensial)**:
  Dihasilkan dari skalar $H$ melalui *autograd PyTorch*. Berfungsi sebagai pegas virtual multi-arah dan medan tolak anti-tabrakan.
* **$J$ (Matriks Interkoneksi)**:
  Matriks *skew-symmetric* ($J = -J^T$):
  $$J = \begin{bmatrix} 0 & j \\ -j & 0 \end{bmatrix}$$
  Mengatur manuver rotasi dan pertukaran posisi tanpa mendisipasi energi.
* **$R$ (Matriks Redaman)**:
  Matriks diagonal positif ($r_1, r_2 > 0$ melalui aktivasi *Softplus*):
  $$R = \begin{bmatrix} r_1 & 0 \\ 0 & r_2 \end{bmatrix}$$
  Berfungsi sebagai gesekan virtual untuk meredam osilasi ketika formasi tercapai.
* **Stabilitas Lyapunov**:
  Turunan energi terhadap waktu memenuhi:
  $$\dot{H} = (\nabla H)^T (J - R) \nabla H = - (\nabla H)^T R \nabla H \le 0$$
  Karena $R$ definit positif, sistem dijamin pasif dan stabil secara asimtotik.

### 2. Graph Attention Layer (GAT)
* Jangkauan komunikasi efektif: $R_{\text{comm}} \le 6.0\text{ meter}$.
* Menghitung bobot atensi $\alpha_{ij}$ secara adaptif menggunakan mekanisme *Multi-Head Self-Attention* (4 heads, embedding 64).
* Memberikan prioritas perhatian pada tetangga terdekat yang berisiko tabrakan atau tetangga acuan pembentuk formasi belah ketupat.

### 3. Kerangka Kerja Pelatihan: Soft Actor-Critic (SAC)
* **Twin Q-Critics ($Q_1, Q_2$)**: Mencegah *overestimation bias* pada nilai Q masa depan.
* **Eksplorasi Entropi**: Kebijakan *Squashed Gaussian* ($\tanh$) dengan temperatur entropi $\alpha = 0.2$.
* **Reward Function**:
  $$\text{Reward} = - \frac{1}{N} \sum_{i \neq j} \left| d_{ij} - 2.0\text{ m} \right|$$
* **Curriculum Learning**: Pemanasan radius sebar (*spawning*) bertahap dari $1.5\text{ m}$ hingga $8.0\text{ m}$.

---

## Lingkungan Latihan Cepat (`SwarmTrainingEnv`)

Pelatihan Reinforcement Learning memerlukan ratusan ribu langkah simulasi. Simulator Gazebo 3D berjalan $\approx 1\times$ *real-time* (membutuhkan waktu berhari-hari). Oleh karena itu, modul [`swarm_training_env.py`](file:///home/a.putra/l-team_ws/src/Simulasi-1/Simulasi_1/swarm_training_env.py) dirancang sebagai **simulator kinematika murni berbasis Python/NumPy**:

* **Kecepatan Komputasi**: **>5.000x lebih cepat dari Gazebo** (1 episode 500 langkah selesai dalam 1–3 milidetik pada CPU).
* **Model Kinematika**: Kinematika sepeda non-holonomik (*bicycle model*) dengan integrasi Euler $50\text{ Hz}$ ($\Delta t = 0.02\text{ s}$).
* **Transformasi Titik Hidung ($d = 0.25\text{ m}$)**:
  $$z = \begin{bmatrix} x + d \cos\theta \\ y + d \sin\theta \end{bmatrix}$$
  Memungkinkan gaya kendali holonomik $(F_x, F_y)$ diinversikan secara mulus menjadi kecepatan $(v, \omega)$.
* **Rejection Sampling**: Memastikan penempatan acak awal memiliki jarak aman $>0.6\text{ m}$ untuk mencegah tabrakan instan.

---

## Struktur Direktori & File

```text
src/Simulasi-1/
├── checkpoints/                        # Model bobot & hasil evaluasi latihan
│   ├── best_actor.pth                  # Bobot neural network Actor terbaik
│   ├── best_critic.pth                 # Bobot neural network Critic terbaik
│   ├── latest_actor.pth                # Checkpoint episode terakhir
│   ├── training_history.json           # Riwayat loss, reward, dan error formasi
│   └── formation_training_curve.png    # Grafik 4-panel kurva konvergensi latihan
├── launch/                             # Skrip peluncuran ROS 2
│   ├── launch.launch.py                # Launch utama: Gazebo + Spawn Robot + Bridge + Nodes
│   ├── agents.launch.py                # Menjalankan 5 node DecentralizedAgentNode
│   ├── front_end.launch.py             # Menjalankan 5 node transformasi titik hidung
│   └── bicycle_inversion.launch.py     # Menjalankan 5 node inversi kinematika sepeda
├── models/                             # Model SDF dan mesh robot
│   ├── x1_robot-1/ s/d x1_robot-5/     # Deskripsi fisik robot & sensor (LiDAR, DiffDrive)
├── worlds/
│   └── default.sdf                     # Dunia simulasi Gazebo (lantai bergrid & pencahayaan)
├── Simulasi_1/                         # Modul logika Python & Node ROS 2
│   ├── DecentralizedAgentNode.py       # Node utama pengendali agen robot terdesentralisasi
│   ├── uwb_tracker_node.py             # Node emulasi perangkat keras Cerdas UWB Tracker
│   ├── front_end_transformation_node.py# Transformasi titik non-holonomik ke holonomik semu
│   ├── bicycle_kinematic_inversion_node.py # Inversi gaya (Fx, Fy) ke gas & setir (v, omega)
│   ├── graph_attention_layer.py        # Lapisan spasial Graph Attention Network (GAT)
│   ├── idapbc_actor.py                 # Otak fisika Port-Hamiltonian Actor (H, J, R)
│   ├── swarm_critic.py                 # Jaringan penilai Twin Critic (Q1, Q2) & reward
│   ├── networked_replay_buffer.py      # Replay buffer multi-agen dengan topologi graf
│   ├── swarm_training_env.py           # Lingkungan simulasi Python cepat (>5000x)
│   └── train_phmarl.py                 # Skrip siklus pelatihan SAC pH-MARL
├── package.xml                         # Metadata dependensi paket ROS 2
├── setup.py                            # Konfigurasi instalasi Python package & console scripts
└── README.md                           # Dokumentasi lengkap paket
```

---

## Prasyarat & Instalasi

### 1. Dependensi Sistem
* **Ubuntu Linux 24.04 LTS (Noble Numbat)**
* **ROS 2 Jazzy Jalisco**
* **Gazebo Sim (GZ Harmonic)** — Simulator default yang terintegrasi dengan ROS 2 Jazzy
* Paket ROS-GZ Bridge untuk ROS 2 Jazzy:
  ```bash
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-ros-gz ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
  ```

### 2. Dependensi Python
```bash
pip3 install torch numpy matplotlib
# Catatan Ubuntu 24.04: Jika menggunakan managed environment (PEP 668), gunakan flag --break-system-packages atau virtualenv
```

### 3. Kompilasi Workspace
```bash
cd ~/l-team_ws
colcon build --packages-select Simulasi-1
source install/setup.bash
```

---

## Panduan Penggunaan

### 1. Melatih Model AI (Training)

Pelatihan dilakukan di simulator cepat tanpa perlu membuka Gazebo:

* **Memulai Pelatihan Baru (misal 100 episode)**:
  ```bash
  python3 src/Simulasi-1/Simulasi_1/train_phmarl.py --episodes 100
  ```

* **Melanjutkan Pelatihan dari Checkpoint Sebelumnya (`--resume`)**:
  ```bash
  python3 src/Simulasi-1/Simulasi_1/train_phmarl.py --episodes 200 --resume
  ```

* **Opsi Parameter Training**:
  | Argumen | Default | Deskripsi |
  | :--- | :--- | :--- |
  | `--episodes` | `100` | Jumlah episode yang akan dijalankan |
  | `--steps-per-episode` | `500` | Jumlah langkah per episode (500 steps = 10 detik @ 50 Hz) |
  | `--batch-size` | `64` | Ukuran mini-batch sampling dari Replay Buffer |
  | `--lr-actor` | `3e-4` | Learning rate untuk IDAPBC_Actor |
  | `--lr-critic` | `3e-4` | Learning rate untuk SwarmCritic |
  | `--alpha` | `0.2` | Koefisien temperatur entropi SAC |
  | `--resume` | `False` | Memuat bobot dari `checkpoints/best_actor.pth` jika tersedia |
  | `--no-curriculum` | `False` | Menonaktifkan penambahan radius acak bertahap |

Setelah pelatihan selesai, grafik konvergensi otomatis diperbarui di [`checkpoints/formation_training_curve.png`](file:///home/a.putra/l-team_ws/src/Simulasi-1/checkpoints/formation_training_curve.png).

---

### 2. Menjalankan Simulasi di Gazebo

Untuk mengeksekusi kontrol formasi secara langsung pada 5 robot di Gazebo Sim:

* **Mode AI Penuh (Menggunakan Bobot Latihan `best_actor.pth`)**:
  ```bash
  ros2 launch Simulasi-1 launch.launch.py use_ai:=true use_virtual_spring:=true
  ```

* **Mode Baseline Virtual Spring (Tanpa AI)**:
  ```bash
  ros2 launch Simulasi-1 launch.launch.py use_ai:=false use_virtual_spring:=true
  ```

* **Mode Headless (Server Saja, Tanpa Tampilan GUI 3D)**:
  ```bash
  ros2 launch Simulasi-1 launch.launch.py headless:=true
  ```

---

### 3. Memantau Topik & Telemetri

Buka terminal baru dan aktifkan environment workspace (`source install/setup.bash`):

* **Melihat Data Pengukuran Jarak UWB Robot 1**:
  ```bash
  ros2 topic echo /agent_0/uwb/links_json
  ```

* **Melihat Matriks Jarak Global Swarm**:
  ```bash
  ros2 topic echo /swarm/distance_matrix
  ```

* **Melihat Perintah Kecepatan Motor Robot 1**:
  ```bash
  ros2 topic echo /agent_0/cmd_vel
  ```

* **Melihat Titik Kontrol Hidung Depan Robot 1**:
  ```bash
  ros2 topic echo /agent_0/front_point
  ```

---

## Parameter Konfigurasi

| Parameter | Nilai Default | Satuan | Keterangan |
| :--- | :--- | :--- | :--- |
| `target_distance` | `2.0` | meter | Jarak formasi ideal antar robot |
| `front_d` | `0.25` | meter | Jarak titik transformasi hidung depan |
| `wheelbase` | `0.5` | meter | Jarak antar poros roda robot ($L$) |
| `max_steer_rad` | `0.523` ($\pm 30^\circ$) | radian | Batas fisik sudut kemudi kemudi sepeda |
| `max_speed` | `1.0` | m/s | Batas kecepatan linear maksimum robot |
| `comm_range` | `6.0` | meter | Jangkauan komunikasi efektif pada GAT |
| `force_scale` | `10.0` | skala | Pengali gaya dorong AI untuk mengatasi inersia |
| `control_rate_hz`| `50.0` | Hz | Frekuensi eksekusi loop kontrol agen |
| `uwb_noise_std` | `0.025` (2.5 cm) | meter | Standar deviasi noise pengukuran sensor UWB |
