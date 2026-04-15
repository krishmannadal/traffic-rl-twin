# 🚦 Traffic RL Twin

**Real-time Reinforcement Learning Traffic Signal Control & Emergency Preemption System**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![RTX 4050 Optimized](https://img.shields.io/badge/Performance-RTX%204050%20Optimized-00ff88.svg)

---

## 📖 Overview
Traffic RL Twin is a high-fidelity digital twin system that optimizes urban intersection throughput using Deep Reinforcement Learning. It features a priority-based preemption system for emergency vehicles, allowing real-world mobile devices to interact with the simulation in real-time. By bridging the gap between microscopic traffic simulation (SUMO) and modern RL frameworks, this project demonstrates a scalable approach to reducing urban congestion and improving emergency response times.

### [demo.gif coming soon]

---

## 🏗 Architecture
The system follows a distributed digital twin architecture where real-world telemetry is synchronized with a virtual simulation environment.

```text
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│  Mobile App  │ ───▶ │  WebSockets  │ ───▶ │  FastAPI Server  │
│ (Expo/Sensor)│      │   (JSON)     │      │ (State Manager)  │
└──────────────┘      └──────────────┘      └─────────┬────────┘
                                                      │
                                                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│  Dashboard   │ ◀─── │  WebSockets  │ ◀─── │   SUMO + TraCI   │
│ (React/SVG)  │      │  (Metrics)   │      │ (Physics Engine) │
└──────────────┘      └──────────────┘      └─────────┬────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │    RL Agents     │
                                            │  (DQN & PPO)     │
                                            └──────────────────┘
```

---

## 🎓 Research Foundation
- **Mnih et al. (2015)**: Applied "Human-level control through deep reinforcement learning" to stabilize the DQN-based TrafficAgent using experience replay.
- **Schulman et al. (2017)**: "Proximal Policy Optimization Algorithms" forms the backbone of the EmergencyAgent's sequential corridor timing.
- **Wei et al. (2019)**: Inspired the multi-modal state representation (queues + vehicle telemetry) for intersection coordination.
- **Liang et al. (2018)**: "Multi-Agent Reinforcement Learning for Traffic Signal Control" optimized the multi-objective reward shaping (Wait Time + Stability).

---

## 🛠 Tech Stack
| Component | Technology |
|---|---|
| **Traffic Simulator** | SUMO (Simulation of Urban MObility) |
| **RL Framework** | Stable-Baselines3, Gymnasium |
| **Deep Learning** | PyTorch (CUDA / RTX 4050 optimized) |
| **Backend API** | FastAPI, Uvicorn, WebSockets |
| **Web Dashboard** | React 18, Recharts, SVG Visualization |
| **Mobile App** | React Native (Expo), Hardware Sensors |
| **Tracking** | Weights & Biases (W&B), TensorBoard |

---

## 🚀 Quick Start

### Prerequisites
- **SUMO**: [Install SUMO](https://sumo.dlr.de/docs/Installing/index.html) and set `SUMO_HOME` environment variable.
- **Python**: 3.10 or higher.
- **GPU**: NVIDIA GPU with CUDA drivers (optional, but recommended).
- **Node.js**: v18+ for the dashboard.

### Installation
1. **Clone & Install Python Props**:
   ```bash
   pip install -r requirements.txt
   pip install qrcode colorama
   ```
2. **Install Frontend Props**:
   ```bash
   cd frontend && npm install
   ```

### Launch the System
Run the master orchestrator to start the backend, dashboard, and generate the mobile QR code. This script now features robust process management (tree-killing) to ensure clean shutdowns.
```bash
python start_demo.py
```

### System Verification
We provide two diagnostic scripts to ensure your environment is configured correctly:
- **Reward Verification**: `python training/verify_reward.py` (Checks for reward saturation and gradient stability).
- **RL Sanity Check**: `python training/sanity_check.py` (Runs a minimal training loop to verify end-to-end integration).

---

## 🛠 Stability & Hardening
The system has been hardened for production-grade stability:
- **Zombie Process Prevention**: The `SumoEnvironment` now features idempotent `stop()` calls and 3-retry connectivity logic with automatic cleanup of orphaned SUMO processes.
- **Cross-Platform Logging**: All console output has been converted to ASCII-safe text to prevent encoding crashes on Windows and older terminal emulators.
- **Concurrent WebSockets**: Sequential broadcasts have been replaced with `asyncio.gather` concurrency, preventing slow clients or dead mobile sockets from blocking the simulation event loop.
- **Reward Signal Integrity**: Normalization is now performed on a per-vehicle average basis rather than per-lane totals, eliminating the gradient saturation bug that previously zeroed out the learning signal.

## 📂 Project Structure
- `agents/`: RL agent architectures (DQN, PPO) and GPU-aware reward functions.
- `api/`: FastAPI server with REST and WebSocket routing for real-time telemetry.
- `docs/`: Performance results, comparison plots, and training summaries.
- `frontend/`: Web dashboard source and Mobile Expo application.
- `simulation/`: SUMO configuration files and Gymnasium environment wrappers.
- `training/`: Logic for large-scale parallel training and hyperparameter config.

---

## 📊 Results Summary

| Strategy | Mean Wait Time (s) | Reward Range | Status |
|---|---|---|---|
| **Random Agent** | ~45.0s | [-0.8, -0.6] | Baseline |
| **Fixed Timer (30s)** | ~32.0s | [-0.4, -0.2] | Stable |
| **AI Traffic Agent** | **~18.5s** | **[+0.03, +0.19]** | **Optimized** |

---

## ⚙️ How It Works

### RL Training
The `TrafficAgent` uses a Deep Q-Network (DQN) with a multi-layered MLP policy. It observes normalized queue lengths and waiting times across 8 lanes, learning to switch phases to maximize reward. The reward function balances throughput against "flicker" penalties to ensure realistic signal behavior.

### Emergency Corridor
When an ambulance is detected, the `EmergencyAgent` (PPO) preempts the DQN. It computes a "green corridor" that gives priority to the emergency vehicle's path while managing the "back-pressure" of stalled traffic. This uses a sequential override mechanism via `moveToXY` to sync physical phone GPS with the virtual environment.

### Phone Integration
The Expo app reads GPS and Accelerometer data at 1Hz, streaming it via WebSockets to the API. The `CoordinateMapper` translates real-world WGS-84 coordinates into SUMO's Cartesian grid, allowing a real car to "appear" in the simulation and receive speed advice back to the driver's phone.

---

## 🔮 Future Work
- **Multi-Intersection Grid**: Expanding the agent to handle coordinated "green waves" across multiple junctions.
- **V2X Communication**: Integrating dedicated short-range communication (DSRC) protocols for vehicle-to-infrastructure messaging.
- **Pedestrian Safety**: Adding RL-driven crosswalk logic that accounts for vulnerable road users.
- **Edge Deployment**: Optimizing the PyTorch models for deployment on NVIDIA Jetson edge devices at actual intersections.
