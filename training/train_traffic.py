"""
train_traffic.py — Main Training Script for Traffic DQN Agent
===============================================================

This is the entry point for training the adaptive traffic signal agent.
Run it from the project root:

    python -m training.train_traffic

Full pipeline:
  1. Print GPU diagnostics
  2. Load config from training/config.yaml
  3. Create output directories
  4. Spin up 4 parallel SUMO environments
  5. Measure baselines (fixed-timer, random agent)
  6. Train DQN for 500K timesteps
  7. Evaluate trained agent
  8. Generate comparison plots and save results

WHY 4 PARALLEL ENVIRONMENTS SPEEDS UP TRAINING
───────────────────────────────────────────────
DQN training has a fundamental bottleneck: SUMO simulation is SLOW.
Each traci.simulationStep() computes vehicle physics, lane-changing,
route decisions, etc.  On a single core, one step takes ~2-5 ms.

With 4 parallel environments:
  • Each env runs its own SUMO process on a separate CPU core.
  • While env_0 is computing its step, envs 1-3 are ALSO computing.
  • The agent collects 4 transitions per wall-clock step instead of 1.
  • Effective data collection speed: ~4× faster.

The GPU (Q-network updates) is NOT the bottleneck — it processes a
256-sample batch in ~1 ms.  The bottleneck is waiting for SUMO.
Parallelising SUMO across CPU cores is the biggest speedup you can get.

With 4 envs at ~200 steps/sec each, total throughput ≈ 800 steps/sec.
500,000 steps ÷ 800 steps/sec ≈ 625 seconds ≈ 10 minutes of pure
training time.  With overhead (logging, checkpoints), expect ~30-45
minutes total on an RTX 4050 + 8-core CPU.

WHAT SubprocVecEnv DOES TECHNICALLY
───────────────────────────────────
SubprocVecEnv (Subprocess Vectorised Environment) works as follows:

  Main process (Python):
    ├── Subprocess 0: runs TrafficEnv on port 8813 (has its own SUMO)
    ├── Subprocess 1: runs TrafficEnv on port 8814 (has its own SUMO)
    ├── Subprocess 2: runs TrafficEnv on port 8815 (has its own SUMO)
    └── Subprocess 3: runs TrafficEnv on port 8816 (has its own SUMO)

  On each step:
    1. Main process sends actions [a0, a1, a2, a3] to all subprocesses
       via multiprocessing.Pipe (inter-process communication).
    2. Each subprocess calls env.step(action) independently.
    3. Main process blocks until ALL subprocesses return.
    4. Results are stacked: obs shape (4, 17), rewards shape (4,), etc.
    5. The DQN receives a batch of 4 transitions at once.

  Each subprocess has its own Python interpreter, its own SUMO instance,
  and its own TraCI connection.  They share NOTHING except the Pipe to
  the main process.  Different ports (8813-8816) prevent SUMO socket
  conflicts.

  Alternative: DummyVecEnv runs all envs in the SAME process sequentially.
  This is simpler but offers ZERO speedup because Python's GIL prevents
  true parallelism.  SubprocVecEnv bypasses the GIL by using OS processes.

WHY WE MEASURE BASELINES BEFORE TRAINING
─────────────────────────────────────────
Baselines answer the question: "Is the trained agent actually BETTER
than simpler alternatives?"

We measure two baselines:

  1. FIXED-TIMER (30-second phases):
     This is how most real-world traffic lights work — a fixed cycle
     regardless of traffic conditions.  If the DQN can't beat a fixed
     timer, adaptive control isn't worth the complexity.

  2. RANDOM AGENT:
     Picks phases uniformly at random.  This is the WORST reasonable
     policy.  If the DQN performs CLOSE to random, training has failed.

Expected ranking after successful training:
    Random (worst) < Fixed Timer < Trained DQN (best)

If the ranking is different, something is wrong:
  • DQN ≈ Random → reward function is broken
  • DQN < Fixed Timer → hyperparameters need tuning
  • DQN >> Fixed Timer → success! The agent learned traffic patterns

HOW TO READ THE COMPARISON CHART
─────────────────────────────────
The output plot (docs/results.png) has 3 subplots:

  Left: Bar chart of mean waiting times
    Lower is better.  Each bar represents one strategy.
    The trained agent's bar should be the shortest.

  Middle: Learning curve (reward vs timesteps)
    Shows how the agent's reward improved during training.
    Should trend upward.  If it's flat or declining, training failed.
    Occasional dips are normal (exploration noise).

  Right: Queue length comparison
    Shows average queue lengths per approach for each strategy.
    The trained agent should have more balanced queues (no approach
    is starved) with lower overall values.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.emergency_env import EmergencyEnv
from simulation.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent, TrafficWandbCallback


# ──────────────────────────────────────────────────────────────────────
#  GPU Diagnostics
# ──────────────────────────────────────────────────────────────────────

def print_gpu_info() -> str:
    """Print full GPU diagnostics and return device string."""
    print("=" * 60)
    print("  GPU DIAGNOSTICS")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_free = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / (1024 ** 3)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()

        print(f"  Device       : {gpu_name}")
        print(f"  VRAM Total   : {vram_total:.1f} GB")
        print(f"  VRAM Free    : {vram_free:.1f} GB")
        print(f"  CUDA Version : {cuda_version}")
        print(f"  cuDNN Version: {cudnn_version}")
        print(f"  PyTorch      : {torch.__version__}")
    else:
        device = "cpu"
        print("  No CUDA GPU detected — training on CPU (will be slow)")
        print(f"  PyTorch      : {torch.__version__}")

    print("=" * 60)
    return device


# ──────────────────────────────────────────────────────────────────────
#  Config Loader
# ──────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load training config from YAML file."""
    config_path = PROJECT_ROOT / "training" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {config_path}")
    return config


# ──────────────────────────────────────────────────────────────────────
#  Directory Setup
# ──────────────────────────────────────────────────────────────────────

def create_directories(config: dict) -> None:
    """Create all output directories."""
    dirs = [
        config["paths"]["model_dir"],
        config["paths"]["log_dir"],
        config["paths"]["tensorboard_dir"],
        "docs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")


# ──────────────────────────────────────────────────────────────────────
#  Environment Factory
# ──────────────────────────────────────────────────────────────────────

def make_env(cfg_path: str, port: int, max_steps: int):
    """
    Return a function that creates a single TrafficEnv instance.

    SubprocVecEnv requires a LIST of callables (not instances) because
    each environment must be constructed INSIDE its own subprocess.
    If we passed an already-created env, Python would try to pickle
    the TraCI connection — which is a live TCP socket and cannot be
    serialised.  Creating the env inside the subprocess avoids this.
    """
    def _init():
        # Use EmergencyEnv instead of TrafficEnv to support preemption events
        env = EmergencyEnv(
            config_path=cfg_path,
            port=port,
            max_steps=max_steps,
        )
        env = Monitor(env)  # wraps env to track episode rewards/lengths
        return env
    return _init


def create_parallel_envs(config: dict) -> SubprocVecEnv:
    """
    Create N parallel SUMO environments using SubprocVecEnv.

    Each environment runs in its own OS process with its own SUMO
    instance on a unique TraCI port to avoid socket conflicts.
    """
    num_envs = config["environment"]["num_envs"]
    base_port = config["environment"]["base_port"]
    cfg_path = config["environment"]["sumo_cfg"]
    max_steps = config["environment"]["max_steps"]

    env_fns = [
        make_env(cfg_path, base_port + i, max_steps)
        for i in range(num_envs)
    ]

    print(f"✓ Creating {num_envs} parallel environments on ports "
          f"{base_port}–{base_port + num_envs - 1}")

    vec_env = SubprocVecEnv(env_fns)
    return vec_env


# ──────────────────────────────────────────────────────────────────────
#  Single Environment (for evaluation/baselines)
# ──────────────────────────────────────────────────────────────────────

def create_single_env(config: dict, port: int = 8820) -> EmergencyEnv:
    """Create a single EmergencyEnv instance for evaluation/testing."""
    env = EmergencyEnv(
        config_path=config["environment"]["sumo_cfg"],
        port=port,
        use_gui=False,
        max_steps=config["environment"]["max_steps"],
    )
    return env


# ──────────────────────────────────────────────────────────────────────
#  Baseline: Fixed Timer
# ──────────────────────────────────────────────────────────────────────

def run_fixed_timer_baseline(
    config: dict, n_episodes: int = 5
) -> dict:
    """
    Run the simulation with a fixed 30-second phase cycle.

    This simulates a traditional traffic light: each phase runs for
    exactly phase_duration seconds, then switches to the next phase.
    No adaptation, no intelligence — just a clock.

    This is the baseline most traffic engineers compare against.
    """
    print("\n📏 Measuring FIXED-TIMER baseline...")

    phase_duration = config["evaluation"]["fixed_timer_phase_duration"]
    env = create_single_env(config, port=8820)

    episode_waits = []
    episode_queues = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0
        step = 0

        while not (done or truncated):
            # Fixed timer: cycle through phases 0→1→2→3→0→...
            # Each phase held for phase_duration steps (= seconds)
            phase = (step // phase_duration) % 4
            obs, reward, done, truncated, info = env.step(phase)
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)
            step += 1

        episode_waits.append(total_wait)
        episode_queues.append(total_queue)
        episode_rewards.append(total_reward)

    env.close()

    results = {
        "mean_waiting_time": float(np.mean(episode_waits)),
        "mean_queue_length": float(np.mean(episode_queues)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_waiting_time": float(np.std(episode_waits)),
    }

    print(
        f"   Fixed-timer baseline ({n_episodes} episodes):\n"
        f"     Mean waiting time: {results['mean_waiting_time']:.1f} s\n"
        f"     Mean queue length: {results['mean_queue_length']:.1f}\n"
        f"     Mean reward      : {results['mean_reward']:.2f}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
#  Baseline: Random Agent
# ──────────────────────────────────────────────────────────────────────

def run_random_baseline(
    config: dict, n_episodes: int = 5
) -> dict:
    """
    Run the simulation with completely random phase selections.

    This is the WORST reasonable policy — it represents zero intelligence.
    If the trained agent performs near or below this, training has failed.
    """
    print("\n🎲 Measuring RANDOM baseline...")

    env = create_single_env(config, port=8821)

    episode_waits = []
    episode_queues = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0

        while not (done or truncated):
            action = env.action_space.sample()  # random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)

        episode_waits.append(total_wait)
        episode_queues.append(total_queue)
        episode_rewards.append(total_reward)

    env.close()

    results = {
        "mean_waiting_time": float(np.mean(episode_waits)),
        "mean_queue_length": float(np.mean(episode_queues)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_waiting_time": float(np.std(episode_waits)),
    }

    print(
        f"   Random baseline ({n_episodes} episodes):\n"
        f"     Mean waiting time: {results['mean_waiting_time']:.1f} s\n"
        f"     Mean queue length: {results['mean_queue_length']:.1f}\n"
        f"     Mean reward      : {results['mean_reward']:.2f}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
#  Post-Training Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_trained_agent(
    model: DQN, config: dict, n_episodes: int = 10
) -> dict:
    """
    Evaluate the trained DQN agent over n_episodes episodes.

    Uses deterministic=True (greedy, no exploration) to measure
    the pure learned policy without ε-noise.
    """
    print("\n🤖 Evaluating TRAINED agent...")

    env = create_single_env(config, port=8822)

    episode_waits = []
    episode_queues = []
    episode_rewards = []
    per_direction_queues = {"N": [], "S": [], "E": [], "W": []}

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0
        ep_dir_queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)

            # Per-direction queue tracking
            q_dir = info.get("queue_per_direction", {})
            for d in ["N", "S", "E", "W"]:
                ep_dir_queues[d] += q_dir.get(d, 0.0)

        episode_waits.append(total_wait)
        episode_queues.append(total_queue)
        episode_rewards.append(total_reward)
        for d in ["N", "S", "E", "W"]:
            per_direction_queues[d].append(ep_dir_queues[d])

    env.close()

    results = {
        "mean_waiting_time": float(np.mean(episode_waits)),
        "mean_queue_length": float(np.mean(episode_queues)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_waiting_time": float(np.std(episode_waits)),
        "std_reward": float(np.std(episode_rewards)),
        "queue_per_direction": {
            d: float(np.mean(v)) for d, v in per_direction_queues.items()
        },
    }

    print(
        f"   Trained agent ({n_episodes} episodes):\n"
        f"     Mean waiting time: {results['mean_waiting_time']:.1f} s\n"
        f"     Mean queue length: {results['mean_queue_length']:.1f}\n"
        f"     Mean reward      : {results['mean_reward']:.2f} "
        f"± {results['std_reward']:.2f}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
#  Comparison Plot
# ──────────────────────────────────────────────────────────────────────

def generate_comparison_plot(
    fixed_baseline: dict,
    random_baseline: dict,
    trained_results: dict,
    reward_history: list,
    save_path: str,
) -> None:
    """
    Generate a 3-panel comparison figure:
      Left   — Bar chart: mean waiting times (lower is better)
      Middle — Learning curve: reward over training timesteps
      Right  — Bar chart: per-direction queue lengths

    HOW TO READ THIS CHART:
    ───────────────────────
    Left panel (Waiting Time Comparison):
      Three bars: Random, Fixed Timer, Trained DQN.
      SHORTER bars = BETTER performance.
      Error bars show standard deviation across episodes.
      The trained agent's bar should be noticeably shorter.

    Middle panel (Learning Curve):
      X-axis = training timesteps.  Y-axis = episode reward.
      Should trend UPWARD over time.
      Early dips are normal (agent is exploring).
      A FLAT curve means the agent isn't learning (check reward fn).
      A curve that rises then CRASHES means instability (lower LR).

    Right panel (Queue Length per Direction):
      Grouped bars: N, S, E, W for each strategy.
      A good agent has LOW and BALANCED queues across all directions.
      If one direction has much higher queues, the agent is being
      unfair (starving that approach of green time).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Traffic Agent Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # ── Left: Waiting Time Bar Chart ──────────────────────────────────
    ax1 = axes[0]
    strategies = ["Random", "Fixed Timer", "Trained DQN"]
    wait_means = [
        random_baseline["mean_waiting_time"],
        fixed_baseline["mean_waiting_time"],
        trained_results["mean_waiting_time"],
    ]
    wait_stds = [
        random_baseline.get("std_waiting_time", 0),
        fixed_baseline.get("std_waiting_time", 0),
        trained_results.get("std_waiting_time", 0),
    ]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax1.bar(strategies, wait_means, yerr=wait_stds, color=colors,
                   capsize=5, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean Total Waiting Time (s)", fontsize=12)
    ax1.set_title("Waiting Time Comparison", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, wait_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:.0f}s", ha="center", va="bottom", fontweight="bold")

    # ── Middle: Learning Curve ────────────────────────────────────────
    ax2 = axes[1]
    if reward_history:
        # Smooth the curve with a rolling average for readability
        window = min(50, len(reward_history) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(
                reward_history,
                np.ones(window) / window,
                mode="valid",
            )
            x_vals = np.linspace(0, 500_000, len(smoothed))
            ax2.plot(x_vals, smoothed, color="#3498db", linewidth=2,
                     label="Smoothed reward")
            # Raw data as transparent background
            x_raw = np.linspace(0, 500_000, len(reward_history))
            ax2.plot(x_raw, reward_history, color="#3498db", alpha=0.15)
        else:
            x_vals = np.linspace(0, 500_000, len(reward_history))
            ax2.plot(x_vals, reward_history, color="#3498db", linewidth=1)

    ax2.set_xlabel("Training Timesteps", fontsize=12)
    ax2.set_ylabel("Episode Reward", fontsize=12)
    ax2.set_title("Learning Curve", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # ── Right: Per-Direction Queue Comparison ─────────────────────────
    ax3 = axes[2]
    directions = ["N", "S", "E", "W"]
    x = np.arange(len(directions))
    width = 0.25

    # Only trained agent has per-direction data; use total for others
    trained_q = [
        trained_results.get("queue_per_direction", {}).get(d, 0)
        for d in directions
    ]
    # Estimate per-direction for baselines (divide total by 4)
    fixed_q_avg = fixed_baseline["mean_queue_length"] / 4
    random_q_avg = random_baseline["mean_queue_length"] / 4

    ax3.bar(x - width, [random_q_avg] * 4, width, label="Random",
            color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax3.bar(x, [fixed_q_avg] * 4, width, label="Fixed Timer",
            color="#f39c12", edgecolor="black", linewidth=0.5)
    ax3.bar(x + width, trained_q, width, label="Trained DQN",
            color="#2ecc71", edgecolor="black", linewidth=0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(directions, fontsize=12)
    ax3.set_ylabel("Mean Queue Length (vehicles)", fontsize=12)
    ax3.set_title("Queue Length per Direction", fontsize=13, fontweight="bold")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Comparison plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
#  Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    """Full training pipeline: diagnostics → baselines → train → evaluate → save."""
    total_start = time.time()

    # ── 1. GPU diagnostics ────────────────────────────────────────────
    device = print_gpu_info()

    # ── 2. Load config ────────────────────────────────────────────────
    config = load_config()

    # ── 3. Create directories ─────────────────────────────────────────
    create_directories(config)

    # ── 4. Measure baselines ──────────────────────────────────────────
    n_baseline = config["evaluation"]["baseline_episodes"]
    fixed_baseline = run_fixed_timer_baseline(config, n_baseline)
    random_baseline = run_random_baseline(config, n_baseline)

    print("\n" + "=" * 60)
    print("  BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Fixed Timer : {fixed_baseline['mean_waiting_time']:.1f}s wait")
    print(f"  Random      : {random_baseline['mean_waiting_time']:.1f}s wait")
    print("=" * 60)

    # ── 5. Create parallel training environments ──────────────────────
    vec_env = create_parallel_envs(config)

    # ── 6. Initialize agent ───────────────────────────────────────────
    agent = TrafficAgent(env=vec_env, model_dir=config["paths"]["model_dir"])

    # ── 7. Train ──────────────────────────────────────────────────────
    total_timesteps = config["training"]["total_timesteps"]
    print(f"\n🚀 Training DQN for {total_timesteps:,} timesteps...")
    print(f"   Device: {device}")
    print(f"   Parallel envs: {config['environment']['num_envs']}")

    training_results = agent.train(total_timesteps=total_timesteps)

    # ── 8. Extract learning curve from SB3 monitor ────────────────────
    # SB3's ep_info_buffer stores the last 100 episode rewards
    reward_history = []
    if hasattr(agent.model, "ep_info_buffer"):
        reward_history = [ep["r"] for ep in agent.model.ep_info_buffer]

    # ── 9. Close parallel envs ────────────────────────────────────────
    vec_env.close()

    # ── 10. Post-training evaluation ──────────────────────────────────
    n_eval = config["evaluation"]["eval_episodes"]
    trained_results = evaluate_trained_agent(
        agent.model, config, n_eval
    )

    # ── 11. Print comparison ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  {'Strategy':<15} {'Wait Time':<15} {'Reward':<12}")
    print(f"  {'─' * 42}")
    print(
        f"  {'Random':<15} "
        f"{random_baseline['mean_waiting_time']:<15.1f} "
        f"{random_baseline['mean_reward']:<12.2f}"
    )
    print(
        f"  {'Fixed Timer':<15} "
        f"{fixed_baseline['mean_waiting_time']:<15.1f} "
        f"{fixed_baseline['mean_reward']:<12.2f}"
    )
    print(
        f"  {'Trained DQN':<15} "
        f"{trained_results['mean_waiting_time']:<15.1f} "
        f"{trained_results['mean_reward']:<12.2f}"
    )
    print("=" * 60)

    # Improvement calculation
    if fixed_baseline["mean_waiting_time"] > 0:
        improvement = (
            (fixed_baseline["mean_waiting_time"] - trained_results["mean_waiting_time"])
            / fixed_baseline["mean_waiting_time"]
            * 100
        )
        print(f"\n  📈 Improvement over fixed timer: {improvement:+.1f}%")

    # ── 12. Generate comparison plot ──────────────────────────────────
    plot_path = config["paths"]["results_plot"]
    generate_comparison_plot(
        fixed_baseline, random_baseline, trained_results,
        reward_history, plot_path,
    )

    # ── 13. Save training summary ─────────────────────────────────────
    total_duration = time.time() - total_start
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "total_timesteps": total_timesteps,
        "training_duration_s": training_results["training_duration_s"],
        "total_pipeline_duration_s": total_duration,
        "baselines": {
            "fixed_timer": fixed_baseline,
            "random": random_baseline,
        },
        "trained_agent": trained_results,
        "model_path": training_results["final_model_path"],
        "config": config,
    }

    summary_path = config["paths"]["training_summary"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved to {summary_path}")

    print(f"\n✅ Full pipeline completed in {total_duration / 60:.1f} minutes")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
