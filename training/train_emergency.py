"""
train_emergency.py — Training Script for Emergency PPO Agent
===============================================================

Trains the PPO-based EmergencyAgent to optimise green corridor timing.
Run from the project root:

    python -m training.train_emergency

Pipeline:
  1. GPU diagnostics
  2. Load config
  3. Create emergency environment
  4. Measure rule-based baseline (greedy corridor)
  5. Train PPO for 200K timesteps
  6. Evaluate learned policy vs rule-based
  7. Generate comparison plot and save results

WHY FEWER TIMESTEPS THAN THE TRAFFIC AGENT
───────────────────────────────────────────
Emergency traversal is a simpler, shorter-horizon task (~60-90 steps
per event).  The policy only needs to learn "give green to the edge
the emergency vehicle is approaching."  200K steps is sufficient for
convergence because:
  • The observation space is the same (17 dims) but the relevant
    features are fewer (emergency flag + vehicle position dominate).
  • The action space is the same (4 phases) but the optimal action
    is strongly correlated with one observation dimension.
  • γ=0.95 → effective horizon ~90 steps → each episode provides
    dense learning signal for the entire traversal.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.emergency_env import EmergencyEnv
from agents.emergency_agent import EmergencyAgent


# ──────────────────────────────────────────────────────────────────────
#  GPU Diagnostics
# ──────────────────────────────────────────────────────────────────────

def print_gpu_info() -> str:
    """Print GPU diagnostics and return device string."""
    print("=" * 60)
    print("  GPU DIAGNOSTICS (Emergency Training)")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Device       : {gpu_name}")
        print(f"  VRAM Total   : {vram_total:.1f} GB")
        print(f"  CUDA Version : {torch.version.cuda}")
        print(f"  PyTorch      : {torch.__version__}")
    else:
        device = "cpu"
        print("  No CUDA GPU detected — training on CPU")
        print(f"  PyTorch      : {torch.__version__}")

    print("=" * 60)
    return device


# ──────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load training config from YAML."""
    config_path = PROJECT_ROOT / "training" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"  Config loaded from {config_path}")
    return config


# ──────────────────────────────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────────────────────────────

def create_env(config: dict, port: int = 8825) -> EmergencyEnv:
    """Create a single EmergencyEnv for training/evaluation."""
    sumo_cfg = config["environment"]["sumo_cfg"]
    max_steps = config["environment"]["max_steps"]

    env = EmergencyEnv(
        config_path=sumo_cfg,
        port=port,
        use_gui=False,
        max_steps=max_steps,
    )
    env = Monitor(env)
    return env


# ──────────────────────────────────────────────────────────────────────
#  Rule-Based Baseline
# ──────────────────────────────────────────────────────────────────────

def run_rule_based_baseline(config: dict, n_episodes: int = 5) -> dict:
    """
    Measure the rule-based (greedy) corridor performance.

    This is the default mode of EmergencyAgent — it applies green to
    every traffic light on the emergency route immediately.  We measure
    total waiting time and emergency travel time as the baseline.
    """
    print("\n  Measuring RULE-BASED corridor baseline...")

    env = EmergencyEnv(
        config_path=config["environment"]["sumo_cfg"],
        port=8826,
        use_gui=False,
        max_steps=config["environment"]["max_steps"],
    )

    episode_waits = []
    episode_rewards = []
    episode_queues = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0
        step = 0

        while not (done or truncated):
            # Rule-based: cycle through phases with fixed timing
            # The env internally handles emergency corridor logic
            phase = (step // 30) % 4
            obs, reward, done, truncated, info = env.step(phase)
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)
            step += 1

        episode_waits.append(total_wait)
        episode_rewards.append(total_reward)
        episode_queues.append(total_queue)

    env.close()

    results = {
        "mean_waiting_time": float(np.mean(episode_waits)),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_queue_length": float(np.mean(episode_queues)),
        "std_waiting_time": float(np.std(episode_waits)),
    }

    print(
        f"   Rule-based baseline ({n_episodes} eps):\n"
        f"     Mean waiting time: {results['mean_waiting_time']:.1f} s\n"
        f"     Mean reward      : {results['mean_reward']:.2f}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
#  Post-Training Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_trained_agent(model: PPO, config: dict, n_episodes: int = 10) -> dict:
    """Evaluate the trained PPO agent with deterministic actions."""
    print("\n  Evaluating TRAINED emergency agent...")

    env = EmergencyEnv(
        config_path=config["environment"]["sumo_cfg"],
        port=8827,
        use_gui=False,
        max_steps=config["environment"]["max_steps"],
    )

    episode_waits = []
    episode_rewards = []
    episode_queues = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)

        episode_waits.append(total_wait)
        episode_rewards.append(total_reward)
        episode_queues.append(total_queue)

    env.close()

    results = {
        "mean_waiting_time": float(np.mean(episode_waits)),
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_queue_length": float(np.mean(episode_queues)),
        "std_waiting_time": float(np.std(episode_waits)),
        "std_reward": float(np.std(episode_rewards)),
    }

    print(
        f"   Trained agent ({n_episodes} eps):\n"
        f"     Mean waiting time: {results['mean_waiting_time']:.1f} s\n"
        f"     Mean reward      : {results['mean_reward']:.2f} "
        f"+/- {results['std_reward']:.2f}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────
#  Comparison Plot
# ──────────────────────────────────────────────────────────────────────

def generate_comparison_plot(
    rule_baseline: dict,
    trained_results: dict,
    reward_history: list,
    save_path: str,
) -> None:
    """Generate a 2-panel comparison: waiting time bars + learning curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Emergency Agent: Learned PPO vs Rule-Based Corridor",
        fontsize=16, fontweight="bold",
    )

    # Left: Waiting Time Comparison
    ax1 = axes[0]
    strategies = ["Rule-Based\n(Greedy)", "Learned\n(PPO)"]
    wait_means = [
        rule_baseline["mean_waiting_time"],
        trained_results["mean_waiting_time"],
    ]
    wait_stds = [
        rule_baseline.get("std_waiting_time", 0),
        trained_results.get("std_waiting_time", 0),
    ]
    colors = ["#f39c12", "#2ecc71"]
    bars = ax1.bar(strategies, wait_means, yerr=wait_stds, color=colors,
                   capsize=5, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean Total Waiting Time (s)", fontsize=12)
    ax1.set_title("Waiting Time Comparison", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, wait_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"{val:.0f}s", ha="center", va="bottom", fontweight="bold")

    # Right: Learning Curve
    ax2 = axes[1]
    if reward_history:
        window = min(50, len(reward_history) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(
                reward_history, np.ones(window) / window, mode="valid",
            )
            x_vals = np.linspace(0, len(reward_history), len(smoothed))
            ax2.plot(x_vals, smoothed, color="#9b59b6", linewidth=2,
                     label="Smoothed reward")
            x_raw = np.arange(len(reward_history))
            ax2.plot(x_raw, reward_history, color="#9b59b6", alpha=0.15)
        else:
            ax2.plot(reward_history, color="#9b59b6", linewidth=1)

    ax2.set_xlabel("Training Episodes", fontsize=12)
    ax2.set_ylabel("Episode Reward", fontsize=12)
    ax2.set_title("PPO Learning Curve", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    """Full emergency agent training pipeline."""
    total_start = time.time()

    # 1. GPU
    device = print_gpu_info()

    # 2. Config
    config = load_config()

    # 3. Directories
    Path(config["paths"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)

    # 4. Rule-based baseline
    n_baseline = config["evaluation"]["baseline_episodes"]
    rule_baseline = run_rule_based_baseline(config, n_baseline)

    # 5. Create training environment
    print("\n  Creating training environment...")
    train_env = create_env(config, port=8828)

    # 6. Initialize PPO
    ppo_cfg = config.get("ppo", {})
    total_timesteps = ppo_cfg.get("total_timesteps", 200_000)

    model = PPO(
        policy=ppo_cfg.get("policy", "MlpPolicy"),
        env=train_env,
        device=device,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 128),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.95),
        policy_kwargs=dict(net_arch=ppo_cfg.get("net_arch", [128, 128])),
        tensorboard_log=config["paths"].get("log_dir", "logs/") + "emergency_ppo",
        verbose=1,
    )

    # 7. Train
    print(f"\n  Training PPO for {total_timesteps:,} timesteps...")
    print(f"   Device: {device}")

    checkpoint_cb = CheckpointCallback(
        save_freq=config["training"].get("save_frequency", 50000),
        save_path=config["paths"]["model_dir"],
        name_prefix="emergency_ppo",
    )

    train_start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )
    train_duration = time.time() - train_start

    # 8. Save final model
    final_path = str(Path(config["paths"]["model_dir"]) / "emergency_ppo_final")
    model.save(final_path)
    print(f"  Final model saved to {final_path}")

    # 9. Extract learning curve
    reward_history = []
    if hasattr(model, "ep_info_buffer"):
        reward_history = [ep["r"] for ep in model.ep_info_buffer]

    # 10. Close training env
    train_env.close()

    # 11. Evaluate
    n_eval = config["evaluation"]["eval_episodes"]
    trained_results = evaluate_trained_agent(model, config, n_eval)

    # 12. Print comparison
    print("\n" + "=" * 60)
    print("  EMERGENCY AGENT — FINAL RESULTS")
    print("=" * 60)
    print(f"  {'Strategy':<20} {'Wait Time':<15} {'Reward':<12}")
    print(f"  {'─' * 47}")
    print(
        f"  {'Rule-Based':<20} "
        f"{rule_baseline['mean_waiting_time']:<15.1f} "
        f"{rule_baseline['mean_reward']:<12.2f}"
    )
    print(
        f"  {'Learned (PPO)':<20} "
        f"{trained_results['mean_waiting_time']:<15.1f} "
        f"{trained_results['mean_reward']:<12.2f}"
    )
    print("=" * 60)

    if rule_baseline["mean_waiting_time"] > 0:
        improvement = (
            (rule_baseline["mean_waiting_time"] - trained_results["mean_waiting_time"])
            / rule_baseline["mean_waiting_time"] * 100
        )
        print(f"\n  Improvement over rule-based: {improvement:+.1f}%")

    # 13. Plot
    plot_path = "docs/emergency_results.png"
    generate_comparison_plot(rule_baseline, trained_results, reward_history, plot_path)

    # 14. Save summary
    total_duration = time.time() - total_start
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "total_timesteps": total_timesteps,
        "training_duration_s": train_duration,
        "total_pipeline_duration_s": total_duration,
        "baselines": {"rule_based": rule_baseline},
        "trained_agent": trained_results,
        "model_path": final_path,
    }

    summary_path = "docs/emergency_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")
    print(f"\n  Full pipeline completed in {total_duration / 60:.1f} minutes")


if __name__ == "__main__":
    main()
