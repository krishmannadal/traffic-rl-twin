"""
evaluate.py — Standalone Evaluation Script
============================================

Evaluate a trained agent without re-training.

Usage:
    python -m training.evaluate --agent traffic
    python -m training.evaluate --agent emergency
    python -m training.evaluate --agent traffic --model models/saved/traffic_dqn_final.zip
    python -m training.evaluate --agent emergency --episodes 20

This script:
  1. Loads a trained model from disk
  2. Runs it in deterministic mode (no exploration)
  3. Prints per-episode and aggregate statistics
  4. Saves results to docs/eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.emergency_env import EmergencyEnv
from simulation.traffic_env import TrafficEnv


def load_config() -> dict:
    """Load training config from YAML."""
    config_path = PROJECT_ROOT / "training" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_eval_env(config: dict, port: int = 8830):
    """Create a single evaluation environment."""
    return EmergencyEnv(
        config_path=config["environment"]["sumo_cfg"],
        port=port,
        use_gui=False,
        max_steps=config["environment"]["max_steps"],
    )


def evaluate(model, env, n_episodes: int, agent_type: str) -> dict:
    """
    Run deterministic evaluation over n_episodes.

    Returns a dict with per-episode and aggregate statistics.
    """
    print(f"\n  Evaluating {agent_type} agent for {n_episodes} episodes...")

    per_episode = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_wait = 0.0
        total_queue = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0.0)
            total_queue += info.get("total_queue_length", 0.0)
            steps += 1

        ep_result = {
            "episode": ep + 1,
            "reward": round(total_reward, 4),
            "waiting_time": round(total_wait, 2),
            "queue_length": round(total_queue, 2),
            "steps": steps,
        }
        per_episode.append(ep_result)
        print(
            f"   Episode {ep+1:>3}/{n_episodes}: "
            f"reward={total_reward:>8.3f}  "
            f"wait={total_wait:>8.1f}s  "
            f"steps={steps}"
        )

    rewards = [e["reward"] for e in per_episode]
    waits = [e["waiting_time"] for e in per_episode]
    queues = [e["queue_length"] for e in per_episode]

    aggregate = {
        "agent_type": agent_type,
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards)), 4),
        "mean_waiting_time": round(float(np.mean(waits)), 2),
        "std_waiting_time": round(float(np.std(waits)), 2),
        "mean_queue_length": round(float(np.mean(queues)), 2),
        "best_reward": round(float(np.max(rewards)), 4),
        "worst_reward": round(float(np.min(rewards)), 4),
    }

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {agent_type.upper()}")
    print("=" * 60)
    print(f"  Mean Reward   : {aggregate['mean_reward']} +/- {aggregate['std_reward']}")
    print(f"  Mean Wait     : {aggregate['mean_waiting_time']}s +/- {aggregate['std_waiting_time']}s")
    print(f"  Mean Queue    : {aggregate['mean_queue_length']}")
    print(f"  Best Reward   : {aggregate['best_reward']}")
    print(f"  Worst Reward  : {aggregate['worst_reward']}")
    print("=" * 60)

    return {"aggregate": aggregate, "per_episode": per_episode}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent")
    parser.add_argument(
        "--agent", type=str, default="traffic",
        choices=["traffic", "emergency"],
        help="Which agent to evaluate (traffic=DQN, emergency=PPO)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the .zip model file. If not provided, uses default location.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--port", type=int, default=8830,
        help="TraCI port for the evaluation environment",
    )
    args = parser.parse_args()

    config = load_config()

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_dir = config["paths"]["model_dir"]
        if args.agent == "traffic":
            model_path = str(Path(model_dir) / "traffic_dqn_final.zip")
        else:
            model_path = str(Path(model_dir) / "emergency_ppo_final.zip")

    # Check model exists
    if not Path(model_path).exists():
        print(f"\n  ERROR: Model file not found: {model_path}")
        print("  Have you trained the agent yet?")
        print(f"    python -m training.train_{'traffic' if args.agent == 'traffic' else 'emergency'}")
        sys.exit(1)

    # Load model
    print(f"  Loading model from {model_path}...")
    if args.agent == "traffic":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)

    # Create environment
    env = create_eval_env(config, port=args.port)

    # Run evaluation
    start_time = time.time()
    results = evaluate(model, env, args.episodes, args.agent)
    eval_duration = time.time() - start_time

    # Clean up
    env.close()

    # Save results
    results["eval_duration_s"] = round(eval_duration, 2)
    results["model_path"] = model_path

    output_path = Path("docs") / f"eval_results_{args.agent}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")
    print(f"  Evaluation completed in {eval_duration:.1f}s")


if __name__ == "__main__":
    main()
