"""
traffic_agent.py — DQN Traffic Signal Control Agent
=====================================================

This module implements a Deep Q-Network (DQN) agent that learns to
control traffic signal phases at a single intersection.  It wraps
stable-baselines3's DQN and adds GPU-aware logging, W&B integration,
and traffic-domain evaluation metrics.

WHY THE REPLAY BUFFER STAYS IN CPU RAM, NOT GPU VRAM
─────────────────────────────────────────────────────
The experience replay buffer stores ALL past transitions (s, a, r, s', done).
With buffer_size=100,000 and an observation size of 17 floats:

    Memory per transition ≈ 17×4 bytes (obs) + 17×4 (next_obs) + 4 (action)
                           + 4 (reward) + 1 (done) ≈ 145 bytes
    Total buffer ≈ 100,000 × 145 ≈ 14.5 MB

14.5 MB is trivial for CPU RAM (you have 16+ GB) but VRAM is precious.
Your RTX 4050 has only 6 GB of VRAM, which must hold:
  • The Q-network (~1 MB for our architecture)
  • The target network (~1 MB, a frozen copy)
  • The current mini-batch (~256 × 145 bytes ≈ 37 KB)
  • PyTorch's computation graph and gradients (~2-10 MB)
  • CUDA context overhead (~300-500 MB)

If we put the full buffer in VRAM, that's 14.5 MB of VRAM wasted on
cold storage that's only touched once per gradient step (when we sample
a mini-batch).  CPU→GPU transfer of a 256-sample batch takes ~0.1 ms —
negligible compared to the forward/backward pass (~1-5 ms).

Rule of thumb: keep HOT data (model weights, current batch, gradients)
on GPU; keep COLD data (replay buffer, logging state) on CPU.

WHAT IS THE TARGET NETWORK AND WHY IT PREVENTS INSTABILITY
──────────────────────────────────────────────────────────
In Q-learning, we update Q(s,a) toward the target:
    target = r + γ × max_a' Q(s', a')

The problem: Q(s', a') is computed by the SAME network we're updating.
So every gradient step changes the target we're chasing — it's like a
dog chasing its own tail.  The targets shift with every update, causing
oscillations and divergence.

Solution: maintain a SECOND copy of the network (the "target network")
that is frozen for `target_update_interval` steps (1000 steps here).
We compute targets using the frozen copy:
    target = r + γ × max_a' Q_target(s', a')

Every 1000 steps, we copy the current network's weights into the target
network.  This gives the targets a stable "anchor" that only shifts
periodically, dramatically reducing oscillation.

Analogy: imagine studying for an exam where the textbook rewrites itself
every time you read a page.  You'd never converge on the material.  The
target network is like making a PHOTOCOPY of the textbook and studying
from the photocopy for a week — the answers are stable long enough for
you to learn.

WHAT EPSILON-GREEDY MEANS FOR A TRAFFIC LIGHT AGENT
────────────────────────────────────────────────────
Epsilon-greedy is the agent's exploration strategy:
    • With probability ε: choose a RANDOM signal phase (explore)
    • With probability 1-ε: choose the phase with highest Q-value (exploit)

For a traffic light specifically:
    Early training (ε ≈ 1.0):
        The agent randomly flips between NS-green, NS-yellow, EW-green,
        EW-yellow.  This looks TERRIBLE — signals flicker, queues build
        up, waiting times explode.  But this chaos is essential because
        the agent needs to see what happens when bad decisions are made.
        Without exploration, it would never discover that holding NS-green
        longer reduces north-south waiting time.

    Mid training (ε ≈ 0.3):
        The agent mostly follows its learned Q-values (e.g., "NS queue is
        long → switch to NS-green") but still occasionally tries unusual
        phases.  This is where it discovers edge cases like "giving EW a
        brief green before the NS queue backs up prevents gridlock."

    Late training (ε = 0.05):
        The agent exploits its learned policy 95% of the time.  The 5%
        exploration prevents it from getting stuck in a local optimum
        (e.g., always alternating NS/EW on a fixed timer when an
        adaptive timing would be better).

    exploration_fraction=0.2 means ε decays from 1.0 to 0.05 over the
    first 20% of training (100,000 of 500,000 steps).

WHY GAMMA=0.99 MEANS THE AGENT PLANS ROUGHLY 100 STEPS AHEAD
─────────────────────────────────────────────────────────────
The discount factor γ determines how much the agent values future
rewards vs immediate rewards.  A reward received k steps in the
future is worth γ^k times its nominal value.

    γ = 0.99
    γ^100 = 0.99^100 ≈ 0.366
    γ^200 = 0.99^200 ≈ 0.134
    γ^500 = 0.99^500 ≈ 0.007

So a reward 100 steps away is worth 36.6% of an immediate reward —
still significant.  A reward 500 steps away is worth only 0.7% —
effectively ignored.  The "planning horizon" is roughly the number
of steps where γ^k drops below ~0.01, which for γ=0.99 is about
log(0.01) / log(0.99) ≈ 460 steps.

For traffic: each step is 1 second, so γ=0.99 makes the agent care
about consequences up to ~7-8 minutes into the future.  This is
perfect for traffic signals:
  • Changing a phase NOW affects queue buildup for the next few minutes.
  • But traffic conditions beyond ~10 minutes are effectively random
    (new vehicles arrive, demand patterns shift).
  • If γ were 0.999 the agent would try to optimise for conditions
    20+ minutes ahead — but it can't predict those, so it would chase
    noise in the Q-values.

WHY BATCH_SIZE=256 IS GOOD FOR RTX 4050
───────────────────────────────────────
Your RTX 4050 has 6 GB VRAM and 2560 CUDA cores.

Batch size affects two things:
  1. GPU utilisation: too small → cores sit idle → slow training
  2. Gradient quality: too small → noisy gradients → unstable training

For our network (3 layers: 256, 256, 128):
  • Forward pass processes all 256 samples in parallel across CUDA cores.
  • Memory per sample: ~17 input floats → 256 → 256 → 128 → 4 output floats
    ≈ 2.6 KB of activations per sample.
  • 256 samples × 2.6 KB ≈ 0.65 MB for the full batch — trivial for 6 GB.

  • batch_size=32 would leave most of the 2560 CUDA cores idle.
  • batch_size=256 gives enough parallelism to keep cores busy.
  • batch_size=1024+ would still fit in memory but gives diminishing
    returns on speed and can actually HURT learning (gradients become
    too averaged, reducing the beneficial noise that helps escape
    local optima).

256 is the sweet spot: enough parallelism for GPU efficiency, small
enough for healthy gradient variance, minimal memory footprint.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Optional: wandb integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
#  Custom W&B Callback
# ──────────────────────────────────────────────────────────────────────

class TrafficWandbCallback(BaseCallback):
    """
    Custom callback that logs traffic-specific metrics to W&B every
    `log_freq` steps.

    Metrics logged:
      • mean_reward        — rolling average over last 100 episodes
      • mean_waiting_time  — from environment info dict
      • epsilon            — current exploration rate
      • gpu_memory_mb      — VRAM allocated (if CUDA)
      • training_fps       — env steps per second
    """

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._start_time: float = 0.0
        self._last_log_step: int = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        # ── Mean reward (last 100 episodes) ───────────────────────────
        # SB3 tracks episode rewards internally in the monitor wrapper.
        # ep_info_buffer is a deque of dicts with key "r" (reward).
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean(
                [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            )
        else:
            mean_reward = 0.0

        # ── Mean waiting time from environment info ───────────────────
        infos = self.locals.get("infos", [{}])
        waiting_times = [
            info.get("total_waiting_time", 0.0) for info in infos
        ]
        mean_waiting_time = np.mean(waiting_times) if waiting_times else 0.0

        # ── Current epsilon ───────────────────────────────────────────
        # SB3 DQN stores the exploration schedule; we read current value
        current_epsilon = self.model.exploration_rate

        # ── Training FPS ──────────────────────────────────────────────
        elapsed = time.time() - self._start_time
        steps_since_last = self.num_timesteps - self._last_log_step
        fps = steps_since_last / max(elapsed, 1e-6)
        self._start_time = time.time()
        self._last_log_step = self.num_timesteps

        # ── GPU memory (if CUDA) ──────────────────────────────────────
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        # ── Log to W&B ────────────────────────────────────────────────
        metrics = {
            "train/mean_reward": mean_reward,
            "train/mean_waiting_time": mean_waiting_time,
            "train/epsilon": current_epsilon,
            "train/gpu_memory_mb": gpu_memory_mb,
            "train/fps": fps,
            "train/timesteps": self.num_timesteps,
        }

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

        # Also log to TensorBoard via SB3's built-in logger
        for key, value in metrics.items():
            self.logger.record(key, value)

        if self.verbose >= 1:
            print(
                f"\n📊 Step {self.num_timesteps:,}  |  "
                f"ε={current_epsilon:.3f}  |  "
                f"mean_R={mean_reward:.2f}  |  "
                f"wait={mean_waiting_time:.1f}s  |  "
                f"GPU={gpu_memory_mb:.0f}MB  |  "
                f"FPS={fps:.0f}"
            )

        return True


# ──────────────────────────────────────────────────────────────────────
#  TrafficAgent
# ──────────────────────────────────────────────────────────────────────

class TrafficAgent:
    """
    DQN agent for adaptive traffic signal control.

    Wraps stable-baselines3's DQN with:
      • GPU-aware device selection (CUDA if available)
      • W&B + TensorBoard dual logging
      • Automatic checkpoint saving with timestamps
      • Traffic-domain evaluation (waiting time, queue metrics)

    Parameters
    ----------
    env : gymnasium.Env
        A Gymnasium-compatible TrafficEnv instance.
    model_dir : str
        Directory where trained model checkpoints are saved.
    """

    def __init__(self, env: Any, model_dir: str = "models/saved/"):
        self.env = env
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ── Device selection ──────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TrafficAgent using: {self.device}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")

        # ── Initialize DQN ────────────────────────────────────────────
        self.model = DQN(
            # ── Policy architecture ───────────────────────────────────
            # MlpPolicy = multi-layer perceptron (fully connected network).
            # net_arch=[256, 256, 128] means:
            #   Input (17) → Dense(256) → ReLU → Dense(256) → ReLU
            #   → Dense(128) → ReLU → Output (4 Q-values, one per phase)
            #
            # 256 neurons in the first two layers give the network enough
            # capacity to learn complex traffic patterns (time-of-day effects,
            # queue interactions between approaches).  128 in the final layer
            # narrows toward the 4-action output.
            policy="MlpPolicy",
            env=env,
            device=self.device,
            policy_kwargs=dict(net_arch=[256, 256, 128]),

            # ── Learning rate ─────────────────────────────────────────
            # 1e-4 is a safe default for DQN.  Too high → Q-values oscillate.
            # Too low → training takes forever.  Adam optimizer (SB3 default)
            # adapts per-parameter, so 1e-4 works across different layer sizes.
            learning_rate=1e-4,

            # ── Replay buffer ─────────────────────────────────────────
            # 100,000 transitions stored in CPU RAM (see module docstring
            # for why NOT in VRAM).
            #
            # Buffer must be large enough to decorrelate samples:
            #   • Too small (1000): the agent keeps replaying recent
            #     experience, overfitting to the current traffic pattern.
            #   • Too large (10M): wastes RAM and dilutes recent experience
            #     with very old, possibly irrelevant transitions.
            #   • 100K is ~27 full episodes (3600 steps each), giving a
            #     good mix of old and recent experience.
            buffer_size=100_000,

            # ── Learning starts ───────────────────────────────────────
            # Don't update the network for the first 5000 steps.
            # Fill the replay buffer with diverse experiences first so
            # early mini-batches aren't all from the same traffic state.
            learning_starts=5_000,

            # ── Batch size ────────────────────────────────────────────
            # 256 samples per gradient step.  See module docstring for
            # why this is optimal for RTX 4050 (6 GB VRAM, 2560 cores).
            batch_size=256,

            # ── Discount factor ───────────────────────────────────────
            # γ=0.99 → agent plans ~460 steps (7-8 minutes) ahead.
            # See module docstring for the math.
            gamma=0.99,

            # ── Training frequency ────────────────────────────────────
            # Update the Q-network every 4 environment steps.
            # This means for every 4 (s,a,r,s') transitions collected,
            # we do 1 gradient descent step on a random mini-batch.
            # Higher values (8, 16) speed up data collection but slow
            # learning per experience.
            train_freq=4,

            # ── Target network update interval ────────────────────────
            # Sync the frozen target network every 1000 gradient steps.
            # See module docstring for why the target network exists.
            target_update_interval=1_000,

            # ── Exploration schedule ──────────────────────────────────
            # ε decays linearly from 1.0 to 0.05 over the first 20% of
            # training.  See module docstring for what this means for
            # a traffic light agent specifically.
            exploration_fraction=0.2,
            exploration_final_eps=0.05,

            # ── Logging ───────────────────────────────────────────────
            tensorboard_log="logs/traffic_dqn",
            verbose=1,
        )

    # ──────────────────────────────────────────────────────────────────
    #  train()
    # ──────────────────────────────────────────────────────────────────

    def train(self, total_timesteps: int = 500_000) -> Dict[str, Any]:
        """
        Train the DQN agent for `total_timesteps` environment steps.

        Workflow:
          1. Initialize W&B run (if available).
          2. Register custom callback for metric logging every 10K steps.
          3. Train via stable-baselines3's .learn() method.
          4. Save final model with timestamp.
          5. Close W&B run.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to train for (default 500K).

        Returns
        -------
        dict
            Training results: final_model_path, total_timesteps, device.
        """
        # ── Initialize W&B ────────────────────────────────────────────
        wandb_run = None
        if WANDB_AVAILABLE:
            wandb_run = wandb.init(
                project="traffic-rl-twin",
                config={
                    "algorithm": "DQN",
                    "total_timesteps": total_timesteps,
                    "learning_rate": 1e-4,
                    "buffer_size": 100_000,
                    "batch_size": 256,
                    "gamma": 0.99,
                    "net_arch": [256, 256, 128],
                    "device": self.device,
                    "exploration_fraction": 0.2,
                    "exploration_final_eps": 0.05,
                },
                sync_tensorboard=True,
                save_code=True,
            )

        # ── Callbacks ─────────────────────────────────────────────────
        callbacks = CallbackList([
            TrafficWandbCallback(log_freq=10_000, verbose=1),
        ])

        # ── Train ─────────────────────────────────────────────────────
        print(f"\n🚀 Starting training: {total_timesteps:,} steps on {self.device}")
        train_start = time.time()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )

        train_duration = time.time() - train_start

        # ── Save final model ──────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"traffic_dqn_{timestamp}"
        self.model.save(str(model_path))
        print(f"\n✓ Training complete in {train_duration:.0f}s")
        print(f"  Model saved to: {model_path}")

        # ── Close W&B ─────────────────────────────────────────────────
        if wandb_run is not None:
            wandb_run.finish()

        return {
            "final_model_path": str(model_path),
            "total_timesteps": total_timesteps,
            "training_duration_s": train_duration,
            "device": self.device,
        }

    # ──────────────────────────────────────────────────────────────────
    #  predict()
    # ──────────────────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Any]:
        """
        Choose an action for the given observation.

        SB3's model.predict() handles device placement internally:
          1. Converts the numpy observation to a PyTorch tensor.
          2. Moves it to self.device (CPU or CUDA).
          3. Runs the forward pass on the Q-network.
          4. Returns the action as a numpy scalar.

        Parameters
        ----------
        observation : np.ndarray, shape (17,)
            Normalised state vector from TrafficEnv.
        deterministic : bool
            True → always pick argmax Q(s,a) (greedy, for evaluation).
            False → follow ε-greedy (for data collection / training).

        Returns
        -------
        action : int
            Signal phase index ∈ {0, 1, 2, 3}.
        _states : None
            Placeholder for recurrent policies (not used by MlpPolicy).
        """
        action, _states = self.model.predict(
            observation, deterministic=deterministic
        )
        return int(action), _states

    # ──────────────────────────────────────────────────────────────────
    #  evaluate()
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Run the current policy for `n_episodes` episodes with NO
        exploration (deterministic=True) and return metrics.

        Parameters
        ----------
        n_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict with keys:
            mean_reward         — average total reward per episode
            std_reward          — standard deviation of rewards
            mean_waiting_time   — average total waiting time per episode
            mean_queue_length   — average total queue length per episode
            mean_episode_length — average number of steps per episode
        """
        episode_rewards = []
        episode_waiting_times = []
        episode_queue_lengths = []
        episode_lengths = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            total_waiting = 0.0
            total_queue = 0.0
            steps = 0

            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                total_waiting += info.get("total_waiting_time", 0.0)
                total_queue += info.get("total_queue_length", 0.0)
                steps += 1

            episode_rewards.append(total_reward)
            episode_waiting_times.append(total_waiting)
            episode_queue_lengths.append(total_queue)
            episode_lengths.append(steps)

        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_waiting_time": float(np.mean(episode_waiting_times)),
            "mean_queue_length": float(np.mean(episode_queue_lengths)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }

        print(
            f"\n📊 Evaluation ({n_episodes} episodes):\n"
            f"   Mean reward       : {results['mean_reward']:.2f} "
            f"± {results['std_reward']:.2f}\n"
            f"   Mean waiting time : {results['mean_waiting_time']:.1f} s\n"
            f"   Mean queue length : {results['mean_queue_length']:.1f}\n"
            f"   Mean ep. length   : {results['mean_episode_length']:.0f} steps"
        )

        return results

    # ──────────────────────────────────────────────────────────────────
    #  save() / load()
    # ──────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the DQN model (network weights + optimizer state) to disk.

        Also saves the replay buffer separately so training can
        resume without losing collected experience.
        """
        self.model.save(path)

        # Save replay buffer for training resumption
        buffer_path = path + "_replay_buffer"
        self.model.save_replay_buffer(buffer_path)

        print(f"✓ Model saved to {path}")
        print(f"  Replay buffer saved to {buffer_path}")

    def load(self, path: str) -> None:
        """
        Load a previously saved DQN model.

        The model is re-attached to the current environment so
        predict() and evaluate() work immediately after loading.
        """
        self.model = DQN.load(path, env=self.env, device=self.device)

        # Attempt to restore the replay buffer (non-fatal if missing)
        buffer_path = path + "_replay_buffer"
        if os.path.exists(buffer_path + ".pkl"):
            self.model.load_replay_buffer(buffer_path)
            print(f"  Replay buffer restored from {buffer_path}")

        print(f"✓ Model loaded from {path}")
