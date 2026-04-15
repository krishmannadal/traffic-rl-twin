"""
emergency_agent.py   Green Corridor Agent for Emergency Vehicles
==================================================================

This module implements an agent that manages traffic signal preemption
when an emergency vehicle (ambulance, fire truck) needs to traverse
the network.  It supports two modes:

  1. RULE-BASED (default)   greedy green corridor, reliable for demos
  2. LEARNED   PPO policy that optimises corridor timing

WHY PPO OVER DQN FOR SEQUENTIAL CORRIDOR DECISIONS
---------------------------------------------------
DQN and PPO are fundamentally different algorithms for different problems:

  DQN (used by TrafficAgent):
      Learns a Q-function: Q(s, a) = "expected future reward if I take
      action a in state s and act optimally afterwards."
      Works best for DISCRETE, SINGLE-STEP decisions with a stable
      environment: "which phase should the light be in RIGHT NOW?"
      Uses a replay buffer -> can reuse old experience (sample efficient).
      Struggles with SEQUENTIAL decision-making where actions have
      delayed, cascading effects.

  PPO (used here for EmergencyAgent):
      Learns a POLICY directly:  (a|s) = "probability of taking action
      a in state s."
      Excels at SEQUENTIAL decisions: "should I preempt THIS intersection
      now, hold it for 10 more seconds, or release it and preempt the
      NEXT one?"
      On-policy: uses only recent experience (less sample efficient but
      more stable for non-stationary problems).
      Handles TEMPORAL CREDIT ASSIGNMENT better than DQN because the
      policy gradient naturally propagates reward through action sequences.

  For emergency corridors, the decision is inherently sequential:
    Step 1: Preempt intersection A -> emergency vehicle enters
    Step 2: Hold green at A for N seconds -> vehicle crosses
    Step 3: Release A, preempt intersection B -> vehicle approaches
    Step 4: Hold green at B -> vehicle crosses
    ...

  Each decision depends on the PREVIOUS ones (you can't preempt B before
  the vehicle is near it, and you must release A before cross-traffic
  gridlocks).  PPO's policy gradient handles this chain of dependencies
  naturally; DQN's single-step Q-values struggle to capture it.

WHY RULE-BASED FIRST: DEMO RELIABILITY VS THEORETICAL OPTIMALITY
-----------------------------------------------------------------
The rule-based corridor is the default for three pragmatic reasons:

  1. RELIABILITY   For a demo/presentation, you need the emergency vehicle
     to actually get through.  A partially trained RL policy might fail
     spectacularly (keep the vehicle stuck at a red light).  The rule-based
     approach has a 100% success rate: if the vehicle is on edge X, edge X
     gets green.

  2. BASELINE COMPARISON   You can't claim your learned policy is "better"
     without a comparison.  The rule-based corridor IS that baseline.
     Improvement is measured as: learned_time < rule_based_time.

  3. TRAINING DATA   The rule-based mode can run during data collection,
     generating trajectories that the PPO agent learns from.  This is
     a form of imitation learning / warm-starting that dramatically
     reduces the number of episodes needed.

  The switch_mode() method lets you flip between modes at runtime, so
  you can demo the reliable version and then show the learned version
  improving upon it.

HOW EMERGENCY AGENT AND TRAFFIC AGENT COORDINATE
-------------------------------------------------
The two agents operate on a simple PRIORITY HIERARCHY:

  +---------------------------------------------------------+
  |  Emergency Agent (HIGH PRIORITY)                        |
  |    When active: takes FULL CONTROL of all signals       |
  |    on the emergency route. Traffic agent's actions are   |
  |    silently discarded by EmergencyEnv.step().            |
  |                                                         |
  |  Traffic Agent (LOW PRIORITY)                           |
  |    When emergency is inactive: normal DQN control.      |
  |    When emergency is active: still runs predict() but   |
  |    its output is ignored. It continues observing and     |
  |    learning (the emergency is part of its experience).   |
  +---------------------------------------------------------+

  The handoff is managed by EmergencyEnv:
      trigger_emergency() -> emergency_active = True -> DQN disabled
      deactivate_emergency() -> emergency_active = False -> DQN re-enabled

  There is NO explicit communication channel between the agents.  The
  traffic agent sees the emergency via observation index 16 (emergency
  flag) and can learn to ANTICIPATE disruption (e.g., pre-clear queues
  on the corridor route before preemption starts).  But during active
  preemption, it has zero authority.

  This mirrors real-world preemption systems: the preemption controller
  has unconditional priority.  The normal timing plan resumes after
  preemption with a brief recovery period.

WHY GAMMA=0.95 (SHORTER PLANNING HORIZON THAN TRAFFIC AGENT)
-------------------------------------------------------------
The traffic agent uses  =0.99 (horizon   460 steps   7-8 minutes)
because traffic patterns evolve slowly   a queue takes minutes to build.

The emergency agent uses  =0.95 because:
     ^20  = 0.95^20    0.358
     ^50  = 0.95^50    0.077
     ^100 = 0.95^100   0.006

  Planning horizon   log(0.01)/log(0.95)   90 steps   1.5 minutes.

  Emergency vehicle traversal of our small network takes 30-90 seconds.
  The agent only needs to plan for the DURATION OF THE EMERGENCY, not
  beyond it.  After the vehicle passes, the emergency agent deactivates
  and its future rewards are zero   there's nothing to plan for.

  A higher   (0.99) would make the agent overweight the "aftermath"
  (traffic recovery after preemption) relative to the "action" (getting
  the vehicle through fast).  Since the EmergencyReward.compute()
  primarily rewards travel time savings, the agent should focus on the
  immediate traversal, not what happens 5 minutes later.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from agents.reward import EmergencyReward

# Optional W&B
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ----------------------------------------------------------------------
#  Emergency W&B Callback
# ----------------------------------------------------------------------

class EmergencyWandbCallback(BaseCallback):
    """
    Logs emergency-specific metrics to W&B every `log_freq` steps.
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        # Mean reward from recent episodes
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer]
            )
        else:
            mean_reward = 0.0

        # GPU memory
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        metrics = {
            "emergency/mean_reward": mean_reward,
            "emergency/gpu_memory_mb": gpu_mem,
            "emergency/timesteps": self.num_timesteps,
        }

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

        for key, value in metrics.items():
            self.logger.record(key, value)

        return True


# ----------------------------------------------------------------------
#  EmergencyAgent
# ----------------------------------------------------------------------

class EmergencyAgent:
    """
    Agent for managing emergency vehicle green corridors.

    Supports two modes:
      "rule_based"   greedy corridor (reliable, default for demos)
      "learned"      PPO-trained policy (potentially better timing)

    Parameters
    ----------
    env : EmergencyEnv
        The emergency-aware environment.
    model_dir : str
        Directory for saving trained models.
    mode : str
        Initial mode   "rule_based" or "learned".
    """

    def __init__(
        self,
        env: Any,
        model_dir: str = "models/saved/",
        mode: str = "rule_based",
    ):
        self.env = env
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        # -- Device selection ------------------------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"EmergencyAgent using: {self.device} | mode: {self.mode}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")

        # -- Emergency state -------------------------------------------
        self._active: bool = False
        self._vehicle_id: str = ""
        self._origin: str = ""
        self._destination: str = ""
        self._activation_time: float = 0.0
        self._metrics_log: List[Dict[str, Any]] = []

        # -- Emergency reward function ---------------------------------
        self._reward_fn = EmergencyReward(device=self.device)

        # -- Initialize PPO (for learned mode) -------------------------
        # PPO uses on-policy learning:
        #   1. Collect n_steps transitions using the current policy.
        #   2. Compute advantages (how much better each action was than average).
        #   3. Update the policy for n_epochs passes over the collected data.
        #   4. Discard the data and collect fresh transitions.
        #
        # Key differences from DQN:
        #     No replay buffer (on-policy = use data once then discard)
        #     n_steps=2048: collect this many transitions before each update
        #     n_epochs=10: reuse the n_steps data 10 times (PPO's clipping
        #     prevents the policy from changing too much per epoch)
        #     batch_size=128: each epoch processes the 2048 transitions in
        #     mini-batches of 128 (2048/128 = 16 gradient steps per epoch)
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            device=self.device,

            # -- Learning rate -----------------------------------------
            # 3e-4 is the default for PPO (from the original paper).
            # Higher than DQN's 1e-4 because PPO uses on-policy data
            # (discarded after each update)   it must learn fast from
            # each batch since it won't see that data again.
            learning_rate=3e-4,

            # -- Rollout length ----------------------------------------
            # Collect 2048 steps of experience before each policy update.
            # For emergency scenarios (~60-90 steps), this means ~20-30
            # full emergency events per update   enough variety to
            # compute stable advantage estimates.
            n_steps=2048,

            # -- Mini-batch size ---------------------------------------
            # Process the 2048 collected transitions in chunks of 128.
            # Smaller than DQN's 256 because PPO processes ALL data
            # each epoch (no sampling from a buffer), so total gradient
            # steps per update = (2048/128)   10 epochs = 160 updates.
            batch_size=128,

            # -- Number of epochs --------------------------------------
            # Reuse each batch of collected data this many times.
            # PPO's clipped surrogate objective ensures the policy doesn't
            # change too drastically despite multiple passes over the
            # same data   this is PPO's key innovation over vanilla PG.
            n_epochs=10,

            # -- Discount factor ---------------------------------------
            #  =0.95 -> planning horizon   90 steps   1.5 minutes.
            # See module docstring for why this is shorter than the
            # traffic agent's  =0.99.
            gamma=0.95,

            # -- Network architecture ----------------------------------
            # Smaller than traffic agent's [256,256,128] because the
            # emergency decision is simpler: "which phase helps the
            # emergency vehicle most at this instant?"  The traffic
            # agent must model complex queue dynamics across 4 approaches;
            # the emergency agent mostly needs to identify which approach
            # the emergency vehicle is on and give it green.
            policy_kwargs=dict(net_arch=[128, 128]),

            tensorboard_log="logs/emergency_ppo",
            verbose=1,
        )

    # ------------------------------------------------------------------
    #  activate()
    # ------------------------------------------------------------------

    def activate(
        self,
        emergency_vehicle_id: str,
        origin: str,
        destination: str,
    ) -> float:
        """
        Activate the emergency agent for a specific vehicle.

        Behaviour depends on self.mode:

          "rule_based":
            Immediately triggers the environment's green corridor logic.
            The corridor is computed greedily and applied in one shot.
            This is fast, reliable, and deterministic.

          "learned":
            Triggers the environment but lets the PPO policy control
            corridor timing.  The policy decides WHEN to apply green,
            how long to hold it, and when to release   potentially
            reducing disruption to cross-traffic.

        Parameters
        ----------
        emergency_vehicle_id : str
            Vehicle ID (e.g. "emergency_001").
        origin : str
            Starting edge (e.g. "south_to_center").
        destination : str
            Target edge (e.g. "center_to_north").

        Returns
        -------
        float
            Estimated baseline travel time (seconds).
        """
        self._active = True
        self._vehicle_id = emergency_vehicle_id
        self._origin = origin
        self._destination = destination
        self._activation_time = time.time()

        print(f"\n[ALERT] EmergencyAgent activated [{self.mode}]: {emergency_vehicle_id}")
        print(f"   Route: {origin} -> {destination}")

        # Trigger the emergency in the environment
        # This spawns the vehicle and computes baseline travel time.
        baseline_time = self.env.trigger_emergency(
            vehicle_id=emergency_vehicle_id,
            origin_edge=origin,
            destination_edge=destination,
        )

        if self.mode == "rule_based":
            # -- Rule-based: apply corridor immediately ----------------
            # The greedy corridor gives green to every traffic light
            # on the route.  Simple, reliable, and predictable.
            corridor = self.env.get_green_corridor(self.env.emergency_route)
            self.env.apply_corridor(corridor)
            print(f"   Corridor applied: {len(corridor)} traffic lights preempted")

        elif self.mode == "learned":
            # -- Learned: PPO will decide corridor timing --------------
            # In this mode, the EmergencyEnv.step() override calls
            # get_green_corridor() and apply_corridor() based on the
            # current vehicle position.  The PPO agent's action refines
            # WHEN and HOW LONG to apply preemption at each intersection.
            print("   PPO policy will control corridor timing dynamically")

        return baseline_time

    # ------------------------------------------------------------------
    #  deactivate()
    # ------------------------------------------------------------------

    def deactivate(self) -> Dict[str, Any]:
        """
        Deactivate the emergency agent and log performance metrics.

        Calls env.deactivate_emergency() to restore normal signal
        operation, then records the outcome for comparison.

        Returns
        -------
        dict
            Metrics including travel_time, baseline_time, time_saved,
            disruption_caused, and mode.
        """
        if not self._active:
            return {"status": "was_not_active"}

        # Metrics to return
        metrics = {}

        # Deactivate in the environment   this logs travel time vs baseline
        if self.env.emergency_active:
            env_log = self.env.deactivate_emergency()
        else:
            # Env may have already deactivated (auto-deactivation in step())
            # Use the last log entry from the environment
            env_log = self.env.emergency_log[-1] if self.env.emergency_log else {}

        # Calculate wall-clock activation duration
        wall_clock_duration = time.time() - self._activation_time

        # Build metrics
        metrics = {
            "vehicle_id": self._vehicle_id,
            "origin": self._origin,
            "destination": self._destination,
            "mode": self.mode,
            "travel_time": env_log.get("emergency_travel_time", 0.0),
            "baseline_time": env_log.get("baseline_travel_time", 0.0),
            "time_saved": env_log.get("time_saved", 0.0),
            "time_saved_pct": env_log.get("time_saved_pct", 0.0),
            "wall_clock_duration_s": wall_clock_duration,
        }

        self._metrics_log.append(metrics)

        # Log to W&B if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "emergency/travel_time": metrics["travel_time"],
                "emergency/baseline_time": metrics["baseline_time"],
                "emergency/time_saved": metrics["time_saved"],
                "emergency/time_saved_pct": metrics["time_saved_pct"],
                "emergency/mode": self.mode,
            })

        print(
            f"\n[OK] EmergencyAgent deactivated [{self.mode}]:\n"
            f"   Travel time : {metrics['travel_time']:.1f}s "
            f"(baseline: {metrics['baseline_time']:.1f}s)\n"
            f"   Time saved  : {metrics['time_saved']:.1f}s "
            f"({metrics['time_saved_pct']:.1f}%)"
        )

        # Reset internal state
        self._active = False
        self._vehicle_id = ""
        self._origin = ""
        self._destination = ""

        return metrics

    # ------------------------------------------------------------------
    #  train()
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 100_000) -> Dict[str, Any]:
        """
        Train the PPO policy for learned corridor timing.

        Only needed when mode == "learned".  The rule-based mode
        requires no training.

        Parameters
        ----------
        total_timesteps : int
            Number of environment steps to train for.

        Returns
        -------
        dict
            Training results.
        """
        if self.mode != "learned":
            print(
                "  [WARN] Training is only needed for 'learned' mode.\n"
                "  Current mode is '{self.mode}'. Switching to 'learned'."
            )
            self.mode = "learned"

        # [SECTION] Initialize W&B
        wandb_run = None
        if WANDB_AVAILABLE:
            wandb_run = wandb.init(
                project="traffic-rl-twin",
                name="emergency-ppo",
                config={
                    "algorithm": "PPO",
                    "total_timesteps": total_timesteps,
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 128,
                    "n_epochs": 10,
                    "gamma": 0.95,
                    "net_arch": [128, 128],
                    "device": self.device,
                },
                sync_tensorboard=True,
            )

        # [SECTION] Callbacks
        callbacks = CallbackList([
            EmergencyWandbCallback(log_freq=5_000, verbose=1),
        ])

        # [SECTION] Train
        print(f"\n[TRAIN] Training EmergencyAgent (PPO): {total_timesteps:,} steps")
        train_start = time.time()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        train_duration = time.time() - train_start

        # [SECTION] Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"emergency_ppo_{timestamp}"
        self.model.save(str(model_path))
        print(f"\n[DONE] Emergency PPO trained in {train_duration:.0f}s")
        print(f"  Model saved to: {model_path}")

        if wandb_run is not None:
            wandb_run.finish()

        return {
            "final_model_path": str(model_path),
            "total_timesteps": total_timesteps,
            "training_duration_s": train_duration,
            "device": self.device,
        }

    # [SECTION] switch_mode()
    # ------------------------------------------------------------------

    def switch_mode(self, mode: str) -> None:
        """
        Toggle between "rule_based" and "learned" modes.

        Parameters
        ----------
        mode : str
            Either "rule_based" or "learned".

        Raises
        ------
        ValueError
            If mode is not one of the two valid options.
        """
        valid_modes = ("rule_based", "learned")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {valid_modes}."
            )

        old_mode = self.mode
        self.mode = mode
        print(f"EmergencyAgent mode: {old_mode} -> {mode}")

        if mode == "learned":
            # Check if a trained model exists
            saved_models = list(self.model_dir.glob("emergency_ppo_*"))
            if not saved_models:
                print(
                "  [WARN] No trained emergency model found. "
                    "Using untrained PPO (random actions).\n"
                    "  Run agent.train() first for meaningful behaviour."
                )

    # [SECTION] predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Any]:
        """
        Choose an action using the PPO policy (learned mode only).

        In rule-based mode, the corridor is applied directly in
        activate()   this method is not called.

        Parameters
        ----------
        observation : np.ndarray
            Current state from the environment.
        deterministic : bool
            If True, use the mode of the action distribution.
            If False, sample from it (adds exploration noise).

        Returns
        -------
        action : int
            The chosen signal phase.
        _states : Any
            Internal LSTM states (None for MlpPolicy).
        """
        action, _states = self.model.predict(
            observation, deterministic=deterministic
        )
        return int(action), _states

    # ------------------------------------------------------------------
    #  save() / load()
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the PPO model to disk."""
        self.model.save(path)
        print(f"[DONE] EmergencyAgent model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a previously trained PPO model.

        Automatically switches mode to "learned" since loading a model
        implies the intent to use the learned policy.
        """
        self.model = PPO.load(path, env=self.env, device=self.device)
        self.mode = "learned"
        print(f"[DONE] EmergencyAgent model loaded from {path}")
        print(f"  Mode automatically set to 'learned'")

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether the agent is currently handling an emergency."""
        return self._active

    @property
    def metrics_history(self) -> List[Dict[str, Any]]:
        """Full log of all emergency events handled this session."""
        return self._metrics_log
