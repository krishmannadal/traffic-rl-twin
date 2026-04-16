"""
traffic_env.py — Gymnasium RL Environment Wrapper for SUMO
============================================================

This module wraps the low-level SumoEnvironment (TraCI bridge) into a
standard Gymnasium interface so it plugs directly into stable-baselines3,
RLlib, CleanRL, or any RL framework that speaks the Gymnasium API.

The Gymnasium contract requires five methods:
    __init__()   → define observation_space, action_space
    reset()      → return (observation, info)
    step(action) → return (observation, reward, terminated, truncated, info)
    render()     → optional visualisation / state dict
    close()      → clean up resources

WHY NORMALISE OBSERVATIONS TO [0, 1]?
──────────────────────────────────────
Neural networks learn by adjusting weights via gradient descent.  If one
input feature is in range [0, 300] (waiting time in seconds) and another
is [0, 4] (phase index), the network's gradients will be dominated by
the large-magnitude feature.  The small feature becomes nearly invisible
to the optimiser.

Normalising ALL features to [0, 1]:
  • Gives every feature equal influence on the initial forward pass.
  • Keeps gradient magnitudes balanced across input dimensions.
  • Prevents neurons from saturating (sigmoid/tanh) on first contact
    with the data — saturated neurons have near-zero gradients and
    stop learning ("dead neuron" problem).
  • Makes hyperparameters (learning rate, weight init) transferable
    across different environments without per-feature tuning.

WHY ONE-HOT ENCODING FOR THE PHASE?
────────────────────────────────────
The signal phase is a CATEGORICAL variable with 4 values {0,1,2,3}.
If we feed it as a single integer, the network sees an implicit ordering:
phase 3 > phase 2 > phase 1 > phase 0.  But there IS no ordering —
"NS green" is not "greater than" "EW green."

One-hot encoding [1,0,0,0], [0,1,0,0], etc. removes the false ordinality.
Each phase becomes an independent binary dimension, so the network can
learn separate weights for each phase without being confused by the
spurious numerical relationships (e.g., the "distance" between phase 0
and phase 3 is not meaningful).

This matters especially for DQN, where the Q-network must learn that
the value of being in phase 0 has no linear relationship to the value
of being in phase 3.

TERMINATED vs TRUNCATED IN GYMNASIUM
─────────────────────────────────────
Gymnasium v1.0+ distinguishes two kinds of episode endings:

  terminated = True
    The environment reached a NATURAL end state.  In traffic this would
    mean all vehicles have left the network (simulation over).  The
    value of the terminal state is ZERO — there is no future reward
    because nothing happens after termination.

  truncated = True
    The episode was CUT SHORT by an external limit (our 3600-step
    budget).  The environment is NOT in a terminal state — if we kept
    going there would be more reward.  The value of a truncated state
    is NOT zero — the RL algorithm must bootstrap (estimate future
    value using the Q-network) to avoid underestimating the policy.

Getting this wrong causes subtle bugs:
  • Treating truncation as termination → the agent learns that the end
    of every episode is worth zero, making the last ~100 steps appear
    worthless.  It "gives up" near the time limit.
  • Treating termination as truncation → the agent bootstraps past a
    real terminal state, overestimating value and producing unstable
    Q-targets.

WHY 3600 STEPS = 1 SIMULATED HOUR
──────────────────────────────────
Our SUMO simulation uses a step length of 1.0 seconds (set in
simulation.sumocfg).  So 3600 simulation steps × 1.0 s/step = 3600
seconds = 1 hour of simulated traffic.  This matches the standard
traffic engineering analysis period (the "peak hour" study) and gives
the agent enough time to experience multiple signal cycles.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from simulation.environment import SumoEnvironment, SimulationState
from agents.reward import TrafficReward


# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────

# Normalisation ceilings — raw SUMO values are divided by these to get [0, 1].
MAX_QUEUE = 30.0          # vehicles per direction (both lanes combined)
MAX_WAIT = 300.0          # seconds — 5 min is a very long red
MAX_VEHICLES = 40.0       # vehicles per direction

NUM_PHASES = 4            # number of signal phases in the TL program
MAX_STEPS = 3600          # 1 simulated hour at 1 s/step

# Direction → lane ID pairs (must match map.edg.xml / map.net.xml)
DIRECTIONS = {
    "N": ["north_to_center_0", "north_to_center_1"],
    "S": ["south_to_center_0", "south_to_center_1"],
    "E": ["east_to_center_0",  "east_to_center_1"],
    "W": ["west_to_center_0",  "west_to_center_1"],
}
DIR_ORDER = ["N", "S", "E", "W"]  # fixed order for observation vector


# ──────────────────────────────────────────────────────────────────────
#  TrafficEnv
# ──────────────────────────────────────────────────────────────────────

class TrafficEnv(gymnasium.Env):
    """
    Gymnasium environment for adaptive traffic signal control.

    Observation space — Box(17,) ∈ [0, 1], dtype float32
    ─────────────────────────────────────────────────────
    Index   Feature                     Source
    ─────   ───────                     ──────
    0–3     Queue lengths (N,S,E,W)     Halting vehicles / MAX_QUEUE
    4–7     Waiting times (N,S,E,W)     Cumulative wait  / MAX_WAIT
    8–11    Vehicle counts (N,S,E,W)    Vehicles on lane / MAX_VEHICLES
    12–15   Current phase (one-hot)     [1,0,0,0] or [0,1,0,0] etc.
    16      Emergency active            0.0 or 1.0

    Action space — Discrete(4)
    ──────────────────────────
    0 → Phase 0: NS green,  EW red
    1 → Phase 1: NS yellow, EW red
    2 → Phase 2: NS red,    EW green
    3 → Phase 3: NS red,    EW yellow

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the SUMO .sumocfg file.
    port : int
        TraCI TCP port.
    use_gui : bool
        Launch SUMO-GUI for visual debugging.
    max_steps : int
        Episode truncation limit (default 3600 = 1 simulated hour).
    """

    metadata = {"render_modes": ["human", "dict"]}

    def __init__(
        self,
        config_path: Optional[str] = None,
        net_file: Optional[str] = None,
        port: int = 8813,
        use_gui: bool = False,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()

        # ── Gymnasium spaces ──────────────────────────────────────────
        #
        # 17-dimensional observation vector, all values in [0, 1].
        # See class docstring for the index layout.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32,
        )

        # 4 discrete actions — one per signal phase.
        self.action_space = spaces.Discrete(NUM_PHASES)

        # ── SUMO bridge ───────────────────────────────────────────────
        self._sumo = SumoEnvironment(
            cfg_path=Path(config_path) if config_path else None,
            net_file=Path(net_file) if net_file else None,
            port=port,
            use_gui=use_gui,
        )

        # ── Reward calculator (GPU-aware, from agents/reward.py) ──────
        self._reward_fn = TrafficReward()

        # ── Episode bookkeeping ───────────────────────────────────────
        self._max_steps = max_steps
        self._step_count: int = 0
        self._last_state: Optional[SimulationState] = None

    # ──────────────────────────────────────────────────────────────────
    #  reset()
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Restart the SUMO simulation from t = 0.

        Returns
        -------
        observation : np.ndarray, shape (17,)
            The initial normalised observation vector.
        info : dict
            Raw un-normalised metrics for logging / dashboard.
        """
        super().reset(seed=seed)

        # Restart SUMO (closes old run if any, launches fresh process)
        sim_state = self._sumo.reset()

        # Reset episode counters
        self._step_count = 0
        self._last_state = sim_state

        # Reset reward function's internal phase tracker
        self._reward_fn.reset()

        observation = self._get_observation(sim_state)
        info = self._build_info(sim_state)

        return observation, info

    # ──────────────────────────────────────────────────────────────────
    #  step(action)
    # ──────────────────────────────────────────────────────────────────

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply an action, advance the simulation, and return the result.

        The action selects a traffic light phase via SumoEnvironment.step(),
        which internally calls traci.trafficlight.setPhase() and then
        traci.simulationStep() to advance the clock by 1 second.

        Parameters
        ----------
        action : int ∈ {0, 1, 2, 3}
            Desired signal phase index.

        Returns
        -------
        observation : np.ndarray   — normalised state (17,)
        reward      : float        — scalar from TrafficReward.compute()
        terminated  : bool         — always False (we run full episodes)
        truncated   : bool         — True after max_steps (3600)
        info        : dict         — raw metrics for logging / API
        """
        # ── Apply action and advance simulation ───────────────────────
        sim_state, sim_done = self._sumo.step(action)
        self._step_count += 1

        # ── Build observation vector ──────────────────────────────────
        observation = self._get_observation(sim_state)

        # ── Compute reward ────────────────────────────────────────────
        # Convert SimulationState to dict format expected by TrafficReward.
        # TrafficReward.compute() needs keys: waiting_time, queue_length,
        # vehicle_count, signal_phase.
        state_dict = {
            "waiting_time": sim_state.waiting_time,
            "queue_length": sim_state.queue_length,
            "vehicle_count": sim_state.vehicle_count,
            "signal_phase": sim_state.signal_phase,
        }
        reward = self._reward_fn.compute(state_dict)

        # ── Determine episode ending ──────────────────────────────────
        terminated, truncated = self._calculate_done(sim_done)

        # ── Build info dict ───────────────────────────────────────────
        info = self._build_info(sim_state)

        # Also include reward breakdown for TensorBoard / W&B logging
        info["reward_components"] = self._reward_fn.explain(state_dict)

        # Store last state for render()
        self._last_state = sim_state

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────
    #  render()
    # ──────────────────────────────────────────────────────────────────

    def render(self) -> Optional[Dict[str, Any]]:
        """
        Return the current simulation state as a dictionary.

        Designed for the FastAPI / WebSocket layer: the frontend calls
        GET /metrics and receives this dict as JSON.  No pixel rendering
        is performed — SUMO-GUI handles visuals if use_gui=True.

        Returns
        -------
        dict or None
            None if no state has been captured yet.
        """
        if self._last_state is None:
            return None
        return self._build_info(self._last_state)

    # ──────────────────────────────────────────────────────────────────
    #  close()
    # ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Cleanly shut down SUMO and release the TraCI port.

        Always call this (or use a context manager) to avoid leaving
        orphan SUMO processes that hold the port open.
        """
        self._sumo.stop()

    # ==================================================================
    #  PRIVATE — _get_observation()
    # ==================================================================

    def _get_observation(self, state: SimulationState) -> np.ndarray:
        """
        Build a 17-dimensional normalised observation from raw SUMO state.

        Layout:
            [0:4]   Queue lengths per direction (N, S, E, W)
            [4:8]   Waiting times per direction (N, S, E, W)
            [8:12]  Vehicle counts per direction (N, S, E, W)
            [12:16] Current phase as one-hot vector
            [16]    Emergency vehicle active flag

        All values are clipped to [0, 1] after normalisation.  Clipping
        is necessary because occasionally extreme values exceed our
        assumed maximums (e.g. a 35-car queue when MAX_QUEUE=30).
        Without clipping, the observation would violate the declared
        Box bounds, which can trigger assertion errors in some RL
        libraries and produce NaN gradients if a value reaches ∞.
        """
        obs = np.zeros(17, dtype=np.float32)

        # ── Indices 0–3: Queue lengths per direction ──────────────────
        for i, direction in enumerate(DIR_ORDER):
            lane_ids = DIRECTIONS[direction]
            raw_queue = sum(state.queue_length.get(lid, 0) for lid in lane_ids)
            obs[i] = np.clip(raw_queue / MAX_QUEUE, 0.0, 1.0)

        # ── Indices 4–7: Waiting times per direction ──────────────────
        for i, direction in enumerate(DIR_ORDER):
            lane_ids = DIRECTIONS[direction]
            raw_wait = sum(state.waiting_time.get(lid, 0.0) for lid in lane_ids)
            obs[4 + i] = np.clip(raw_wait / MAX_WAIT, 0.0, 1.0)

        # ── Indices 8–11: Vehicle counts per direction ────────────────
        for i, direction in enumerate(DIR_ORDER):
            lane_ids = DIRECTIONS[direction]
            raw_veh = sum(state.vehicle_count.get(lid, 0) for lid in lane_ids)
            obs[8 + i] = np.clip(raw_veh / MAX_VEHICLES, 0.0, 1.0)

        # ── Indices 12–15: Current phase as one-hot ───────────────────
        #
        # One-hot encoding: only the index matching the active phase is 1.0,
        # all others are 0.0.  This avoids the false ordinality problem
        # described in the module docstring.
        phase = state.signal_phase
        if 0 <= phase < NUM_PHASES:
            obs[12 + phase] = 1.0

        # ── Index 16: Emergency vehicle active ────────────────────────
        obs[16] = 1.0 if state.emergency_present else 0.0

        return obs

    # ==================================================================
    #  PRIVATE — _calculate_done()
    # ==================================================================

    def _calculate_done(self, sim_done: bool) -> Tuple[bool, bool]:
        """
        Determine whether the episode should end.

        Parameters
        ----------
        sim_done : bool
            True if SUMO reports no more vehicles expected (natural end).

        Returns
        -------
        terminated : bool
            True if the simulation reached a natural end (all vehicles
            have left the network).  In practice this rarely happens
            during training because our flows run for the full 3600 s.

            When terminated=True, the RL algorithm sets V(s_terminal) = 0
            because there is genuinely no future reward.

        truncated : bool
            True if we've hit our step budget (max_steps = 3600).
            The simulation COULD continue — there are still vehicles
            and phases to manage — but we cut the episode to keep
            training manageable and to expose the agent to many fresh
            resets.

            When truncated=True, the algorithm MUST bootstrap: it uses
            the Q-network to estimate V(s_truncated) rather than
            setting it to zero.  This is why the terminated/truncated
            distinction exists in Gymnasium — conflating them causes
            the agent to systematically undervalue late-episode states.
        """
        # Natural simulation end — all vehicles departed
        terminated = sim_done

        # Budget exhausted — 3600 steps = 3600 seconds = 1 hour
        truncated = self._step_count >= self._max_steps

        return terminated, truncated

    # ==================================================================
    #  close()
    # ==================================================================

    def close(self):
        """Clean up resources by stopping the TraCI bridge."""
        if hasattr(self, '_sumo') and self._sumo is not None:
            self._sumo.stop()
        super().close()

    # ==================================================================
    #  PRIVATE — _build_info()
    # ==================================================================

    def _build_info(self, state: SimulationState) -> Dict[str, Any]:
        """
        Build the info dict returned alongside the observation.

        This dict is NOT consumed by the RL algorithm.  It carries raw,
        un-normalised data for:
          • The TensorBoard / W&B logger
          • The FastAPI /metrics endpoint
          • The frontend dashboard

        Keys
        ----
        total_waiting_time : float
            Sum of waiting times across all 8 lanes (seconds).
            Used by TrafficAgent's TensorBoard callback.
        queue_per_direction : dict
            {N, S, E, W} → queue length (vehicles).
        waiting_per_direction : dict
            {N, S, E, W} → cumulative wait (seconds).
        vehicles_per_direction : dict
            {N, S, E, W} → vehicle count.
        total_queue_length : float
        total_vehicles : int
        signal_phase : int
        signal_state : str
        emergency_present : bool
        emergency_lanes : list
        sim_step : int
        episode_step : int
        """
        queues = {}
        waits = {}
        vehicles = {}
        for direction in DIR_ORDER:
            lane_ids = DIRECTIONS[direction]
            queues[direction] = sum(
                state.queue_length.get(lid, 0) for lid in lane_ids
            )
            waits[direction] = sum(
                state.waiting_time.get(lid, 0.0) for lid in lane_ids
            )
            vehicles[direction] = sum(
                state.vehicle_count.get(lid, 0) for lid in lane_ids
            )

        return {
            # Scalar summaries — consumed by traffic_agent.py callback
            "total_waiting_time": sum(waits.values()),
            "total_queue_length": sum(queues.values()),
            "total_vehicles": sum(vehicles.values()),

            # Per-direction detail — consumed by dashboard
            "queue_per_direction": queues,
            "waiting_per_direction": waits,
            "vehicles_per_direction": vehicles,

            # Signal state
            "signal_phase": state.signal_phase,
            "signal_state": state.signal_state,

            # Emergency status
            "emergency_present": state.emergency_present,
            "emergency_lanes": state.emergency_lanes,

            # Timing
            "sim_step": state.step,
            "episode_step": self._step_count,
        }
