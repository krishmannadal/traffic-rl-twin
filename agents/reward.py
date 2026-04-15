"""
reward.py — GPU-Aware Reward Functions for Traffic RL Twin
============================================================

This module defines modular, PyTorch-backed reward functions that can run
on CUDA if a GPU is available.  Using PyTorch tensors for reward computation
might seem like overkill for a scalar output, but it:

  1. Keeps the computation graph on-device if you ever want to differentiate
     through the reward (e.g. for reward-model fine-tuning or meta-learning).
  2. Makes batch reward computation trivial if you move to vectorised envs.
  3. Establishes a pattern that scales to more complex reward models.

REWARD DESIGN PHILOSOPHY
─────────────────────────
Getting the reward function right is arguably the HARDEST part of applied
RL.  A poorly designed reward creates **perverse incentives** — the agent
finds a way to maximise the number you gave it, but the resulting behaviour
is nothing like what you wanted.

Classic examples of wrong rewards causing unexpected emergent behaviour:
  • A boat-racing agent that discovered it could score more by spinning
    in circles to collect small bonuses than by finishing the race.
  • A robot hand that was rewarded for "grasping" and learned to slam the
    object so hard it stuck to the fingers via friction — technically a
    "grasp" but useless in practice.
  • A traffic agent rewarded ONLY for throughput that learns to give
    permanent green to the high-volume corridor and starves the side
    streets — throughput is high but some drivers wait 15 minutes.

In traffic control specifically, common reward pitfalls include:
  • Rewarding only throughput → agent ignores fairness (some lanes starve).
  • Rewarding only low waiting time → agent flickers signals every step
    because the *instantaneous* waiting drops when a green flashes briefly.
  • No stability term → the agent changes phases 60× per minute, which is
    physically impossible (actuators take 3–5 s) and dangerous for drivers.

WHAT IS REWARD SHAPING?
───────────────────────
Reward shaping is the practice of adding intermediate reward signals that
guide the agent toward good behaviour *before* it discovers the sparse
natural reward on its own.  Our stability_penalty is a classic example:

  • The natural reward (reduce waiting time) doesn't directly say
    "don't flicker."  An agent could theoretically discover that stable
    phases help — but it would take millions of steps of trial and error.

  • By explicitly penalising phase switches, we "shape" the reward
    landscape so the agent immediately understands that switching has a
    cost.  This dramatically speeds up learning.

  • The risk of reward shaping: if your shaped reward is *inconsistent*
    with the true objective, the agent optimises your shaping term
    instead of the real goal.  That's why we keep the stability weight
    small (0.1) — it nudges, it doesn't dominate.

WHY CLAMP REWARDS TO [-1, 1]?
──────────────────────────────
Neural networks (especially Q-networks) are sensitive to the scale of
their targets.  If rewards swing wildly (e.g. -500 to +500):
  • Gradient magnitudes explode → weight updates overshoot → divergence.
  • The replay buffer mixes old (small-reward) and new (large-reward)
    experiences, creating wildly inconsistent TD targets.
  • Value function saturates at extreme outputs, losing discriminative
    power in the "normal" range.

Clamping to [-1, 1] keeps Q-targets bounded, gradients well-scaled, and
training stable.  It's the same principle behind Huber loss and reward
clipping in Atari DQN.
"""

import torch
from typing import Any, Dict


# ──────────────────────────────────────────────────────────────────────
#  Device selection
# ──────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────
#  TrafficReward
# ──────────────────────────────────────────────────────────────────────

class TrafficReward:
    """
    Multi-component reward function for the traffic signal control agent.

    All intermediate calculations are performed as PyTorch tensors on the
    specified device (CPU or CUDA), then the final scalar is returned as
    a plain Python float for Gymnasium compatibility.

    Component Weights — Why These Numbers?
    ───────────────────────────────────────
    The four weights (0.4, 0.3, 0.2, 0.1) sum to 1.0.  Each component is
    individually clamped to its own weight range, so the total reward
    naturally falls in [-0.8, +0.2] without a hard global clamp.  This
    preserves gradient signal across all components simultaneously.

    Weight rationale (ordered by importance):

      0.4 — WAITING TIME PENALTY
        Waiting time is the primary Key Performance Indicator (KPI) for
        traffic signals in the real world.  Transportation engineers
        measure intersection performance by "average delay per vehicle."
        Giving this the highest weight ensures the agent prioritises
        what matters most: reducing how long people sit at red lights.

      0.3 — QUEUE LENGTH PENALTY
        Queue length is the second most critical metric.  Long queues
        can physically spill back and block upstream intersections
        (gridlock).  It's slightly less weighted than waiting time
        because a long queue that's moving slowly is better than a
        short queue that's completely stuck — waiting time captures
        the "stuck" aspect.

      0.2 — THROUGHPUT BONUS
        This is the only *positive* reward component.  It rewards the
        agent for actually moving vehicles through the intersection.
        Without it, the agent could learn to "cheat" by keeping all
        lights red (zero waiting time because no vehicles enter the
        intersection).  The throughput bonus ensures the agent can't
        game the system by refusing to serve traffic.

      0.1 — STABILITY PENALTY
        The smallest weight because it's a *shaping* term, not a true
        objective.  We want stable phases but not at the expense of
        serving traffic.  If this weight were 0.4, the agent would
        learn "never switch" — which keeps the penalty at zero but
        ignores congestion on non-green approaches.

    Parameters
    ----------
    device : str
        PyTorch device string — "cuda" or "cpu".
    """

    # Component weights — adjust these to shift the agent's priorities
    W_WAIT = 0.4
    W_QUEUE = 0.3
    W_THROUGHPUT = 0.2
    W_STABILITY = 0.1

    # Normalisation constants
    #
    # MAX_WAIT_PER_VEHICLE (not per lane!) — see compute() docstring for why.
    # 300 seconds = 5 minutes of one vehicle sitting at a red.  Any single
    # vehicle waiting longer than this is an extreme outlier.
    MAX_WAIT_PER_VEHICLE = 300.0
    MAX_QUEUE_PER_LANE = 30       # vehicles — capacity of one approach
    MAX_THROUGHPUT_PER_STEP = 20  # vehicles passing per simulation step
    NUM_LANES = 8                 # 2 lanes × 4 approaches

    def __init__(self, device: str = device):
        self.device = torch.device(device)
        self._prev_phase: int = -1  # track previous phase for stability

    def compute(self, state_dict: Dict[str, Any]) -> float:
        """
        Compute the scalar reward from raw SumoEnvironment.get_state() output.

        CRITICAL NORMALISATION FIX — PER-VEHICLE AVERAGE
        ─────────────────────────────────────────────────
        traci.lane.getWaitingTime() returns the SUM of cumulative waiting
        times for ALL vehicles currently on that lane.  If 15 vehicles
        each wait 60 seconds, the returned value is 900 seconds — NOT 60.

        The old code divided this raw 900s by MAX_WAIT_PER_LANE (300),
        producing a ratio of 3.0.  After weighting (×0.4 = -1.2) and
        clamping to [-1, 1], the result was always -1.0 during even
        moderate congestion.  This DESTROYS gradient signal:

        WHAT "GRADIENT SIGNAL" MEANS IN RL CONTEXT
        ───────────────────────────────────────────
        In DQN, the Q-network learns to predict expected future reward
        for each (state, action) pair.  Training computes:

            loss = (Q_predicted - Q_target)²
            Q_target = reward + γ × max_a' Q(next_state, a')

        If reward is always -1.0 for both "bad" and "catastrophic" states,
        Q_target is identical for both → the gradient of the loss w.r.t.
        the network weights is zero → the network CANNOT distinguish
        between them → no learning happens in the "bad-to-worse" region.

        This is called "reward saturation" — the signal is clipped flat,
        just like an audio amplifier that clips all loud sounds to the
        same maximum, making it impossible to tell a shout from a scream.

        The fix: divide lane total by vehicle count to get PER-VEHICLE
        average waiting time, THEN normalise against MAX_WAIT_PER_VEHICLE.
        Now 15 vehicles × 60s → avg = 60s → 60/300 = 0.2 (well within
        the useful range, far from saturation).

        WHY CLAMP COMPONENTS INDIVIDUALLY
        ──────────────────────────────────
        If we only clamp the TOTAL, a single extreme component (e.g.
        waiting time = -3.0) can push the total past the clamp floor,
        and improvements in OTHER components (queue, throughput) become
        invisible — the total stays at -1.0 regardless.

        By clamping each component to its OWN valid range FIRST:
          waiting_penalty  ∈ [-0.4,  0.0]
          queue_penalty    ∈ [-0.3,  0.0]
          throughput_bonus ∈ [ 0.0,  0.2]
          stability_penalty∈ [-0.1,  0.0]

        Every component ALWAYS contributes a meaningful gradient to the
        total.  The agent can see "queue improved but waiting got worse"
        because the queue component changes independently of waiting.
        The final sum naturally falls in [-0.8, +0.2] — no hard clamp
        on the total is needed, and no component can mask another.

        Parameters
        ----------
        state_dict : dict
            Must contain keys: "waiting_time" (dict), "queue_length" (dict),
            "vehicle_count" (dict), "signal_phase" (int).
            These come from SimulationState's attributes.

        Returns
        -------
        float
            Reward value in [-0.8, +0.2] (naturally bounded by
            per-component clamping, no hard clamp on the total).
        """
        # ── Extract raw values from the state dict ────────────────────
        waiting_times = list(state_dict.get("waiting_time", {}).values())
        queue_lengths = list(state_dict.get("queue_length", {}).values())
        vehicle_counts = list(state_dict.get("vehicle_count", {}).values())
        current_phase = state_dict.get("signal_phase", 0)

        # ── Move to PyTorch tensors on device ─────────────────────────
        wait_tensor = torch.tensor(
            waiting_times, dtype=torch.float32, device=self.device
        )
        queue_tensor = torch.tensor(
            queue_lengths, dtype=torch.float32, device=self.device
        )
        veh_tensor = torch.tensor(
            vehicle_counts, dtype=torch.float32, device=self.device
        )

        # ── Component 1: Waiting time penalty ─────────────────────────
        # PER-VEHICLE AVERAGE waiting time, not raw lane total.
        #
        # traci.lane.getWaitingTime() returns sum of ALL vehicles' waits
        # on that lane.  We divide by the vehicle count on each lane to
        # get the average wait experienced by a single vehicle.
        #
        # max(count, 1) prevents division by zero on empty lanes.
        # Empty lanes contribute 0/1 = 0 waiting time, which is correct.
        safe_counts = torch.clamp(veh_tensor, min=1.0)
        avg_wait_per_vehicle = wait_tensor / safe_counts  # per-lane avg

        # Mean across all lanes, normalised against the per-vehicle max.
        # Result before weighting: [0.0, 1.0+] (can exceed 1.0 in extreme
        # cases, which is why we clamp the component after weighting).
        wait_ratio = avg_wait_per_vehicle.mean() / self.MAX_WAIT_PER_VEHICLE

        # Weight and clamp to this component's valid range: [-0.4, 0.0]
        waiting_penalty = torch.clamp(
            -self.W_WAIT * wait_ratio, min=-self.W_WAIT, max=0.0
        )

        # ── Component 2: Queue length penalty ─────────────────────────
        # Mean queue per lane, normalised by lane capacity.
        queue_ratio = queue_tensor.mean() / self.MAX_QUEUE_PER_LANE

        # Clamp to [-0.3, 0.0]
        queue_penalty = torch.clamp(
            -self.W_QUEUE * queue_ratio, min=-self.W_QUEUE, max=0.0
        )

        # ── Component 3: Throughput bonus ─────────────────────────────
        # Total vehicles across all lanes as a proxy for throughput.
        # More vehicles being served = higher bonus.
        throughput_ratio = veh_tensor.sum() / self.MAX_THROUGHPUT_PER_STEP

        # Clamp to [0.0, 0.2] — prevents runaway positive reward
        # during traffic surges from dominating the signal
        throughput_bonus = torch.clamp(
            self.W_THROUGHPUT * throughput_ratio, min=0.0, max=self.W_THROUGHPUT
        )

        # ── Component 4: Stability penalty ────────────────────────────
        # Binary: -0.1 if phase changed from the previous step, else 0.
        # Already naturally in [-0.1, 0.0], no clamping needed.
        #
        # This is REWARD SHAPING: we inject domain knowledge ("phase
        # switches are expensive") directly into the reward so the agent
        # doesn't have to discover it through millions of bad experiences.
        phase_changed = (
            current_phase != self._prev_phase
            and self._prev_phase != -1  # don't penalise the very first step
        )
        stability_penalty = torch.tensor(
            -self.W_STABILITY if phase_changed else 0.0,
            dtype=torch.float32,
            device=self.device,
        )

        # ── Combine ──────────────────────────────────────────────────
        # Each component is already clamped to its own range, so the
        # total naturally falls in [-0.8, +0.2].  No hard clamp needed.
        # Every component's gradient flows independently to the Q-network.
        total = waiting_penalty + queue_penalty + throughput_bonus + stability_penalty

        # ── Update state for next step ────────────────────────────────
        self._prev_phase = current_phase

        # Return plain float for Gymnasium compatibility
        return total.item()

    def explain(self, state_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute each reward component separately and return as a dict.

        Returns both RAW (pre-clamp) and NORMALISED (post-clamp) values
        for each component, enabling dashboard visualisation of how close
        each component is to saturation.

        Useful for:
          • TensorBoard / W&B dashboards — plot each component over time
            to see which aspect of the reward the agent is improving.
          • Debugging — if total reward is high but waiting_penalty is
            terrible, the throughput bonus might be masking a problem.
          • Reward tuning — compare component magnitudes to decide if
            weights need adjustment.
          • Saturation detection — if raw ≈ clamped for all components,
            the reward is well-scaled.  If raw >> clamped, the normalisation
            constants need adjustment.

        Parameters
        ----------
        state_dict : dict
            Same format as compute().

        Returns
        -------
        dict with keys:
            Raw values (before per-component clamping):
              "waiting_penalty_raw", "queue_penalty_raw",
              "throughput_bonus_raw", "stability_penalty"

            Clamped values (what actually enters the reward sum):
              "waiting_penalty", "queue_penalty",
              "throughput_bonus", "stability_penalty"

            Diagnostic:
              "avg_wait_per_vehicle" — the per-vehicle average wait (seconds)
              "total" — sum of clamped components (final reward)
        """
        waiting_times = list(state_dict.get("waiting_time", {}).values())
        queue_lengths = list(state_dict.get("queue_length", {}).values())
        vehicle_counts = list(state_dict.get("vehicle_count", {}).values())
        current_phase = state_dict.get("signal_phase", 0)

        wait_tensor = torch.tensor(
            waiting_times, dtype=torch.float32, device=self.device
        )
        queue_tensor = torch.tensor(
            queue_lengths, dtype=torch.float32, device=self.device
        )
        veh_tensor = torch.tensor(
            vehicle_counts, dtype=torch.float32, device=self.device
        )

        # Per-vehicle average waiting time
        safe_counts = torch.clamp(veh_tensor, min=1.0)
        avg_wait_per_vehicle = wait_tensor / safe_counts
        avg_wait_scalar = avg_wait_per_vehicle.mean().item()

        wait_ratio = avg_wait_per_vehicle.mean() / self.MAX_WAIT_PER_VEHICLE
        queue_ratio = queue_tensor.mean() / self.MAX_QUEUE_PER_LANE
        throughput_ratio = veh_tensor.sum() / self.MAX_THROUGHPUT_PER_STEP

        # Raw (pre-clamp) values
        wp_raw = (-self.W_WAIT * wait_ratio).item()
        qp_raw = (-self.W_QUEUE * queue_ratio).item()
        tb_raw = (self.W_THROUGHPUT * throughput_ratio).item()

        # Clamped values
        wp = max(-self.W_WAIT, min(0.0, wp_raw))
        qp = max(-self.W_QUEUE, min(0.0, qp_raw))
        tb = max(0.0, min(self.W_THROUGHPUT, tb_raw))

        phase_changed = (
            current_phase != self._prev_phase
            and self._prev_phase != -1
        )
        sp = -self.W_STABILITY if phase_changed else 0.0

        total = wp + qp + tb + sp

        # NOTE: explain() does NOT update self._prev_phase so it can be
        # called alongside compute() without double-counting phase changes.

        return {
            # Raw values (before per-component clamping)
            "waiting_penalty_raw": wp_raw,
            "queue_penalty_raw": qp_raw,
            "throughput_bonus_raw": tb_raw,
            # Clamped values (what enters the reward sum)
            "waiting_penalty": wp,
            "queue_penalty": qp,
            "throughput_bonus": tb,
            "stability_penalty": sp,
            # Diagnostic
            "avg_wait_per_vehicle": avg_wait_scalar,
            "total": total,
        }

    def reset(self) -> None:
        """Reset internal state (call at the start of each episode)."""
        self._prev_phase = -1


# ──────────────────────────────────────────────────────────────────────
#  EmergencyReward
# ──────────────────────────────────────────────────────────────────────

class EmergencyReward:
    """
    Reward function for the emergency-vehicle green-corridor agent.

    This is a SEPARATE reward from TrafficReward because the emergency
    agent has fundamentally different objectives:
      • TrafficReward minimises waiting time for ALL vehicles equally.
      • EmergencyReward prioritises ONE vehicle (the ambulance/firetruck)
        while trying to minimise collateral disruption to everyone else.

    These objectives conflict — giving a permanent green corridor to the
    emergency vehicle causes massive delays on cross-streets.  The two
    reward components balance this trade-off:

      time_saved_bonus  (+2.0 max):
        How much faster did the emergency vehicle arrive compared to
        a baseline (no preemption)?  Scaled by baseline time so the
        bonus is proportional to the improvement.

      disruption_penalty (-0.5 × disruption):
        How much did the green corridor hurt normal traffic?
        `traffic_disruption` is a scalar ∈ [0, 1] representing the
        fractional increase in average delay for non-emergency vehicles.

    The wider clamp range [-2, +2] (vs [-1, +1] for TrafficReward)
    reflects the higher stakes: saving a life is worth more than a
    marginal throughput improvement, but we still can't ignore the
    disruption entirely.

    Parameters
    ----------
    device : str
        PyTorch device string — "cuda" or "cpu".
    """

    def __init__(self, device: str = device):
        self.device = torch.device(device)

    def compute(
        self,
        emergency_travel_time: float,
        baseline_time: float,
        traffic_disruption: float,
        is_arrival: bool = False,
    ) -> float:
        """
        Compute the emergency-corridor reward.

        Reward Structure:
          1. Step Penalty (-0.05): Constant pressure to arrive quickly.
          2. Disruption Penalty (-1.0 max): Based on traffic wait times.
          3. Arrival Bonus (+2.0 max): Awarded only on the step of arrival,
             proportional to time saved vs baseline.

        Parameters
        ----------
        emergency_travel_time : float
            Actual travel time of the emergency vehicle (seconds).
        baseline_time : float
            Expected travel time without any signal preemption (seconds).
        traffic_disruption : float
            Normalised disruption score ∈ [0, 1].
        is_arrival : bool
            True if the vehicle just left the network.

        Returns
        -------
        float
            Reward value clamped to [-2.5, 2.5].
        """
        # ── Step Penalty ──────────────────────────────────────────────
        # Small constant penalty to encourage finishing the task.
        reward = torch.tensor(-0.05, dtype=torch.float32, device=self.device)

        # ── Disruption Penalty ────────────────────────────────────────
        # Scales with traffic build-up on cross-streets.
        disruption = torch.tensor(
            -1.0 * traffic_disruption,
            dtype=torch.float32,
            device=self.device,
        )
        reward += disruption

        # ── Arrival Bonus ─────────────────────────────────────────────
        # Awarded only at the very end.
        if is_arrival:
            safe_baseline = max(baseline_time, 1.0)
            # Bonus = 2.0 * (Time Saved / Baseline)
            # Max +2.0 if B=60 and T=0 (impossible, but theoretical ceiling)
            bonus = torch.tensor(
                2.0 * (safe_baseline - emergency_travel_time) / safe_baseline,
                dtype=torch.float32,
                device=self.device,
            )
            reward += bonus

        return torch.clamp(reward, min=-2.5, max=2.5).item()

    def explain(
        self,
        emergency_travel_time: float,
        baseline_time: float,
        traffic_disruption: float,
        is_arrival: bool = False,
    ) -> Dict[str, float]:
        """Return each component separately for logging."""
        step_penalty = -0.05
        disruption = -1.0 * traffic_disruption
        arrival_bonus = 0.0

        if is_arrival:
            safe_baseline = max(baseline_time, 1.0)
            arrival_bonus = 2.0 * (safe_baseline - emergency_travel_time) / safe_baseline

        total = step_penalty + disruption + arrival_bonus
        total_clamped = max(-2.5, min(2.5, total))

        return {
            "step_penalty": step_penalty,
            "disruption_penalty": disruption,
            "arrival_bonus": arrival_bonus,
            "total_pre_clamp": total,
            "total": total_clamped,
        }
