"""
emergency_env.py — Emergency Vehicle Green-Corridor Environment
=================================================================

This module extends TrafficEnv with the ability to handle emergency
vehicle scenarios.  When an emergency vehicle (ambulance, fire truck)
enters the network, this environment overrides normal signal control
to create a "green corridor" — a sequence of green phases that clears
the path for the emergency vehicle.

WHAT IS A GREEN CORRIDOR?
─────────────────────────
Physically, a green corridor means:

  1. The traffic light ahead of the emergency vehicle turns green for
     the lane the emergency vehicle is approaching on.
  2. All conflicting movements (cross-traffic) get red.
  3. As the emergency vehicle progresses through its route, each
     successive traffic light repeats this: green for the emergency
     vehicle's lane, red for everyone else.

The result is an uninterrupted "wave of green" that the emergency
vehicle rides through the network.  In real cities this is called
"signal preemption" and is implemented via sensors (Opticom, GPS)
that detect emergency vehicles approaching an intersection.

The trade-off: every second the corridor is active, cross-traffic is
blocked.  Longer corridors cause more disruption.  The goal is to
minimise emergency travel time while limiting collateral delay.

WHY RULE-BASED CORRIDOR BEFORE A LEARNED APPROACH?
───────────────────────────────────────────────────
We start with a simple greedy rule ("green the edge the vehicle is on,
red everything else") for several practical reasons:

  1. SAFETY BASELINE — Emergency vehicle preemption is safety-critical.
     A trained RL policy might fail catastrophically on an unseen
     scenario (distribution shift).  The rule-based approach is
     predictable and verifiably correct: if the vehicle is on edge X,
     edge X gets green.  Period.

  2. SAMPLE EFFICIENCY — Emergency events are RARE (by design).
     Training an RL agent on rare events requires millions of episodes,
     most of which contain zero emergencies.  The agent would need a
     massive amount of data to learn what the rule gives us for free.

  3. PROGRESSIVE COMPLEXITY — The project roadmap is:
       Phase 1: Rule-based corridor (this file)  ← WE ARE HERE
       Phase 2: RL agent that learns WHEN to activate/deactivate
                the corridor and how aggressively to preempt.
       Phase 3: Full multi-agent system where the traffic agent and
                emergency agent negotiate via a shared reward.

  4. COMPARISON BASELINE — You can't evaluate a learned policy without
     comparing it to something.  The rule-based corridor IS that
     baseline.  If your RL agent can't beat the greedy rule, it's
     not worth deploying.

HOW TWO AGENTS HAND OFF CONTROL
────────────────────────────────
The architecture uses a simple priority-based handoff:

  Normal operation:
    TrafficEnv.step(action) → TrafficAgent's DQN chooses the phase
    └─ emergency_active = False

  Emergency detected / triggered:
    EmergencyEnv.step(action) → ignores the action parameter entirely
    └─ emergency_active = True
    └─ green corridor logic takes over
    └─ the DQN's output is discarded (it still runs, but has no effect)

  Emergency resolved:
    deactivate_emergency() → sets emergency_active = False
    └─ next call to step(action) routes back to TrafficEnv logic
    └─ DQN regains control

This is conceptually similar to how real-world preemption systems work:
the preemption controller has ABSOLUTE PRIORITY over the normal timing
plan.  When preemption ends, the controller hands back to the normal
cycle, typically after a brief recovery period.

WHY PRE-CALCULATE BASELINE TRAVEL TIME?
────────────────────────────────────────
The EmergencyReward needs to know "how much did the corridor help?"
To answer that, we need the counterfactual: how long WOULD the
emergency vehicle have taken WITHOUT any signal preemption?

We estimate this using free-flow travel time:
    baseline = route_length / emergency_vehicle_max_speed

This is a simplification (real baseline would require a parallel
simulation without preemption), but it serves as a reasonable upper
bound on the "best possible without help" time.  The reward then
measures how much FASTER the vehicle arrived compared to this baseline.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TraCI is imported via SumoEnvironment, but we need direct access
# for emergency vehicle spawning and route queries.
import traci

from simulation.traffic_env import TrafficEnv, NUM_PHASES
from simulation.environment import (
    SumoEnvironment,
    SimulationState,
    TL_JUNCTION_ID,
)
from agents.reward import EmergencyReward


class EmergencyEnv(TrafficEnv):
    """
    Extension of TrafficEnv that handles emergency vehicle scenarios.

    When an emergency vehicle is active, this environment overrides
    the RL agent's signal decisions with a rule-based green corridor.
    When no emergency is active, it behaves identically to TrafficEnv.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to SUMO .sumocfg file.
    port : int
        TraCI TCP port.
    use_gui : bool
        Launch SUMO-GUI.
    max_steps : int
        Episode truncation limit.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        net_file: Optional[str] = None,
        port: int = 8813,
        use_gui: bool = False,
        max_steps: int = 3600,
    ):
        super().__init__(
            config_path=config_path,
            net_file=net_file,
            port=port,
            use_gui=use_gui,
            max_steps=max_steps,
        )

        # ── Emergency state tracking ──────────────────────────────────
        self.emergency_active: bool = False
        self.emergency_vehicle_id: str = ""
        self.emergency_route: List[str] = []
        self.emergency_start_time: float = 0.0
        self.baseline_travel_time: float = 0.0

        # ── Emergency reward function ─────────────────────────────────
        self._emergency_reward_fn = EmergencyReward()

        # ── Logging ───────────────────────────────────────────────────
        self._emergency_log: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────
    #  reset() — extend parent to clear emergency state
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and clear all emergency state."""
        obs, info = super().reset(seed=seed, options=options)

        self.emergency_active = False
        self.emergency_vehicle_id = ""
        self.emergency_route = []
        self.emergency_start_time = 0.0
        self.baseline_travel_time = 0.0
        self._emergency_log = []

        return obs, info

    # ──────────────────────────────────────────────────────────────────
    #  trigger_emergency()
    # ──────────────────────────────────────────────────────────────────

    def trigger_emergency(
        self,
        vehicle_id: str,
        origin_edge: str,
        destination_edge: str,
    ) -> float:
        """
        Spawn an emergency vehicle and activate the green corridor.

        This method:
          1. Adds a new emergency vehicle to the running SUMO simulation
             using TraCI's vehicle.add() — no need to restart the sim.
          2. Finds the shortest route from origin to destination.
          3. Estimates a baseline travel time (free-flow, no preemption).
          4. Sets emergency_active = True so step() switches to corridor mode.

        Parameters
        ----------
        vehicle_id : str
            Unique ID for the emergency vehicle (e.g. "emergency_003").
        origin_edge : str
            Starting edge ID (e.g. "south_to_center").
        destination_edge : str
            Target edge ID (e.g. "center_to_north").

        Returns
        -------
        float
            Estimated baseline travel time in seconds (without preemption).
        """
        # ── Find the route ────────────────────────────────────────────
        # traci.simulation.findRoute() uses Dijkstra's algorithm on the
        # SUMO network to find the shortest path between two edges.
        # It returns a Stage object with .edges (list of edge IDs) and
        # .travelTime (estimated travel time at current speeds).
        route_result = traci.simulation.findRoute(
            origin_edge, destination_edge, vType="emergency_vehicle"
        )
        route_edges = list(route_result.edges)

        if not route_edges:
            print(
                f"⚠ Could not find route from {origin_edge} → "
                f"{destination_edge}. Emergency not triggered."
            )
            return 0.0

        # ── Add the vehicle to the live simulation ────────────────────
        # traci.vehicle.add() injects a vehicle into the network at the
        # current simulation time.  We specify:
        #   routeID=""   → we'll set the route manually via changeTarget
        #   typeID       → must match a <vType> in routes.rou.xml
        #   departLane   → "best" lets SUMO pick the least-congested lane
        #   departSpeed  → "max" starts at the vehicle's max speed
        route_id = f"emergency_route_{vehicle_id}"

        # Create a route object in SUMO first
        traci.route.add(route_id, route_edges)

        # Then add the vehicle using that route
        traci.vehicle.add(
            vehID=vehicle_id,
            routeID=route_id,
            typeID="emergency_vehicle",
            departLane="best",
            departSpeed="max",
        )

        # ── Calculate baseline travel time ────────────────────────────
        # We estimate baseline as: total_route_length / max_speed.
        #
        # This is the time it would take if the vehicle drove at full
        # speed with no stops — essentially the best case WITHOUT signal
        # preemption (the vehicle would still face some red lights in
        # practice, so real baseline would be longer).
        #
        # A more accurate baseline would require running a parallel
        # simulation without preemption, but that doubles compute cost.
        # This approximation is good enough for reward computation.
        total_length = 0.0
        for edge_id in route_edges:
            # traci.edge.getLength() returns the length of an edge in meters.
            # For multi-lane edges, all lanes have the same length.
            total_length += traci.edge.getLength(edge_id)

        # Emergency vehicle max speed from vType definition (20 m/s)
        emergency_max_speed = 20.0
        try:
            emergency_max_speed = traci.vehicletype.getMaxSpeed(
                "emergency_vehicle"
            )
        except traci.TraCIException:
            pass  # fall back to default 20 m/s

        # Baseline = distance / speed  (with a small buffer for intersection delay)
        # Add 5 seconds per intersection crossing as a rough red-light penalty
        num_intersections = max(len(route_edges) - 1, 0)
        self.baseline_travel_time = (
            total_length / emergency_max_speed
        ) + (num_intersections * 5.0)

        # ── Activate emergency mode ───────────────────────────────────
        self.emergency_active = True
        self.emergency_vehicle_id = vehicle_id
        self.emergency_route = route_edges
        self.emergency_start_time = traci.simulation.getTime()

        print(
            f"🚨 Emergency triggered: {vehicle_id}\n"
            f"   Route       : {' → '.join(route_edges)}\n"
            f"   Route length: {total_length:.0f} m\n"
            f"   Baseline ETA: {self.baseline_travel_time:.1f} s"
        )

        return self.baseline_travel_time

    # ──────────────────────────────────────────────────────────────────
    #  get_green_corridor()
    # ──────────────────────────────────────────────────────────────────

    def get_green_corridor(
        self, route: List[str]
    ) -> List[Tuple[str, int]]:
        """
        Determine which traffic lights need green and which phase to set.

        Takes the emergency vehicle's current route and returns a list of
        (traffic_light_id, phase_index) pairs that will give green to
        each edge on the route.

        Algorithm (simple greedy):
          For each traffic light that controls an edge on the route:
            1. Look at all phases in the TL program.
            2. Find the phase that gives GREEN to the most links
               coming from one of the route's edges.
            3. Return that (tl_id, phase_index) pair.

        This is a greedy heuristic — it does NOT consider timing or
        coordination between successive lights.  A learned agent could
        improve on this by anticipating the vehicle's arrival time and
        pre-clearing intersections.

        Parameters
        ----------
        route : list of str
            Edge IDs in the emergency vehicle's remaining route.

        Returns
        -------
        list of (traffic_light_id, phase_index) tuples
        """
        corridor: List[Tuple[str, int]] = []

        # Get all traffic lights in the network
        # traci.trafficlight.getIDList() returns every TL junction ID.
        tl_ids = traci.trafficlight.getIDList()

        for tl_id in tl_ids:
            # getControlledLinks(tl_id) returns a list where each element
            # corresponds to one signal link index.  Each element is a list
            # of (incoming_lane, outgoing_lane, internal_lane) tuples.
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)

            # Check if any link feeds from an edge on the emergency route
            route_link_indices = []
            for link_idx, link_group in enumerate(controlled_links):
                if not link_group:
                    continue
                incoming_lane = link_group[0][0]
                # Extract edge from lane ID (drop "_0" or "_1" suffix)
                incoming_edge = "_".join(incoming_lane.split("_")[:-1])
                if incoming_edge in route:
                    route_link_indices.append(link_idx)

            if not route_link_indices:
                continue  # this TL doesn't control any route edges

            # Find the phase that gives green to the most route links
            logics = traci.trafficlight.getAllProgramLogics(tl_id)
            if not logics:
                continue

            best_phase = 0
            best_score = -1

            for phase_idx, phase in enumerate(logics[0].phases):
                # phase.state is a string like "GGGggrrrrrGGGgg..."
                # Count how many route links get 'G' or 'g' (green) in this phase
                score = sum(
                    1
                    for li in route_link_indices
                    if li < len(phase.state) and phase.state[li] in ("G", "g")
                )
                if score > best_score:
                    best_score = score
                    best_phase = phase_idx

            corridor.append((tl_id, best_phase))

        return corridor

    # ──────────────────────────────────────────────────────────────────
    #  apply_corridor()
    # ──────────────────────────────────────────────────────────────────

    def apply_corridor(
        self, corridor: List[Tuple[str, int]]
    ) -> None:
        """
        Apply green corridor phase changes to SUMO via TraCI.

        For each (traffic_light_id, phase_index) pair in the corridor:
          • Set the TL to the specified phase.
          • Override the phase duration to hold it until we say otherwise.

        Only the traffic lights on the emergency route are affected.
        All other intersections continue their normal programmatic
        timing — this is important because needlessly disrupting
        unrelated intersections increases global delay without
        helping the emergency vehicle.

        Parameters
        ----------
        corridor : list of (str, int)
            Output of get_green_corridor().
        """
        for tl_id, phase_idx in corridor:
            # setPhase() immediately switches to the requested phase.
            # This is the raw TraCI call that physically changes the light.
            traci.trafficlight.setPhase(tl_id, phase_idx)

            # setPhaseDuration() overrides the automatic timer.
            # We set a very large value so the phase holds until
            # we explicitly change it again (agent-controlled timing).
            traci.trafficlight.setPhaseDuration(tl_id, 1_000_000)

    # ──────────────────────────────────────────────────────────────────
    #  deactivate_emergency()
    # ──────────────────────────────────────────────────────────────────

    def deactivate_emergency(self) -> Dict[str, float]:
        """
        Deactivate the green corridor and return control to TrafficEnv.

        This method:
          1. Logs the emergency travel time vs baseline.
          2. Resets the traffic light to its default program so the
             normal phase cycle resumes.
          3. Sets emergency_active = False.

        Returns
        -------
        dict with keys:
            emergency_travel_time : actual seconds to traverse
            baseline_travel_time  : estimated seconds without preemption
            time_saved            : baseline - actual (positive = faster)
            time_saved_pct        : percentage improvement
        """
        # Calculate actual emergency travel time
        current_time = traci.simulation.getTime()
        emergency_travel_time = current_time - self.emergency_start_time

        time_saved = self.baseline_travel_time - emergency_travel_time
        time_saved_pct = (
            (time_saved / self.baseline_travel_time * 100.0)
            if self.baseline_travel_time > 0
            else 0.0
        )

        # ── Restore normal signal operation ───────────────────────────
        # setProgram("0") resets the TL to its default programmatic
        # timing cycle (the one defined in map.net.xml).  This is how
        # we "hand back" control from the emergency corridor to the
        # normal TrafficEnv / DQN agent.
        try:
            traci.trafficlight.setProgram(TL_JUNCTION_ID, "0")
        except traci.TraCIException:
            pass  # TL may already be on program "0"

        # ── Log the event ─────────────────────────────────────────────
        log_entry = {
            "vehicle_id": self.emergency_vehicle_id,
            "route": self.emergency_route,
            "emergency_travel_time": emergency_travel_time,
            "baseline_travel_time": self.baseline_travel_time,
            "time_saved": time_saved,
            "time_saved_pct": time_saved_pct,
        }
        self._emergency_log.append(log_entry)

        print(
            f"✅ Emergency resolved: {self.emergency_vehicle_id}\n"
            f"   Travel time : {emergency_travel_time:.1f} s "
            f"(baseline: {self.baseline_travel_time:.1f} s)\n"
            f"   Time saved  : {time_saved:.1f} s ({time_saved_pct:.1f}%)"
        )

        # ── Reset emergency state ─────────────────────────────────────
        self.emergency_active = False
        self.emergency_vehicle_id = ""
        self.emergency_route = []
        self.emergency_start_time = 0.0
        self.baseline_travel_time = 0.0

        return log_entry

    # ──────────────────────────────────────────────────────────────────
    #  step(action) — OVERRIDE
    # ──────────────────────────────────────────────────────────────────

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Override TrafficEnv.step() to handle emergency vehicle scenarios.

        Control flow:
        ─────────────
        If emergency_active is TRUE:
            The `action` parameter from the DQN agent is IGNORED.
            Instead, we:
              1. Check if the emergency vehicle is still in the network.
              2. Compute the green corridor for its current remaining route.
              3. Apply the corridor phases via TraCI.
              4. Advance the simulation.
              5. Compute reward using EmergencyReward.
            If the vehicle has left the network (arrived or despawned),
            we automatically call deactivate_emergency().

        If emergency_active is FALSE:
            We call the parent TrafficEnv.step(action) as normal.
            The DQN agent's action is applied and normal traffic flow
            reward is computed.

        This priority-based handoff mirrors real-world preemption systems:
        when the preemption trigger fires, the normal timing plan is
        suspended unconditionally.  The normal controller doesn't get
        to "vote" — emergency preemption has absolute priority.
        """

        if not self.emergency_active:
            # ── Normal mode: delegate to TrafficEnv ───────────────────
            return super().step(action)

        # ── Emergency mode: override signal control ───────────────────

        # 1. Check if emergency vehicle is still in the network
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_still_present = self.emergency_vehicle_id in vehicle_ids

        if not vehicle_still_present:
            # Emergency vehicle has just left the network - compute FINAL reward
            # (including arrival bonus) before deactivating.
            current_time = traci.simulation.getTime()
            elapsed = current_time - self.emergency_start_time
            
            # Calculate disruption one last time
            total_wait = sum(self._sumo.get_last_state().waiting_time.values())
            max_possible_wait = 300.0 * 8
            traffic_disruption = min(total_wait / max_possible_wait, 1.0)
            
            arrival_reward = self._emergency_reward_fn.compute(
                emergency_travel_time=elapsed,
                baseline_time=max(self.baseline_travel_time, 1.0),
                traffic_disruption=traffic_disruption,
                is_arrival=True
            )
            
            # Deactivate and revert to normal control
            self.deactivate_emergency()
            
            # Take a normal step but override its reward with our completion bonus
            obs, _, terminated, truncated, info = super().step(action)
            info["emergency_just_resolved"] = True
            info["reward_components"] = self._emergency_reward_fn.explain(
                emergency_travel_time=elapsed,
                baseline_time=max(self.baseline_travel_time, 1.0),
                traffic_disruption=traffic_disruption,
                is_arrival=True
            )
            return obs, arrival_reward, terminated, truncated, info
        # 2. Get the vehicle's current position and remaining route
        try:
            current_edge = traci.vehicle.getRoadID(self.emergency_vehicle_id)
            route_index = traci.vehicle.getRouteIndex(
                self.emergency_vehicle_id
            )
            full_route = traci.vehicle.getRoute(self.emergency_vehicle_id)
            remaining_route = list(full_route[route_index:])
        except traci.TraCIException:
            # Vehicle may have just despawned between checks
            self.deactivate_emergency()
            return super().step(action)

        # 3. Compute green corridor for remaining route
        corridor = self.get_green_corridor(remaining_route)

        # 4. Apply corridor phases
        self.apply_corridor(corridor)

        # 5. Advance simulation (use the corridor-determined phase for
        #    our junction, ignoring the DQN's action)
        corridor_phase = 0
        for tl_id, phase_idx in corridor:
            if tl_id == TL_JUNCTION_ID:
                corridor_phase = phase_idx
                break

        # Call SumoEnvironment.step() directly (NOT super().step() which
        # would apply the DQN's action).
        sim_state, sim_done = self._sumo.step(corridor_phase)
        self._step_count += 1

        # 6. Build observation
        observation = self._get_observation(sim_state)

        # 7. Compute emergency reward
        #    Traffic disruption = normalised increase in waiting time
        total_wait = sum(sim_state.waiting_time.values())
        # Rough normalisation: max expected wait across all lanes
        max_possible_wait = 300.0 * 8  # 300s × 8 lanes
        traffic_disruption = min(total_wait / max_possible_wait, 1.0)

        # For the intermediate steps (vehicle still en route), we use
        # a partial reward based on current disruption
        elapsed = traci.simulation.getTime() - self.emergency_start_time
        reward = self._emergency_reward_fn.compute(
            emergency_travel_time=elapsed,
            baseline_time=max(self.baseline_travel_time, 1.0),
            traffic_disruption=traffic_disruption,
            is_arrival=False
        )

        # 8. Check termination
        terminated = sim_done
        truncated = self._step_count >= self._max_steps

        # 9. Build info dict
        info = self._build_info(sim_state)
        info["emergency_active"] = True
        info["emergency_vehicle_id"] = self.emergency_vehicle_id
        info["emergency_remaining_route"] = remaining_route
        info["emergency_elapsed_time"] = elapsed
        info["emergency_baseline_time"] = self.baseline_travel_time
        info["corridor"] = corridor
        info["reward_components"] = self._emergency_reward_fn.explain(
            emergency_travel_time=elapsed,
            baseline_time=max(self.baseline_travel_time, 1.0),
            traffic_disruption=traffic_disruption,
            is_arrival=False
        )

        self._last_state = sim_state

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────
    #  _get_observation() — OVERRIDE
    # ──────────────────────────────────────────────────────────────────

    def _get_observation(self, state: SimulationState) -> np.ndarray:
        """
        Build the 17-dim observation, overriding index 16 to reflect
        the actual emergency_active state tracked by this class.

        The parent TrafficEnv reads state.emergency_present from SUMO
        (which detects emergency vehicles by type).  But EmergencyEnv
        has its own tracking (self.emergency_active) which is more
        reliable because:
          • It's set explicitly by trigger_emergency() and cleared by
            deactivate_emergency() — no race conditions.
          • It persists even if the vehicle briefly enters an internal
            junction lane (where SUMO's detection can miss it).
        """
        obs = super()._get_observation(state)

        # Override index 16 with our own tracking flag
        obs[16] = 1.0 if self.emergency_active else 0.0

        return obs

    # ──────────────────────────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def emergency_log(self) -> List[Dict[str, Any]]:
        """Return the log of all emergency events this episode."""
        return self._emergency_log
