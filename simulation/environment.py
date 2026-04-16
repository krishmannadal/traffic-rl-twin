"""
environment.py — SUMO-TraCI Interface for Traffic RL Twin
==========================================================

This module provides the low-level bridge between Python and the SUMO
traffic simulator via the TraCI (Traffic Control Interface) protocol.

TraCI works as a client-server protocol:
  • SUMO runs as the *server* (opens a TCP socket on a given port)
  • This Python class is the *client* that sends commands and reads back
    state every simulation step

Port 8813 is used so multiple SUMO instances can run on a machine without
clashing with the default port 8813 (you can change it in start()).
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# TraCI is bundled with SUMO.  The SUMO_HOME env variable must point to
# your SUMO installation directory (e.g. C:\Program Files\Eclipse\Sumo).
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set.\n"
        "Set it to your SUMO installation directory, e.g.:\n"
        "  Windows: setx SUMO_HOME \"C:\\Program Files\\Eclipse\\Sumo\"\n"
        "  Linux  : export SUMO_HOME=/usr/share/sumo"
    )

import traci
import traci.constants as tc

# ──────────────────────────────────────────────────────────────────────
#  Config paths
# ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_CFG_PATH = _PROJECT_ROOT / "simulation" / "sumo_configs" / "simulation.sumocfg"

# ID of the traffic light junction as defined in map.nod.xml
TL_JUNCTION_ID = "center"

# Emergency vehicles are identified by a vehicle type prefix
EMERGENCY_TYPE_PREFIX = "emergency"


# ──────────────────────────────────────────────────────────────────────
#  Dataclass for a clean state snapshot
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimulationState:
    """
    A structured snapshot of the intersection state at the current
    simulation step.

    Attributes
    ----------
    vehicle_count   : dict mapping lane_id → number of vehicles on that lane
    queue_length    : dict mapping lane_id → number of halting vehicles
                      (speed < 0.1 m/s) — this is the "queue"
    waiting_time    : dict mapping lane_id → accumulated waiting time (s)
                      for all vehicles on that lane
    signal_phase    : int — index of the currently active traffic-light phase
    signal_state    : str — raw SUMO phase string, e.g. "GGGggrrrrrGGGgg..."
    step            : int — current simulation step number
    emergency_present : bool — True when ≥1 emergency vehicle is in the net
    emergency_lanes : list of lane IDs containing emergency vehicles
    """

    vehicle_count: Dict[str, int] = field(default_factory=dict)
    queue_length: Dict[str, int] = field(default_factory=dict)
    waiting_time: Dict[str, float] = field(default_factory=dict)
    signal_phase: int = 0
    signal_state: str = ""
    step: int = 0
    emergency_present: bool = False
    emergency_lanes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
#  SumoEnvironment
# ──────────────────────────────────────────────────────────────────────

class SumoEnvironment:
    """
    Low-level TraCI wrapper for the single-intersection SUMO simulation.

    Typical usage
    -------------
    >>> env = SumoEnvironment(use_gui=False)
    >>> env.start()
    >>> for _ in range(1000):
    ...     state = env.get_state()
    ...     env.step(action=0)
    >>> env.stop()

    Parameters
    ----------
    cfg_path : str or Path, optional
        Path to the SUMO .sumocfg file.  Defaults to SUMO_CFG_PATH.
    port : int
        TCP port on which SUMO listens for TraCI connections.
    use_gui : bool
        When True SUMO-GUI is launched so you can watch the simulation.
        Set to False during training for speed.
    step_length : float
        Simulation step duration in seconds.
    min_phase_duration : int
        Minimum number of simulation steps a signal phase must run before
        the agent is allowed to switch it again (prevents flickering).
    """

    def __init__(
        self,
        cfg_path: Optional[Path] = None,
        net_file: Optional[Path] = None,
        port: int = 8813,
        use_gui: bool = False,
        step_length: float = 1.0,
        min_phase_duration: int = 5,
    ):
        self.cfg_path = Path(cfg_path) if cfg_path else SUMO_CFG_PATH
        self.net_file = Path(net_file) if net_file else None
        self.port = port
        self.use_gui = use_gui
        self.step_length = step_length
        self.min_phase_duration = min_phase_duration

        self._sumo_proc: Optional[subprocess.Popen] = None
        self._connected: bool = False
        self._current_step: int = 0
        self._steps_since_phase_change: int = 0

        # Caches filled on first connection
        self.tl_id: str = TL_JUNCTION_ID         # dynamic TL ID
        self._controlled_lanes: List[str] = []   # lanes feeding self.tl_id
        self._num_phases: int = 0                # total phases in the TL program

    # ──────────────────────────────────────────────────────────────────
    #  start() — launch SUMO and open TraCI connection
    # ──────────────────────────────────────────────────────────────────

    # Maximum connection retry count before force-killing the child process
    _MAX_START_RETRIES = 3

    def start(self) -> None:
        """
        Launch SUMO as a subprocess and establish the TraCI connection.

        Retry logic (3 attempts):
          On each failed attempt the SUMO subprocess is force-killed before
          retrying, because a half-started SUMO still holds the TCP port.
          If all retries fail, RuntimeError is raised with a diagnostic
          message so the caller (FastAPI / training script) gets a clear
          error instead of an indefinite hang.

        WHAT A ZOMBIE PROCESS IS AND WHY IT MATTERS HERE
        ─────────────────────────────────────────────────
        A "zombie process" is an OS process that has either:
          a) finished but its parent hasn't read its exit code (Unix), or
          b) is still running but NO Python variable references it anymore
             (our case on Windows — the Popen handle is lost).

        If traci.start() launches SUMO (subprocess) but fails to open the
        TCP socket (network race, port collision), the SUMO process stays
        alive in the background, still listening on port 8813.  The next
        call to start() tries to bind the SAME port and fails again —
        creating ANOTHER zombie.  Eventually the machine has N orphan SUMO
        processes consuming RAM and holding N ports.

        The fix: on any exception from traci.start(), we explicitly kill
        the subprocess via _kill_sumo_process() before retrying.  This
        guarantees exactly 0 or 1 SUMO processes alive at any time.
        """
        if self._connected:
            print("SumoEnvironment: already connected — call stop() first.")
            return

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        sumo_cmd = [
            sumo_binary,
            "--configuration-file", str(self.cfg_path),
        ]
        
        if self.net_file and self.net_file.exists():
            sumo_cmd.extend(["--net-file", str(self.net_file)])

        sumo_cmd.extend([
            "--step-length", str(self.step_length),
            "--no-step-log",
            "--no-warnings",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
        ])

        last_error: Optional[Exception] = None

        for attempt in range(1, self._MAX_START_RETRIES + 1):
            try:
                traci.start(sumo_cmd, port=self.port)
                self._connected = True
                self._current_step = 0
                self._steps_since_phase_change = 0

                # ── Query static network info once ────────────────────
                tls = traci.trafficlight.getIDList()
                if TL_JUNCTION_ID in tls:
                    self.tl_id = TL_JUNCTION_ID
                elif tls:
                    self.tl_id = tls[0]
                else:
                    self.tl_id = TL_JUNCTION_ID
                    
                self._controlled_lanes = list(
                    traci.trafficlight.getControlledLanes(self.tl_id)
                )
                logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
                self._num_phases = len(logics[0].phases) if logics else 4

                print(
                    f"SumoEnvironment started on port {self.port}."
                    f" (attempt {attempt}/{self._MAX_START_RETRIES})\n"
                    f"  Controlled lanes : {self._controlled_lanes}\n"
                    f"  Signal phases    : {self._num_phases}"
                )
                return  # success — exit the retry loop

            except Exception as e:
                last_error = e
                print(
                    f"  [WARN] SUMO start attempt {attempt}/{self._MAX_START_RETRIES} "
                    f"failed: {e}"
                )
                # Kill any half-started SUMO before retrying
                self._kill_sumo_process()
                time.sleep(0.5 * attempt)  # back off slightly between retries

        # All retries exhausted — raise so caller gets a 500, not a hang
        raise RuntimeError(
            f"Failed to start SUMO on port {self.port} after "
            f"{self._MAX_START_RETRIES} attempts.  Last error: {last_error}\n"
            f"Check that no other SUMO process is using port {self.port} "
            f"(use 'netstat -ano | findstr {self.port}' on Windows)."
        )

    # ──────────────────────────────────────────────────────────────────
    #  stop() — cleanly shut down TraCI and the SUMO process
    # ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """
        Send the TraCI 'close' command and terminate the SUMO process.

        WHY stop() MUST BE IDEMPOTENT
        ─────────────────────────────
        "Idempotent" means calling stop() N times has the same effect as
        calling it once — no errors, no side-effects on the second call.

        stop() can be called from multiple code paths:
          1. Normal shutdown: training loop ends → stop()
          2. Exception handler: _handle_traci_error catches a crash → stop()
          3. Context manager: __exit__ fires → stop()
          4. FastAPI lifespan shutdown event → stop()

        If stop() raised an error on the second call (e.g. trying to close
        an already-closed socket, or killing an already-dead process), the
        second caller would crash, potentially leaving OTHER resources
        uncleaned.  Idempotency means every code path can safely call
        stop() without checking whether someone else already did.

        Implementation:
          • self._connected flag gates the TraCI close attempt
          • _kill_sumo_process() checks if the process is alive before killing
          • Both set their flags to the "already done" state after executing
        """
        if not self._connected:
            return

        # Attempt graceful TraCI close first
        try:
            traci.close()
        except Exception as e:
            print(f"  [WARN] TraCI close raised (non-fatal): {e}")

        self._connected = False
        self._current_step = 0

        # Force-kill the subprocess as a safety net in case traci.close()
        # failed to actually terminate SUMO (e.g., the TCP connection was
        # already broken so CMD_CLOSE never reached the server).
        self._kill_sumo_process()

        print("SumoEnvironment: stopped.")

    # ──────────────────────────────────────────────────────────────────
    #  reset() — restart the simulation from t=0
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> "SimulationState":
        """
        Restart the simulation by closing and reopening the TraCI connection.

        Returns
        -------
        SimulationState
            The initial state at simulation step 0.
        """
        self.stop()
        time.sleep(0.1)   # give SUMO a moment to release the port
        self.start()
        return self.get_state()

    # ──────────────────────────────────────────────────────────────────
    #  get_state() — read all relevant metrics from SUMO
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> SimulationState:
        """
        Query SUMO for the current intersection state and return it as a
        structured SimulationState.

        TraCI calls used and what they do
        -----------------------------------
        traci.lane.getLastStepVehicleNumber(laneID)
            → How many vehicles passed over the lane in the last sim step.
              Used as a proxy for instantaneous traffic density.

        traci.lane.getLastStepHaltingNumber(laneID)
            → Number of vehicles whose speed is below 0.1 m/s on that lane.
              This IS the queue length — the count of vehicles waiting at red.

        traci.lane.getWaitingTime(laneID)
            → Sum of accumulated waiting time (s) of *all* vehicles on the
              lane right now.  High values = vehicles stuck a long time.

        traci.trafficlight.getPhase(tlID)
            → Integer index of the currently active phase in the TL program
              (0 = first phase in tlLogic, e.g. North-South green).

        traci.trafficlight.getRedYellowGreenState(tlID)
            → Raw phase string like "GGGggrrrr..." where each character
              controls one lane link at the junction.

        traci.vehicle.getIDList()
            → Returns a tuple of all vehicle IDs currently in the network.
              Used to detect emergency vehicles by checking their type.

        traci.vehicle.getTypeID(vehID)
            → Returns the vehicle type string (e.g. "passenger", "emergency").
              Emergency vehicles have type IDs starting with EMERGENCY_TYPE_PREFIX.

        traci.vehicle.getLaneID(vehID)
            → The lane the vehicle is currently on.  We use this to tell the
              EmergencyAgent which lanes need a green corridor.
        """
        self._assert_connected()

        state = SimulationState(step=self._current_step)

        # ── Per-lane metrics ──────────────────────────────────────────
        for lane_id in self._controlled_lanes:
            # Vehicle count: how many vehicles are on this lane right now
            state.vehicle_count[lane_id] = (
                traci.lane.getLastStepVehicleNumber(lane_id)
            )

            # Queue length: vehicles that are effectively stationary (waiting)
            state.queue_length[lane_id] = (
                traci.lane.getLastStepHaltingNumber(lane_id)
            )

            # Cumulative waiting time for all vehicles on this lane
            state.waiting_time[lane_id] = (
                traci.lane.getWaitingTime(lane_id)
            )

        # ── Traffic light state ───────────────────────────────────────
        # Wait, getPhase returns an int index of the current phase pattern
        state.signal_phase = traci.trafficlight.getPhase(self.tl_id)

        # Get the actual string representation of the phase (e.g. "GGrr")
        state.signal_state = traci.trafficlight.getRedYellowGreenState(
            self.tl_id
        )

        # ── Emergency vehicle detection ───────────────────────────────
        all_vehicle_ids = traci.vehicle.getIDList()
        emergency_lanes: List[str] = []

        for veh_id in all_vehicle_ids:
            # getTypeID returns the <vType> id assigned in the route file
            veh_type = traci.vehicle.getTypeID(veh_id)
            if veh_type.startswith(EMERGENCY_TYPE_PREFIX):
                # getLaneID returns "edgeID_laneIndex", e.g. "north_to_center_0"
                lane = traci.vehicle.getLaneID(veh_id)
                if lane and not lane.startswith(":"):
                    # Internal junction lanes start with ":"  — skip them
                    emergency_lanes.append(lane)

        state.emergency_present = len(emergency_lanes) > 0
        state.emergency_lanes = emergency_lanes

        return state

    # ──────────────────────────────────────────────────────────────────
    #  step(action) — apply a phase change then advance the simulation
    # ──────────────────────────────────────────────────────────────────

    def step(
        self,
        action: int,
    ) -> Tuple[SimulationState, bool]:
        """
        Apply a traffic signal phase change and advance the simulation by
        one step (step_length seconds).

        WHY try/except/finally IS USED HERE
        ────────────────────────────────────
        Every TraCI call in this method is a TCP round-trip.  Any of them
        can fail if SUMO crashes, is killed externally, or the network
        drops:
          • traci.trafficlight.setPhase → network write fails
          • traci.simulationStep → SUMO segfaults mid-step
          • traci.simulation.getMinExpectedNumber → socket EOF

        The `try` block wraps ALL TraCI calls so we catch TraCIException
        (and its subclasses like FatalTraCIError) in one place.

        WHY `finally` MATTERS EVEN WHEN `except` CATCHES THE ERROR
        ─────────────────────────────────────────────────────────────
        The `except` block handles the error (logging, reconnect attempt).
        But `except` does NOT run if:
          • A *different* exception type is raised (e.g., KeyboardInterrupt)
          • The except block itself raises a new exception
          • The thread is cancelled by asyncio

        `finally` ALWAYS runs — even in those cases.  This is where we
        place the step counter increment and any bookkeeping that must
        happen regardless of success or failure, so the internal state
        never gets out of sync with SUMO.

        Parameters
        ----------
        action : int
            The desired signal phase index (0 … num_phases-1).

        Returns
        -------
        state : SimulationState
            The new simulation state *after* the step.
        done : bool
            True when SUMO reports the simulation has ended.
        """
        self._assert_connected()

        try:
            # ── Guard: enforce minimum phase duration ─────────────────
            if action < self.action_space.n:
                current_phase = traci.trafficlight.getPhase(self.tl_id)
                if action != current_phase:
                    # Enforce minimum cycle length
                    if self._steps_since_phase_change >= self.min_phase_duration:
                        # Switch phase directly
                        traci.trafficlight.setPhase(self.tl_id, action)
                        traci.trafficlight.setPhaseDuration(self.tl_id, 1_000_000)
                        self._steps_since_phase_change = 0
                else:
                    self._steps_since_phase_change += 1

            # ── Advance the simulation by one step ────────────────────
            traci.simulationStep()
            self._current_step += 1

            # ── Check terminal condition ──────────────────────────────
            done = traci.simulation.getMinExpectedNumber() == 0

            # Read the fresh state after the step
            new_state = self.get_state()
            return new_state, done

        except traci.exceptions.TraCIException as e:
            self._handle_traci_error(e)
            # _handle_traci_error always re-raises, but make the type
            # checker happy with an explicit raise
            raise

        # NOTE: No `finally` block for step counter increment.
        # If traci.simulationStep() fails, SUMO did NOT advance its
        # internal clock.  Incrementing _current_step anyway would
        # permanently desync our counter from SUMO's actual time,
        # causing every subsequent get_state().step to be wrong.

    # ──────────────────────────────────────────────────────────────────
    #  Emergency vehicle helpers
    # ──────────────────────────────────────────────────────────────────

    def get_emergency_vehicles(self) -> List[Dict]:
        """
        Return a list of dicts describing every emergency vehicle currently
        in the network.

        Each dict contains:
            id      — vehicle ID string
            lane    — current lane ID
            edge    — current edge ID (road segment)
            speed   — current speed in m/s
            route   — list of remaining edge IDs to their destination
        """
        self._assert_connected()
        result = []
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id).startswith(EMERGENCY_TYPE_PREFIX):
                # getLaneID / getEdgeID / getSpeed are straightforward getters
                lane_id = traci.vehicle.getLaneID(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                speed   = traci.vehicle.getSpeed(veh_id)

                # getRoute returns the full planned route; getRouteIndex is
                # how far along it the vehicle currently is.
                full_route = traci.vehicle.getRoute(veh_id)
                route_idx  = traci.vehicle.getRouteIndex(veh_id)
                remaining_route = list(full_route[route_idx:])

                result.append({
                    "id": veh_id,
                    "lane": lane_id,
                    "edge": edge_id,
                    "speed": speed,
                    "route": remaining_route,
                })
        return result

    def force_green_corridor(self, edge_ids: List[str]) -> None:
        """
        Force green for all traffic-light links that serve the given edges.

        This is a raw override used by the EmergencyAgent to clear a path.
        It builds a phase string where links feeding `edge_ids` are 'G'
        and all others are 'r'.

        Parameters
        ----------
        edge_ids : list of str
            Edges (road segments) that should receive a green light.
        """
        self._assert_connected()

        # getControlledLinks returns a list-of-lists: each element is a list
        # of (incoming_lane, outgoing_lane, internal_lane) tuples for one
        # which gives a list of tuples like (lanes_controlled_by_index_0, ...)
        controlled_links = traci.trafficlight.getControlledLinks(self.tl_id)
        
        # We need a phase string with length == number of signal links
        phase_chars = ["r"] * len(controlled_links)
        for i, link_group in enumerate(controlled_links):
            if not link_group:
                continue
            incoming_lane = link_group[0][0]  # e.g. "north_to_center_0"
            # Extract the edge part (drop the lane index suffix after "_")
            incoming_edge = "_".join(incoming_lane.split("_")[:-1])
            if incoming_edge in edge_ids:
                phase_chars[i] = "G"   # unconditional green
            else:
                phase_chars[i] = "r"   # red for all others

        custom_phase_str = "".join(phase_chars)

        # setRedYellowGreenState() lets you write an arbitrary phase string
        # bypassing the programmatic phase logic entirely.  Use with care
        # - invalid strings can cause SUMO to raise an error.
        traci.trafficlight.setRedYellowGreenState(
            self.tl_id, custom_phase_str
        )

    # ──────────────────────────────────────────────────────────────────
    #  Properties / convenience accessors
    # ──────────────────────────────────────────────────────────────────

    @property
    def controlled_lanes(self) -> List[str]:
        """Return the list of lanes controlled by the traffic light."""
        return self._controlled_lanes

    @property
    def num_phases(self) -> int:
        """Number of signal phases defined in the SUMO TL program."""
        return self._num_phases

    @property
    def is_connected(self) -> bool:
        """True when a live TraCI connection exists."""
        return self._connected

    @property
    def current_step(self) -> int:
        """The current simulation step counter."""
        return self._current_step

    # ──────────────────────────────────────────────────────────────────
    #  Context manager support  (with SumoEnvironment() as env: ...)
    # ──────────────────────────────────────────────────────────────────
    #
    #  Usage:
    #    with SumoEnvironment(cfg_path="...") as env:
    #        env.start()  # optional — __enter__ calls start() for you
    #        for _ in range(1000):
    #            state, done = env.step(action)
    #    # ← __exit__ calls stop() HERE, even if the body raised
    #
    #  This is the Pythonic way to guarantee resource cleanup.
    #  If any line inside `with` raises (TraCIException, KeyboardInterrupt,
    #  even SystemExit), __exit__ still runs, which calls our idempotent
    #  stop(), which kills the SUMO subprocess.  No zombies.

    def __enter__(self) -> "SumoEnvironment":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Always stop SUMO on context exit, regardless of exception."""
        self.stop()
        # Returning None (falsy) lets any exception propagate to the caller.
        # We do NOT swallow exceptions — the caller should see the error.

    # ──────────────────────────────────────────────────────────────────
    #  Error handling and process management
    # ──────────────────────────────────────────────────────────────────

    def _handle_traci_error(self, error: Exception) -> None:
        """
        Handle a TraCI communication failure during simulation.

        Strategy:
          1. Log the error with simulation time for debugging
          2. Attempt a single reconnect (SUMO may still be alive but the
             TCP socket died — e.g. after a transient network hiccup)
          3. If reconnect fails → call stop() to kill the subprocess
          4. Always re-raise so FastAPI returns HTTP 500, not a silent hang

        This method is the single point of failure handling for ALL TraCI
        errors in step(), get_state(), and force_green_corridor().  Having
        one handler avoids duplicating cleanup logic across methods.
        """
        sim_time = self._current_step * self.step_length
        print(
            f"  [ERROR] TraCI error at simulation time {sim_time:.1f}s "
            f"(step {self._current_step}): {error}"
        )

        # Attempt single reconnect
        try:
            print("  [RETRY] Attempting TraCI reconnect...")
            traci.close()
            time.sleep(0.3)
            traci.start(
                [
                    "sumo-gui" if self.use_gui else "sumo",
                    "--configuration-file", str(self.cfg_path),
                    "--remote-port", str(self.port),
                    "--step-length", str(self.step_length),
                    "--no-step-log",
                    "--no-warnings",
                    "--time-to-teleport", "-1",
                    "--collision.action", "warn",
                ],
                port=self.port,
            )
            self._connected = True
            print("  [OK] Reconnected to SUMO.")
        except Exception as reconnect_error:
            print(
                f"  [FAIL] Reconnect failed: {reconnect_error}. "
                f"Stopping SUMO to prevent zombie."
            )
            self.stop()

        # Always re-raise so the caller (FastAPI route handler, training
        # loop) sees the error and can return an appropriate response
        # (HTTP 500, episode termination, etc.) instead of silently
        # continuing with a broken connection.
        raise RuntimeError(
            f"TraCI communication failure at step {self._current_step}: {error}"
        ) from error

    def _kill_sumo_process(self) -> None:
        """
        Force-kill the SUMO subprocess if it is still running.

        Safe to call multiple times (idempotent) — if the process is
        already dead or was never started, this is a no-op.

        Uses Popen.poll() to check if the process is still alive before
        attempting kill(), which prevents:
          • PermissionError on Windows (killing an already-dead PID)
          • OSError on Linux ("No such process")
          • ProcessLookupError on macOS
        """
        # TraCI stores the subprocess handle internally.  Access it
        # through traci's connection list if available.
        try:
            # traci keeps active connections in traci._connections dict.
            # Each connection has a .sumoProcess attribute.
            for conn in list(getattr(traci, '_connections', {}).values()):
                proc = getattr(conn, 'sumoProcess', None)
                if proc and proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                    print(f"  [OK] Killed SUMO subprocess (PID {proc.pid}).")
        except Exception as e:
            print(f"  [WARN] Error killing SUMO subprocess: {e}")

    # ──────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "SumoEnvironment is not connected. Call start() first."
            )
