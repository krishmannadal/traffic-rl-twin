"""
websocket.py - WebSocket Connection Manager & Endpoint Definitions
===================================================================

This module handles ALL real-time communication between the server
and clients:
  - React dashboard (browser) -> /ws/dashboard
  - React Native / Expo (phone) -> /ws/vehicle/{vehicle_id}

WHY WEBSOCKET OVER REST POLLING FOR REAL-TIME DATA
---------------------------------------------------
REST (HTTP) polling means the client repeatedly asks "what's the state?"
on a fixed interval:
    Client: GET /metrics/current   -> Server: {waiting: 42.1, ...}
    Client: GET /metrics/current   -> Server: {waiting: 41.9, ...}
    Client: GET /metrics/current   -> Server: {waiting: 43.0, ...}
    ... (every 500ms, 120 requests/minute)

Problems with polling:
  1. LATENCY: You only get updates at the poll interval.  At 500ms
     polling, you're always 0-500ms behind reality.
  2. OVERHEAD: Each HTTP request has TCP handshake + HTTP headers
     (200-800 bytes) even if the payload is tiny.  120 requests/min =
     significant network and CPU overhead on the server.
  3. WASTED REQUESTS: If nothing changed, you still made the request
     and discarded the identical response.

WebSocket solves all three:
  1. PUSH: The server sends updates IMMEDIATELY when state changes -
     sub-millisecond latency after the step completes.
  2. PERSISTENT CONNECTION: One TCP handshake for the session lifetime.
     Subsequent messages have only a 2-14 byte framing overhead.
  3. EVENT-DRIVEN: The server only sends when there's new data.
     If the simulation pauses, no bytes are sent.

For a simulation running at 1 step/second with 4 connected dashboards:
  - Polling (500ms): 480 HTTP requests/min, ~300KB overhead/min
  - WebSocket: 4 pushes/sec, ~4KB data/min (75x less overhead)

WHY SIMULATION STATE AND TRAINING METRICS ARE SEPARATE CHANNELS
---------------------------------------------------------------
Simulation state (what the intersection looks like RIGHT NOW) and
training metrics (what the agent learned over time) have completely
different consumers, frequencies, and use cases:

  Simulation state:
    Frequency: every simulation step (1 Hz during live sim)
    Consumers: dashboard map widget, phone app, digital twin viewer
    Content: signal phases, queue lengths, vehicle positions
    Purpose: "show me what's happening at the intersection"

  Training metrics:
    Frequency: every 10,000 training steps (roughly every 30-60 sec)
    Consumers: TensorBoard-like dashboard panel, researcher's notebook
    Content: reward curve, Q-loss, epsilon, FPS
    Purpose: "is the agent learning? should I stop training?"

Mixing them into one channel creates two problems:
  1. INTERFERENCE: A 10MB training checkpoint message would block the
     dashboard from receiving signal state updates for seconds.
  2. COMPLEXITY: The frontend would need to inspect every message to
     figure out which panel to update - messy client-side routing.

Separate channels let dashboard panels subscribe only to what they
need and process each at the appropriate rate.

WHAT HAPPENS WHEN A CLIENT DISCONNECTS MID-SIMULATION
------------------------------------------------------
WebSocket disconnections happen silently from the server's perspective -
the socket is closed on the client side (browser tab closed, phone
screen locked, network drop) but the server's send() call only fails
when it next tries to write to the dead socket.

Our broadcast() method handles this gracefully:
  1. Iterates through all active connections.
  2. Calls websocket.send_text(payload) inside a try/except.
  3. If the send FAILS (ConnectionClosedError or any exception), the
     connection is added to a `to_remove` list.
  4. After the iteration, all dead connections are removed from the
     active list atomically.

This ensures:
  - The broadcast never crashes because of one dead client.
  - Dead connections are cleaned up immediately (no zombie sockets).
  - The simulation keeps running regardless of client state.

The same pattern applies to vehicle connections.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Create the websocket router so endpoints can live in this file
# and be mounted in main.py alongside the REST routers.
ws_router = APIRouter(tags=["WebSocket"])


# ----------------------------------------------------------------------
#  ConnectionManager
# ----------------------------------------------------------------------

class ConnectionManager:
    """
    Central registry of all active WebSocket connections.

    Maintains two separate pools:
      active_connections  - dashboard browser clients (React)
      vehicle_connections - phone clients keyed by vehicle ID

    All broadcast operations are async to not block the event loop.
    Each send is wrapped in try/except so a dead client never crashes
    the broadcast loop (see module docstring for details).
    """

    def __init__(self):
        # Dashboard clients (browsers, admin panels)
        self.active_connections: List[WebSocket] = []

        # Phone / vehicle clients: {vehicle_id: WebSocket}
        self.vehicle_connections: Dict[str, WebSocket] = {}

    # -- Dashboard connections -----------------------------------------

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a dashboard WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(
            f"  [WS] Dashboard client connected. "
            f"Total: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a dashboard client that has disconnected."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(
            f"  [WS] Dashboard client disconnected. "
            f"Total: {len(self.active_connections)}"
        )

    # -- Vehicle / phone connections -----------------------------------

    async def connect_vehicle(
        self, vehicle_id: str, websocket: WebSocket
    ) -> None:
        """
        Accept and register a phone app connection for a specific vehicle.

        One phone = one vehicle_id.  If a phone reconnects while already
        registered, the old connection is replaced (handles page refreshes).
        """
        await websocket.accept()
        self.vehicle_connections[vehicle_id] = websocket
        print(f"  [CAR] Vehicle client connected: {vehicle_id}")

    def disconnect_vehicle(self, vehicle_id: str) -> None:
        """Remove a phone client that has disconnected."""
        removed = self.vehicle_connections.pop(vehicle_id, None)
        if removed:
            print(f"  [CAR] Vehicle client disconnected: {vehicle_id}")

    # -- Broadcast to all dashboards -----------------------------------

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """
        Send a JSON payload to ALL connected dashboard clients concurrently.

        WHY asyncio.gather INSTEAD OF SEQUENTIAL AWAIT
        ----------------------------------------------
        The sequential pattern:
            for conn in connections:
                await conn.send_text(payload)   # blocks here per client

        must wait for EACH TCP write to complete before starting the next.
        If client A has a full receive buffer (slow phone network, sleeping
        laptop lid), the event loop suspends at that `await` and the entire
        server freezes: no simulation steps advance, no heartbeats process,
        no REST requests are served - the simulation clock effectively stops.

        asyncio.gather fires ALL sends simultaneously as a single batch of
        concurrent coroutines.  The event loop interleaves their I/O waits,
        so a slow client only delays its own send, not every other client's.
        Total broadcast time = max(individual send times) instead of sum.

        WHY return_exceptions=True PREVENTS ONE BAD CLIENT CRASHING ALL
        ---------------------------------------------------------------
        Without return_exceptions=True (the default), asyncio.gather raises
        the FIRST exception it encounters and cancels all remaining sends.
        If client 3 of 10 has a dead socket, clients 4-10 never receive
        the payload.  With return_exceptions=True, exceptions are returned
        as values in the results list - every client gets served, and we
        clean up failed connections in a second pass after all sends finish.

        WHAT HAPPENS TO SIMULATION CLOCK DURING SEQUENTIAL BLOCKING I/O
        ------------------------------------------------------------------
        FastAPI runs on a single-threaded asyncio event loop.  Each `await`
        yields control to the loop - but only if the awaitable can complete
        quickly.  A WebSocket send to a client whose TCP receive buffer is
        full CANNOT complete until the client reads data.  The event loop
        parks on that `await` indefinitely, queuing up every other coroutine:
          - Simulation step callbacks sit waiting
          - REST /health requests time out
          - Other WebSocket clients stop receiving updates
          - The simulation wall-clock time diverges from sim time
        With gather, the blocking is limited to the slowest single client
        and does not cascade to the rest of the system.
        """
        if not self.active_connections:
            return

        payload = json.dumps(data)

        # Snapshot the list so late disconnects during gather don't mutate
        # active_connections mid-flight.
        connections = list(self.active_connections)

        results = await asyncio.gather(
            *[conn.send_text(payload) for conn in connections],
            return_exceptions=True,
        )

        # Clean up any connections that failed during the concurrent send
        failed = [
            conn
            for conn, result in zip(connections, results)
            if isinstance(result, Exception)
        ]
        for conn in failed:
            self.disconnect(conn)

    # -- Send to a single dashboard connection -------------------------

    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """
        Send a JSON payload to one specific dashboard WebSocket client.

        Used by the dashboard endpoint to reply to control messages
        (ping/pong, subscribe acknowledgements, unknown-message echoes)
        without broadcasting to every connected client.

        Parameters
        ----------
        message : dict
            The payload to serialise and send.
        websocket : WebSocket
            The specific client connection to target.
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            # Connection is dead - remove it from the pool silently
            self.disconnect(websocket)

    # -- Send to single vehicle ----------------------------------------

    async def send_to_vehicle(
        self, vehicle_id: str, data: Dict[str, Any]
    ) -> bool:
        """
        Send a JSON payload to a specific vehicle's phone app.

        Uses gather with a single coroutine so error handling is uniform
        with broadcast().  The pattern is identical to broadcast() but
        targets one client by vehicle_id rather than the entire pool.

        Returns True if the send succeeded, False if the vehicle is not
        connected or the send raised an exception.
        """
        websocket = self.vehicle_connections.get(vehicle_id)
        if websocket is None:
            return False

        results = await asyncio.gather(
            websocket.send_text(json.dumps(data)),
            return_exceptions=True,
        )
        if isinstance(results[0], Exception):
            self.disconnect_vehicle(vehicle_id)
            return False
        return True

    # -- Properties ----------------------------------------------------

    @property
    def active_connections_count(self) -> int:
        """Number of active dashboard connections."""
        return len(self.active_connections)

    @property
    def vehicle_connections_count(self) -> int:
        """Number of active vehicle/phone connections."""
        return len(self.vehicle_connections)


# ----------------------------------------------------------------------
#  Simulation State Broadcaster
# ----------------------------------------------------------------------

# Maps laneName prefixes -> cardinal directions for clean JSON keys
_DIRECTION_MAP = {
    "north_to_center": "north",
    "south_to_center": "south",
    "east_to_center":  "east",
    "west_to_center":  "west",
}

# Signal phase index → human-readable label
_PHASE_LABELS = {
    0: "NS_GREEN",
    1: "NS_YELLOW",
    2: "EW_GREEN",
    3: "EW_YELLOW",
}


def _group_by_direction(
    raw: Dict[str, float], scale: float = 1.0
) -> Dict[str, float]:
    """
    Collapse per-lane values into per-direction sums.

    raw keys look like "north_to_center_0", "north_to_center_1".
    We strip the lane suffix and sum both lanes into one direction bucket.
    """
    grouped: Dict[str, float] = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
    for lane_id, value in raw.items():
        for prefix, direction in _DIRECTION_MAP.items():
            if lane_id.startswith(prefix):
                grouped[direction] += value / scale
                break
    return {k: round(v, 2) for k, v in grouped.items()}


async def broadcast_simulation_state(
    manager: ConnectionManager,
    env_state: Dict[str, Any],
    current_reward: float = 0.0,
    step: int = 0,
) -> None:
    """
    Format a raw SumoEnvironment state dict into a clean JSON payload
    and broadcast it to all connected dashboard clients.

    This is the hot path — called once per simulation step (1 Hz).
    Keep it fast: no blocking I/O, no heavy computation.

    Parameters
    ----------
    manager : ConnectionManager
        The global connection manager from main.py.
    env_state : dict
        Raw state from TrafficEnv.render() or SumoEnvironment.get_state().
        Expected keys: signal_phase, queue_length, waiting_time,
                       vehicle_count, emergency_present, emergency_lanes,
                       sim_step.
    current_reward : float
        Reward from the last step (for dashboard sparkline).
    step : int
        Current episode step number.

    Broadcast Payload Schema
    ────────────────────────
    {
      "type": "simulation_state",
      "timestamp": float,          // server epoch time (for latency calc)
      "step": int,
      "simulation_time": float,    // SUMO simulation clock (seconds)
      "signal": {
        "phase_index": int,        // 0-3
        "phase_label": str,        // "NS_GREEN" etc.
        "phase_per_direction": {   // which direction is green/yellow/red
          "north": str, "south": str, "east": str, "west": str
        }
      },
      "vehicle_counts": {"north": int, "south": int, "east": int, "west": int},
      "queue_lengths":  {"north": int, "south": int, "east": int, "west": int},
      "waiting_times":  {"north": float, ...},  // seconds
      "current_reward": float,
      "emergency_active": bool,
      "emergency_position": str | null   // edge ID of emergency vehicle
    }
    """
    phase_index = env_state.get("signal_phase", 0)
    phase_label = _PHASE_LABELS.get(phase_index, "UNKNOWN")

    # Determine which movement each direction currently has
    # based on signal phase index
    _phase_direction_state = {
        0: {"north": "GREEN",  "south": "GREEN",  "east": "RED",    "west": "RED"},
        1: {"north": "YELLOW", "south": "YELLOW", "east": "RED",    "west": "RED"},
        2: {"north": "RED",    "south": "RED",    "east": "GREEN",  "west": "GREEN"},
        3: {"north": "RED",    "south": "RED",    "east": "YELLOW", "west": "YELLOW"},
    }

    # Emergency position: first edge in emergency_lanes list, or null
    emergency_lanes: List[str] = env_state.get("emergency_lanes", [])
    emergency_position: Optional[str] = emergency_lanes[0] if emergency_lanes else None

    payload = {
        "type": "simulation_state",
        "timestamp": time.time(),
        "step": step,
        "simulation_time": float(env_state.get("sim_step", step)),
        "signal": {
            "phase_index": phase_index,
            "phase_label": phase_label,
            "phase_per_direction": _phase_direction_state.get(phase_index, {}),
        },
        "vehicle_counts": _group_by_direction(
            env_state.get("vehicle_count", {})
        ),
        "queue_lengths": _group_by_direction(
            env_state.get("queue_length", {})
        ),
        "waiting_times": _group_by_direction(
            env_state.get("waiting_time", {})
        ),
        "current_reward": round(current_reward, 4),
        "emergency_active": bool(env_state.get("emergency_present", False)),
        "emergency_position": emergency_position,
    }

    await manager.broadcast(payload)


# ──────────────────────────────────────────────────────────────────────
#  Training Metrics Broadcaster
# ──────────────────────────────────────────────────────────────────────

async def broadcast_training_metrics(
    manager: ConnectionManager,
    metrics: Dict[str, Any],
) -> None:
    """
    Format and broadcast training progress metrics to all dashboards.

    Called by TrafficWandbCallback every 10,000 training steps.
    Kept separate from simulation state so dashboard panels can
    subscribe independently and update at different rates.

    Parameters
    ----------
    manager : ConnectionManager
        The global connection manager from main.py.
    metrics : dict
        Training metrics from the W&B callback.

    Broadcast Payload Schema
    ────────────────────────
    {
      "type": "training_metrics",
      "timestamp": float,
      "timestep": int,
      "reward": float,          // mean reward over last 100 episodes
      "loss": float,            // TD loss (Q-value MSE)
      "epsilon": float,         // current exploration rate
      "fps": float,             // environment steps per second
      "gpu_memory_mb": float    // VRAM in use (0 if CPU)
    }
    """
    payload = {
        "type": "training_metrics",
        "timestamp": time.time(),
        "timestep": int(metrics.get("timestep", 0)),
        "reward": round(float(metrics.get("reward", 0.0)), 4),
        "loss": round(float(metrics.get("loss", 0.0)), 6),
        "epsilon": round(float(metrics.get("epsilon", 0.0)), 4),
        "fps": round(float(metrics.get("fps", 0.0)), 1),
        "gpu_memory_mb": round(float(metrics.get("gpu_memory_mb", 0.0)), 1),
    }

    await manager.broadcast(payload)


# ──────────────────────────────────────────────────────────────────────
#  WebSocket Endpoints
# ──────────────────────────────────────────────────────────────────────

@ws_router.websocket("/ws/dashboard")
async def dashboard_ws_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the React dashboard.

    Connection lifecycle:
      1. Client opens ws://host:8000/ws/dashboard
      2. Server accepts → adds to active_connections pool
      3. Server pushes simulation_state and training_metrics messages
         as they are generated (via broadcast calls from routes)
      4. Client may send control messages:
         {"type": "ping"} → server replies {"type": "pong"}
         {"type": "subscribe", "channel": "metrics"} → acknowledged
      5. On disconnect (tab closed / network drop) → removed from pool

    Note: the actual state broadcasting is TRIGGERED by the simulation
    route (POST /simulation/step), not by this endpoint.  This endpoint
    only manages the connection lifecycle.  The manager is imported
    from main.py in routes/simulation.py.
    """
    from api.main import manager as global_manager

    await global_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any client → server messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"type": "unknown", "raw": data}

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await global_manager.send_personal(
                    {"type": "pong", "timestamp": time.time()}, websocket
                )
            elif msg_type == "subscribe":
                await global_manager.send_personal(
                    {
                        "type": "subscribed",
                        "channel": msg.get("channel", "all"),
                        "message": "Subscription acknowledged. State will be pushed on each sim step.",
                    },
                    websocket,
                )
            else:
                # Unknown message type — acknowledge receipt
                await global_manager.send_personal(
                    {"type": "ack", "received": msg}, websocket
                )

    except WebSocketDisconnect:
        global_manager.disconnect(websocket)


@ws_router.websocket("/ws/vehicle/{vehicle_id}")
async def vehicle_ws_endpoint(websocket: WebSocket, vehicle_id: str):
    """
    WebSocket endpoint for a phone app tracking a specific vehicle.

    Each phone connects with the ID of the vehicle it's tracking,
    e.g. ws://host:8000/ws/vehicle/emergency_001.

    The server will push targeted messages when:
      • The vehicle's signal ahead changes
      • The vehicle's waiting time exceeds a threshold
      • An emergency corridor is activated for this vehicle

    Connection lifecycle:
      1. Phone opens connection with vehicle_id in the URL
      2. Server accepts → adds to vehicle_connections[vehicle_id]
      3. Server sends vehicle-specific updates via send_to_vehicle()
      4. Phone sends {"type": "heartbeat"} to keep connection alive
      5. On disconnect → vehicle_connections[vehicle_id] is removed

    Note: if the same vehicle_id reconnects (e.g. app restart), the
    old connection is replaced by the new one.
    """
    from api.main import manager as global_manager

    await global_manager.connect_vehicle(vehicle_id, websocket)

    # Send an initial welcome message with connection confirmation
    await global_manager.send_to_vehicle(
        vehicle_id,
        {
            "type": "connected",
            "vehicle_id": vehicle_id,
            "message": f"Tracking vehicle '{vehicle_id}'. Updates will be pushed as simulation progresses.",
            "timestamp": time.time(),
        },
    )

    try:
        while True:
            # Keep connection alive; handle heartbeat and status requests
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"type": "unknown"}

            msg_type = msg.get("type", "")

            if msg_type == "heartbeat":
                await global_manager.send_to_vehicle(
                    vehicle_id,
                    {"type": "heartbeat_ack", "timestamp": time.time()},
                )
            elif msg_type == "status":
                await global_manager.send_to_vehicle(
                    vehicle_id,
                    {
                        "type": "status",
                        "vehicle_id": vehicle_id,
                        "connected": True,
                        "dashboard_clients": global_manager.active_connections_count,
                    },
                )

    except WebSocketDisconnect:
        global_manager.disconnect_vehicle(vehicle_id)
