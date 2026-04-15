"""
routes/vehicles.py — Phone Vehicle Integration Endpoints
=========================================================

This module handles the two-way link between real phones in the field
and the SUMO simulation:

    Phone              →  WebSocket  →  This module
    (GPS, speed, accel)                 (map, inject into SUMO, respond)

    This module        →  WebSocket  →  Phone
    (signal ahead, speed advice,        (shows on phone UI)
     time_to_green)

HOW moveToXY DIFFERS FROM NORMAL SUMO VEHICLE ROUTING
──────────────────────────────────────────────────────
Normally, SUMO vehicles navigate autonomously:
  1. You define a route in routes.rou.xml (sequence of edges).
  2. SUMO's car-following and lane-changing models move the vehicle
     along the route each step, respecting lane geometry and signals.
  3. You don't touch the vehicle — it drives itself.

traci.vehicle.moveToXY() bypasses ALL of this:
  • It teleports the vehicle to exact (x, y) coordinates every step,
    overriding the car-following model entirely.
  • The vehicle's speed, acceleration, and route adherence are IGNORED
    by SUMO's physics — you supply the real position from the phone's GPS.
  • SUMO still renders the vehicle on the correct lane (it snaps to the
    nearest lane from the (x, y) point) and reports it to TraCI's
    getControlledLinks() (so signal states are still computed correctly).
  • Signal interaction STILL WORKS — even a moveToXY vehicle stops at
    red lights in SUMO's physics, but you can suppress that with the
    keepRoute parameter if the real car doesn't stop.

  Normal routing: SUMO drives the vehicle (deterministic simulator).
  moveToXY:       The real world drives the vehicle (GPS ground truth).

  keepRoute=2 tells SUMO to snap to the lane nearest to (x, y) even
  if that lane isn't on the vehicle's declared route.  This handles GPS
  drift and road geometry mismatches gracefully.

WHY WE NEED A COORDINATE MAPPER
────────────────────────────────
The phone provides GPS (lat, lng) — WGS-84 decimal degrees.
SUMO stores its network in a flat Cartesian grid (metres from origin).
The junction at the intersection centre might be at SUMO (x=350, y=210),
not at (lat=12.9725, lng=77.5925).

Without the mapper, you'd try to call traci.vehicle.moveToXY(12.9725,
77.5925, ...) which would place the vehicle 12.9 metres up the Y-axis
and 77.5 metres along the X-axis — completely wrong.

The CoordinateMapper does a one-time calibration at startup:
  • Reads the SUMO network bounds from map.net.xml via sumolib.
  • Maps those bounds onto your GPS bounding box (configured in init).
  • Performs linear interpolation on every GPS update.

HOW signal_ahead IS CALCULATED FOR A SPECIFIC VEHICLE POSITION
───────────────────────────────────────────────────────────────
Given the vehicle's current SUMO (x, y), we:

  1. Find the nearest lane: sumolib.net.getNeighboringEdges() → lane_id.

  2. Find which traffic light controls that lane:
     traci.lane.getLinks(lane_id) returns a list of connections each
     with a controlling traffic light link index.

  3. Read the current signal state for that link:
     traci.trafficlight.getRedYellowGreenState(tl_id) returns a string
     like "GGGggrrrr".  The index into this string is the link index
     from step 2.  'G'/'g' = green, 'y'/'Y' = yellow, 'r'/'R' = red.

  4. Estimate time_to_green:
     traci.trafficlight.getNextSwitch(tl_id) returns the simulation time
     when the current phase will switch.  If the current signal is red,
     this is approximately when green will arrive — but it's the phase
     SWITCH time, not necessarily the green-for-this-lane time.

  The speed_advice is computed using the "green wave" principle:
     If the vehicle is 100m from a junction and green starts in 8s,
     the ideal speed is 100m / 8s = 12.5 m/s.
     If the vehicle should slow to avoid catching a red:
     If green ends in 5s and vehicle is 80m away at 13 m/s → it'll
     arrive in 80/13 = 6.1s AFTER green ends → advise slow down.
"""

import json
import time
from typing import Any, Dict, Optional

import traci
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import api.main as state
from simulation.coordinate_mapper import CoordinateMapper

router = APIRouter()

# ── Module-level CoordinateMapper (loaded once, reused per connection) ──
_mapper: Optional[CoordinateMapper] = None

def _get_mapper() -> CoordinateMapper:
    global _mapper
    if _mapper is None:
        _mapper = CoordinateMapper()
    return _mapper


# ── Active vehicle registry ──────────────────────────────────────────────
# Stores metadata about currently connected phone clients.
# {vehicle_id: {lat, lng, speed, edge_id, is_emergency, connected_at}}
_active_vehicles: Dict[str, Dict[str, Any]] = {}


# ──────────────────────────────────────────────────────────────────────
#  Signal Advice Calculator
# ──────────────────────────────────────────────────────────────────────

def _get_signal_advice(
    sumo_x: float, sumo_y: float, vehicle_speed: float
) -> Dict[str, Any]:
    """
    Look up the traffic light state ahead of a vehicle at (sumo_x, sumo_y).

    Returns a dict suitable for sending back to the phone:
        signal_ahead   : "green" | "yellow" | "red"
        time_to_green  : estimated seconds until next green (0 if already green)
        speed_advice   : recommended speed in m/s to hit the green wave

    Falls back to safe defaults if SUMO isn't running or lane not found.
    """
    defaults = {
        "signal_ahead": "unknown",
        "time_to_green": 0,
        "speed_advice": vehicle_speed,
        "emergency_mode": False,
    }

    if not state._sumo_running:
        return defaults

    try:
        mapper = _get_mapper()

        # 1. Find the lane the vehicle is on
        lane_id = mapper.find_nearest_lane(sumo_x, sumo_y)
        if lane_id is None:
            return defaults

        # 2. Get the links (downstream connections) from this lane
        # Each link is a tuple, last two elements are tl_id and link_index.
        # traci.lane.getLinks returns list of:
        #   (lane_id, via_lane, has_priority, is_open, has_foe, tl_id,
        #    tl_link_index, direction, length)
        links = traci.lane.getLinks(lane_id)
        if not links:
            return defaults

        # Find the first link that has a traffic light controlling it
        tl_id = None
        tl_link_index = None
        for link in links:
            if link[5]:  # tl_id is the 6th element, empty string if none
                tl_id = link[5]
                tl_link_index = int(link[6])
                break

        if tl_id is None:
            # No traffic light on this lane — vehicle can proceed freely
            return {**defaults, "signal_ahead": "green", "time_to_green": 0}

        # 3. Read the current signal state string
        # e.g. "GGGrrrGGGrrr" — one character per controlled link
        state_str = traci.trafficlight.getRedYellowGreenState(tl_id)
        if tl_link_index >= len(state_str):
            return defaults

        signal_char = state_str[tl_link_index]
        if signal_char in ("G", "g"):
            signal_ahead = "green"
        elif signal_char in ("y", "Y"):
            signal_ahead = "yellow"
        else:
            signal_ahead = "red"

        # 4. Estimate time until next switch
        # getNextSwitch() returns absolute simulation time of next phase change
        sim_time = traci.simulation.getTime()
        next_switch = traci.trafficlight.getNextSwitch(tl_id)
        secs_until_switch = max(0.0, next_switch - sim_time)

        # If currently red, the switch will bring a different phase —
        # which MIGHT be green for this lane (or another yellow).
        # We return the switch time as an approximation.
        time_to_green = int(secs_until_switch) if signal_ahead == "red" else 0

        # 5. Calculate speed advice (green wave)
        # Approximate distance to the stop line using lane length
        lane_length = traci.lane.getLength(lane_id)
        lane_position = 0.0
        # Try to get precise position if vehicle is already in SUMO
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getLaneID(vid) == lane_id:
                lane_position = traci.vehicle.getLanePosition(vid)
                break

        dist_to_stop_line = max(1.0, lane_length - lane_position)
        speed_advice = vehicle_speed  # default: keep current speed

        if signal_ahead == "red" and secs_until_switch > 0:
            # Option A: slow down to arrive exactly when green starts
            ideal_speed = dist_to_stop_line / secs_until_switch
            # Clamp between 2 m/s (don't stop) and 13.89 m/s (speed limit)
            speed_advice = max(2.0, min(13.89, ideal_speed))
        elif signal_ahead == "green" and secs_until_switch > 0:
            # Option B: make sure we CLEAR the junction before yellow
            ideal_speed = dist_to_stop_line / secs_until_switch
            speed_advice = max(vehicle_speed, min(13.89, ideal_speed))

        return {
            "signal_ahead": signal_ahead,
            "time_to_green": time_to_green,
            "speed_advice": round(speed_advice, 2),
            "emergency_mode": bool(
                state.emergency_agent and state.emergency_agent.is_active
            ),
        }

    except Exception as e:
        # TraCI errors (connection lost, invalid lane) → safe defaults
        print(f"  ⚠ Signal advice error: {e}")
        return defaults


# ──────────────────────────────────────────────────────────────────────
#  Vehicle Injection / Position Update
# ──────────────────────────────────────────────────────────────────────

def _inject_or_update_vehicle(
    vehicle_id: str,
    sumo_x: float,
    sumo_y: float,
    speed: float,
    is_emergency: bool,
    edge_id: str,
) -> None:
    """
    Add the vehicle to SUMO if it doesn't exist, or teleport it
    to the latest GPS-derived position using moveToXY.

    Normal SUMO routing vs. moveToXY
    ─────────────────────────────────
    Normal: SUMO drives the vehicle step-by-step along a declared route.
    moveToXY: We override the vehicle's position each step with real GPS.

    keepRoute=2 snaps the vehicle to the nearest lane from (x, y),
    ignoring any declared route.  This handles GPS drift (a reading
    a few metres off the road) without causing errors.

    angle=0 (north) is a placeholder; SUMO uses this for rendering only.
    """
    if not state._sumo_running:
        return

    try:
        existing_ids = traci.vehicle.getIDList()

        if vehicle_id not in existing_ids:
            # ── First appearance: add to simulation ───────────────────
            vType = "emergency_vehicle" if is_emergency else "normal_car"

            # Create a minimal one-edge route at the entry edge.
            # moveToXY will override the actual position immediately,
            # but SUMO requires a route to add a vehicle.
            route_id = f"_route_{vehicle_id}"
            if edge_id:
                try:
                    traci.route.add(route_id, [edge_id])
                except traci.TraCIException:
                    pass  # route may already exist from a previous session

            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route_id,
                typeID=vType,
                departLane="best",
                departSpeed="0",
            )

            # Immediately teleport to GPS position so it appears correctly
            traci.vehicle.moveToXY(
                vehID=vehicle_id,
                edgeID="",      # empty = let SUMO figure it out
                laneIndex=0,
                x=sumo_x,
                y=sumo_y,
                angle=0.0,
                keepRoute=2,    # snap to nearest lane, ignore declared route
            )
            traci.vehicle.setSpeed(vehicle_id, speed)
            print(f"  🚗 Vehicle added to SUMO: {vehicle_id} ({vType})")

        else:
            # ── Already present: update position to latest GPS fix ─────
            # This is the key difference from normal routing:
            # instead of letting SUMO compute motion, we FORCE the
            # position from real-world GPS every second.
            traci.vehicle.moveToXY(
                vehID=vehicle_id,
                edgeID="",
                laneIndex=0,
                x=sumo_x,
                y=sumo_y,
                angle=0.0,
                keepRoute=2,
            )
            traci.vehicle.setSpeed(vehicle_id, speed)

    except Exception as e:
        print(f"  ⚠ SUMO vehicle update error({vehicle_id}): {e}")


def _remove_vehicle_from_sumo(vehicle_id: str) -> None:
    """Remove a vehicle from the running SUMO simulation."""
    if not state._sumo_running:
        return
    try:
        existing = traci.vehicle.getIDList()
        if vehicle_id in existing:
            traci.vehicle.remove(vehicle_id)
            print(f"  🗑  Vehicle removed from SUMO: {vehicle_id}")
    except Exception as e:
        print(f"  ⚠ Error removing vehicle {vehicle_id}: {e}")


# ──────────────────────────────────────────────────────────────────────
#  WebSocket: /ws/vehicle/{vehicle_id}
# ──────────────────────────────────────────────────────────────────────

@router.websocket("/ws/vehicle/{vehicle_id}")
async def vehicle_websocket(websocket: WebSocket, vehicle_id: str):
    """
    Real-time two-way WebSocket channel for a single phone/vehicle.

    Phone sends every ~1 second:
        {
          "vehicle_id":    str,
          "latitude":      float,
          "longitude":     float,
          "speed":         float,          // m/s
          "acceleration":  float,          // m/s²
          "is_emergency":  bool,
          "timestamp":     float           // epoch seconds
        }

    Server responds immediately:
        {
          "signal_ahead":  str,            // "green" | "yellow" | "red"
          "time_to_green": int,            // seconds until next green
          "speed_advice":  float,          // recommended m/s
          "emergency_mode": bool
        }
    """
    mapper = _get_mapper()

    # ── Connect ───────────────────────────────────────────────────────
    await state.manager.connect_vehicle(vehicle_id, websocket)
    _active_vehicles[vehicle_id] = {
        "vehicle_id": vehicle_id,
        "lat": 0.0,
        "lng": 0.0,
        "speed": 0.0,
        "edge_id": "",
        "is_emergency": False,
        "connected_at": time.time(),
    }

    # Send welcome handshake
    await state.manager.send_to_vehicle(
        vehicle_id,
        {
            "type": "connected",
            "vehicle_id": vehicle_id,
            "demo_bounds": mapper.get_demo_bounds(),
            "timestamp": time.time(),
        },
    )

    try:
        while True:
            raw = await websocket.receive_text()

            # ── Parse incoming GPS update ─────────────────────────────
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await state.manager.send_to_vehicle(
                    vehicle_id, {"error": "Invalid JSON"}
                )
                continue

            lat = float(payload.get("latitude", 0.0))
            lng = float(payload.get("longitude", 0.0))
            speed = float(payload.get("speed", 0.0))
            is_emergency = bool(payload.get("is_emergency", False))

            # ── Map GPS → SUMO coordinates ────────────────────────────
            # CoordinateMapper handles out-of-bounds GPS gracefully
            # by returning a random valid edge (demo fallback mode).
            if mapper.is_within_bounds(lat, lng):
                sumo_x, sumo_y = mapper.gps_to_sumo(lat, lng)
                edge_id = mapper.find_nearest_edge(sumo_x, sumo_y) or ""
            else:
                # Out of bounds — use demo fallback edge, centre of network
                edge_id = mapper.map_to_random_edge()
                boundary = mapper._net.getBoundary()
                sumo_x = (boundary[0] + boundary[2]) / 2
                sumo_y = (boundary[1] + boundary[3]) / 2

            # ── Update active vehicle registry ────────────────────────
            _active_vehicles[vehicle_id].update({
                "lat": lat, "lng": lng, "speed": speed,
                "edge_id": edge_id, "is_emergency": is_emergency,
            })

            # ── Inject / update vehicle in SUMO ───────────────────────
            _inject_or_update_vehicle(
                vehicle_id, sumo_x, sumo_y, speed, is_emergency, edge_id
            )

            # ── Trigger emergency corridor if flagged ─────────────────
            if is_emergency and state.emergency_agent:
                if not state.emergency_agent.is_active:
                    # Default destination: opposite side of the intersection
                    _destinations = {
                        "south_to_center": "center_to_north",
                        "north_to_center": "center_to_south",
                        "east_to_center":  "center_to_west",
                        "west_to_center":  "center_to_east",
                    }
                    dest_edge = _destinations.get(edge_id, "center_to_north")
                    state.emergency_agent.activate(
                        emergency_vehicle_id=vehicle_id,
                        origin=edge_id,
                        destination=dest_edge,
                    )

            # ── Compute signal advice for this vehicle's position ─────
            advice = _get_signal_advice(sumo_x, sumo_y, speed)

            # ── Reply to phone ─────────────────────────────────────────
            await state.manager.send_to_vehicle(vehicle_id, advice)

    except WebSocketDisconnect:
        pass

    finally:
        # ── Disconnect cleanup ────────────────────────────────────────
        _remove_vehicle_from_sumo(vehicle_id)

        # Deactivate emergency if this vehicle was running a corridor
        if (
            state.emergency_agent
            and state.emergency_agent.is_active
            and state.emergency_agent._vehicle_id == vehicle_id
        ):
            state.emergency_agent.deactivate()

        state.manager.disconnect_vehicle(vehicle_id)
        _active_vehicles.pop(vehicle_id, None)
        print(f"  📴 Vehicle disconnected: {vehicle_id}")


# ──────────────────────────────────────────────────────────────────────
#  REST: GET /vehicles/active
# ──────────────────────────────────────────────────────────────────────

@router.get("/active", summary="List all currently connected vehicle clients")
async def get_active_vehicles() -> Dict[str, Any]:
    """
    Return metadata for all phone clients currently connected.

    Used by the dashboard to render vehicle icons on the map.
    Each entry includes the last known GPS position, speed, and
    whether the vehicle is in emergency mode.
    """
    # Enrich with SUMO positions if simulation is running
    enriched = []
    for vid, vdata in _active_vehicles.items():
        entry = dict(vdata)
        if state._sumo_running:
            try:
                if vid in traci.vehicle.getIDList():
                    entry["sumo_edge"] = traci.vehicle.getRoadID(vid)
                    entry["sumo_waiting_time"] = round(
                        traci.vehicle.getWaitingTime(vid), 2
                    )
            except Exception:
                pass
        enriched.append(entry)

    return {
        "count": len(enriched),
        "vehicles": enriched,
        "simulation_running": state._sumo_running,
    }
