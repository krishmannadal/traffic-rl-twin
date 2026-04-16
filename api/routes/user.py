from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional, Dict
import traci
import uuid
import time
from api.state import sumo_env, manager

router = APIRouter()

class UserRegisterRequest(BaseModel):
    device_id: str
    vehicle_type: str = "car"

class UserRouteRequest(BaseModel):
    origin_edge: str
    destination_edge: str
    vehicle_type: str = "car"

@router.post("/register")
async def register_user(request: UserRegisterRequest):
    """
    Register a mobile device and assign a virtual vehicle.
    """
    user_id = str(uuid.uuid4())
    vehicle_id = f"user_{request.device_id[:8]}_{int(time.time())}"
    
    return {
        "user_id": user_id,
        "vehicle_id": vehicle_id,
        "registration_time": time.time()
    }

@router.post("/route")
async def get_route(request: UserRouteRequest):
    """
    Calculate optimal route via TraCI.
    """
    if not sumo_env.is_connected:
        raise HTTPException(status_code=400, detail="Simulation not running")

    try:
        route_info = traci.simulation.findRoute(request.origin_edge, request.destination_edge)
        
        if not route_info.edges:
            raise HTTPException(status_code=404, detail="No route found")

        # Estimate travel time based on distance and speed limits
        distance = route_info.length
        travel_time = route_info.travelTime
        
        # Determine congestion level
        congestion = "low"
        if travel_time > (distance / 13.89) * 1.5: # More than 50% slower than 50km/h
            congestion = "high"
        elif travel_time > (distance / 13.89) * 1.2:
            congestion = "medium"

        return {
            "route": list(route_info.edges),
            "estimated_time": int(travel_time),
            "distance_meters": round(distance, 2),
            "signal_count": sum(1 for e in route_info.edges if traci.trafficlight.getIDList()), # Rough approx
            "current_congestion": congestion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signal_ahead/{vehicle_id}")
async def get_signal_ahead(vehicle_id: str):
    """
    Look ahead on vehicle's route for the next traffic light.
    """
    if not sumo_env.is_connected:
        raise HTTPException(status_code=400, detail="Simulation not running")

    try:
        # Find vehicle position and route
        if vehicle_id not in traci.vehicle.getIDList():
            raise HTTPException(status_code=404, detail="Vehicle not found in simulation")
            
        route = traci.vehicle.getRoute(vehicle_id)
        current_idx = traci.vehicle.getRouteIndex(vehicle_id)
        remaining_edges = route[current_idx:]
        
        # Scan for next signal
        for edge_id in remaining_edges:
            # Check if this edge is controlled by a TLS
            tls_ids = traci.trafficlight.getIDList()
            for tls in tls_ids:
                links = traci.trafficlight.getControlledLinks(tls)
                for link_group in links:
                    for link in link_group:
                        if link[0].startswith(edge_id):
                            # Found it
                            state = traci.trafficlight.getRedYellowGreenState(tls)
                            # Current demo assumes simple NS/EW phases
                            # This is a placeholder for real logic
                            is_green = 'G' in state.upper() 
                            
                            return {
                                "signal_id": tls,
                                "current_phase": "GREEN" if is_green else "RED",
                                "time_to_green": traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime(),
                                "recommended_speed": 13.89 if is_green else 5.0
                            }
        
        return {"message": "No signals remaining on route"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/eta/{vehicle_id}")
async def get_eta(vehicle_id: str):
    """
    Calculate real-time ETA for the assigned user vehicle.
    """
    if vehicle_id not in traci.vehicle.getIDList():
        return {"eta_seconds": 0, "distance_remaining": 0.0, "status": "not_in_simulation"}

    route = traci.vehicle.getRoute(vehicle_id)
    idx = traci.vehicle.getRouteIndex(vehicle_id)
    rem = route[idx:]
    
    # Simple distance sum
    dist = sum(traci.lane.getLength(f"{e}_0") for e in rem)
    speed = traci.vehicle.getSpeed(vehicle_id)
    if speed < 0.1: speed = 5.0 # Avoid div by zero
    
    eta = dist / speed
    
    return {
        "eta_seconds": int(eta),
        "distance_remaining": round(dist, 2),
        "congestion_ahead": "moderate",
        "ai_time_saved": 45 # Mock for demo
    }
