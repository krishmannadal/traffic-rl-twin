from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import traci
from simulation.net_converter import NetConverter
from api.state import sumo_env

router = APIRouter()
converter = NetConverter()

class MapNode(BaseModel):
    id: str
    x: float
    y: float
    type: str = "priority"

class MapEdge(BaseModel):
    id: str
    from_node: str
    to_node: str
    lanes: int = 1
    speed: float = 13.89

class MapBuildRequest(BaseModel):
    nodes: List[MapNode]
    edges: List[MapEdge]
    map_name: str = "custom"

class VehicleInjectionRequest(BaseModel):
    edge_id: str
    count: int
    vehicle_type: str = "car"
    destination_edge: Optional[str] = None

@router.get("/map/list")
async def list_maps():
    """List all custom maps available."""
    from pathlib import Path
    maps_dir = Path("simulation/sumo_configs/custom_maps")
    if not maps_dir.exists():
        return {"maps": []}
        
    maps = []
    for file in maps_dir.glob("*.net.xml"):
        name = file.name.replace(".net.xml", "")
        maps.append(name)
        
    return {"maps": maps}

@router.post("/map/build")
async def build_map(request: MapBuildRequest):
    """
    Convert canvas coordinates to SUMO XML and generate network.
    """
    # map_nodes = [{"id": n.id, "x": n.x, "y": n.y, "type": n.type} for n in request.nodes]
    # map_edges = [{"id": e.id, "from": e.from_node, "to": e.to_node, "lanes": e.lanes, "speed": e.speed} for e in request.edges]
    
    # We use field names from the request model mapped to NetConverter requirements
    map_nodes = [node.dict() for node in request.nodes]
    map_edges = []
    for edge in request.edges:
        e_dict = edge.dict()
        e_dict["from"] = e_dict.pop("from_node")
        e_dict["to"] = e_dict.pop("to_node")
        map_edges.append(e_dict)

    result = converter.convert_to_sumo(map_nodes, map_edges, map_name=request.map_name)
    
    if result["success"] == "false":
        raise HTTPException(status_code=500, detail=result.get("error", "Map build failed"))
    
    return result

@router.post("/simulation/add_vehicles")
async def add_vehicles(request: VehicleInjectionRequest):
    """
    Inject vehicles into the running simulation with route validation.
    """
    if not sumo_env.is_connected:
        raise HTTPException(status_code=400, detail="Simulation not running")

    try:
        vehicles_added = []
        for i in range(request.count):
            veh_id = f"{request.vehicle_type}_{request.edge_id}_{traci.simulation.getTime()}_{i}"
            
            # 1. Route Validation
            # If no destination given, try to find a valid one or use current edge as a trivial route
            dest = request.destination_edge
            if not dest:
                # Fallback: get any connected edge or just use origin as a 1-edge route
                dest = request.edge_id
            
            try:
                # Verify connectivity
                route_info = traci.simulation.findRoute(request.edge_id, dest)
                if not route_info.edges:
                    raise ValueError(f"No path from {request.edge_id} to {dest}")
                
                route_id = f"route_{veh_id}"
                traci.route.add(route_id, list(route_info.edges))
                
                # 2. Injection
                traci.vehicle.add(veh_id, route_id, typeID=request.vehicle_type)
                vehicles_added.append(veh_id)
            except Exception as e:
                print(f"Failed to add vehicle {veh_id}: {e}")
                continue

        return {
            "status": "partial" if len(vehicles_added) < request.count else "success",
            "vehicles_added": vehicles_added,
            "count": len(vehicles_added),
            "total_in_sim": traci.vehicle.getIDCount()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
