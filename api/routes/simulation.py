"""
routes/simulation.py — Simulation Control Endpoints
===================================================
"""
import asyncio
import time
import uuid
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

import api.state as state
from api.websocket import broadcast_simulation_state
from api.routes.metrics import metrics_history, MAX_HISTORY
from simulation.emergency_env import EmergencyEnv
from agents.traffic_agent import TrafficAgent
from agents.emergency_agent import EmergencyAgent

router = APIRouter()

# ── Local State ────────────────────────────────────────────────────────
_loop_active = False
_needs_reset = False
_current_step = 0
_simulation_time = 0.0
_speed_multiplier = 1.0


# ── Request Models ─────────────────────────────────────────────────────
class StartRequest(BaseModel):
    map_name: Optional[str] = None

class EmergencyRequest(BaseModel):
    origin_edge: str
    destination_edge: str

class SpeedRequest(BaseModel):
    multiplier: float


# ── Helpers ────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    config_path = Path("training/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def _init_envs_if_needed(map_name: Optional[str] = None):
    """Lazy initialize the environments and agents if not already done."""
    config = _load_config()
    
    net_file = None
    if map_name and map_name != "default":
        net_file = f"simulation/sumo_configs/custom_maps/{map_name}.net.xml"
        # Recreate env if it exists to load the new map
        if state.emergency_env is not None:
            try:
                state.emergency_env.close()
            except Exception:
                pass
            state.emergency_env = None
            state.traffic_env = None
            state.traffic_agent = None
            state.emergency_agent = None
    
    if state.emergency_env is None:
        sumo_cfg = config.get("sumo", {}).get("config_path", "simulation/sumo_configs/simulation.sumocfg")
        base_port = config.get("sumo", {}).get("base_port", 8813)
        
        # Use EmergencyEnv as our primary env since it supports both normal + emergency logic
        state.emergency_env = EmergencyEnv(
            config_path=sumo_cfg, 
            net_file=net_file,
            port=base_port, 
            use_gui=False, 
            max_steps=864000  # very large max_steps for continuous demo
        )
        state.traffic_env = state.emergency_env
        
    if state.traffic_agent is None:
        state.traffic_agent = TrafficAgent(env=state.emergency_env)
        
    if state.emergency_agent is None:
        state.emergency_agent = EmergencyAgent(env=state.emergency_env)


# ── Background Task ────────────────────────────────────────────────────

async def simulation_loop():
    """
    Continuous background loop that steps the SUMO simulation and
    broadcasts the state to WebSocket clients via the ConnectionManager.
    """
    global _loop_active, _needs_reset, _current_step, _simulation_time
    
    env = state.emergency_env
    t_agent = state.traffic_agent
    e_agent = state.emergency_agent
    
    # Initial reset
    obs, info = env.reset()
    _current_step = 0
    _simulation_time = 0.0
    
    config = _load_config()
    base_step_length = config.get("sumo", {}).get("step_length", 1.0)
    
    while _loop_active and state._sumo_running:
        start_time = time.time()
        
        # Handle reset request gracefully between steps
        if _needs_reset:
            obs, info = env.reset()
            _current_step = 0
            _simulation_time = 0.0
            _needs_reset = False
            
        # 1. Decide which agent has control
        if e_agent and e_agent.is_active:
            # Emergency is active. PPO agent dictates control if in learned mode,
            # otherwise rule-based runs automatically via env override.
            if e_agent.mode == "learned":
                action, _ = e_agent.predict(obs, deterministic=True)
            else:
                action = 0  # Dummy action, ignored by rule-based EmergencyEnv logic
        else:
            # Normal traffic operations
            if t_agent:
                action, _ = t_agent.predict(obs, deterministic=True)
            else:
                # Fallback purely random action if not fully loaded/trained
                action = env.action_space.sample()
                
        # 2. Advance the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # ── SYNC AGENT STATE ──────────────────────────────────────────
        # If the environment just finished an emergency corridor (e.g. vehicle left)
        # but the agent object is still flagged as active, deactivate it to 
        # ensure metrics are logged and the system reverts to normal control.
        if e_agent and e_agent.is_active and not info.get("emergency_active", False):
            e_agent.deactivate()
            # print("  [Loop] Sync: Emergency agent deactivated.")
        
        _current_step += 1
        _simulation_time = float(info.get("sim_step", _current_step))
        
        # 3. Broadcast to WebSockets
        try:
            await broadcast_simulation_state(
                manager=state.manager,
                env_state=info,
                current_reward=reward,
                step=_current_step
            )
        except Exception as e:
            print(f"Error broadcasting state: {e}")
        
        # 3b. Record to in-memory metrics history (feeds /metrics/* endpoints)
        try:
            total_wait = info.get("total_waiting_time", 0.0)
            total_queue = info.get("total_queue_length", 0.0)
            metrics_history.append({
                "step": _current_step,
                "reward": float(reward),
                "waiting_time": float(total_wait),
                "queue_length": float(total_queue),
                "timestamp": time.time(),
            })
            # Maintain circular buffer size
            if len(metrics_history) > MAX_HISTORY:
                metrics_history.pop(0)
        except Exception:
            pass  # Non-critical — don't crash the sim loop for metrics
            
        # 4. Handle episode boundaries
        if terminated or truncated:
            obs, info = env.reset()
            
        # 5. Timing constraint (enforce simulation speed)
        # target_delay is how long this step SHOULD take in wall-clock time
        target_delay = base_step_length / max(0.1, _speed_multiplier)
        elapsed = time.time() - start_time
        sleep_time = max(0.001, target_delay - elapsed)  # Min 1ms sleep to yield event loop
        
        await asyncio.sleep(sleep_time)


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/start", summary="Start the SUMO simulation")
async def start_simulation(background_tasks: BackgroundTasks, request: Optional[StartRequest] = None) -> Dict[str, Any]:
    """Start the background simulation loop and the SUMO process."""
    global _loop_active
    
    if state._sumo_running:
        raise HTTPException(
            status_code=400, detail="Simulation is already running."
        )
        
    try:
        map_name = request.map_name if request else None
        _init_envs_if_needed(map_name=map_name)
        
        state._sumo_running = True
        _loop_active = True
        
        background_tasks.add_task(simulation_loop)
        
        return {"status": "started", "simulation_time": 0.0}
        
    except Exception as e:
        state._sumo_running = False
        _loop_active = False
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", summary="Stop the simulation")
async def stop_simulation() -> Dict[str, Any]:
    """Halt the simulation loop and cleanly shut down SUMO."""
    global _loop_active
    
    if not state._sumo_running:
        raise HTTPException(
            status_code=400, detail="Simulation is not currently running."
        )
        
    _loop_active = False
    state._sumo_running = False
    
    if state.emergency_env:
        # Closing the environment frees the TraCI port
        state.emergency_env.close()
        
    return {"status": "stopped", "final_time": _simulation_time}


@router.post("/reset", summary="Reset simulation state")
async def reset_simulation() -> Dict[str, Any]:
    """Reset the running simulation back to t=0."""
    global _needs_reset
    
    if not state._sumo_running:
        raise HTTPException(
            status_code=400, detail="Start the simulation before resetting."
        )
        
    _needs_reset = True
    return {"status": "reset"}


@router.post("/emergency", summary="Trigger an emergency vehicle")
async def trigger_emergency(req: EmergencyRequest) -> Dict[str, Any]:
    """
    Inject an emergency vehicle at the specified origin edge and route
    it to the destination edge, activating the green-corridor priority.
    """
    if not state._sumo_running or not state.emergency_agent:
        raise HTTPException(
            status_code=400,
            detail="Simulation must be running to trigger an emergency."
        )
        
    try:
        # Generate short unique ID
        veh_id = f"emergency_{uuid.uuid4().hex[:6]}"
        
        # This spawns the vehicle in SUMO and activates corridor logic
        est_time = state.emergency_agent.activate(
            emergency_vehicle_id=veh_id,
            origin=req.origin_edge,
            destination=req.destination_edge
        )
        
        return {
            "status": "emergency_triggered",
            "vehicle_id": veh_id,
            "estimated_travel_time": round(est_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger emergency: {e}")


@router.get("/status", summary="Get core API and simulation status")
async def get_status() -> Dict[str, Any]:
    """Current top-level state of the simulation platform."""
    agent_mode = "unknown"
    if state.emergency_agent and state.emergency_agent.is_active:
        agent_mode = "emergency"
    elif state.traffic_agent:
        agent_mode = "traffic"
        
    return {
        "running": state._sumo_running,
        "current_step": _current_step,
        "simulation_time": _simulation_time,
        "agent_mode": agent_mode,
        "connected_clients": state.manager.active_connections_count
    }


@router.post("/speed", summary="Adjust simulation playback speed")
async def set_speed(req: SpeedRequest) -> Dict[str, Any]:
    """
    Change how fast the simulation loop advances. 
    multiplier: 1.0 = real-time, 5.0 = 5x faster, 0.5 = half speed.
    """
    global _speed_multiplier
    
    if req.multiplier <= 0.0:
        raise HTTPException(
            status_code=400, detail="Speed multiplier must be strictly positive."
        )
        
    _speed_multiplier = float(req.multiplier)
    return {"speed_multiplier": _speed_multiplier}
