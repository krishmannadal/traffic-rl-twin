from typing import Optional, Any
from simulation.environment import SumoEnvironment
from simulation.emergency_env import EmergencyEnv
from agents.traffic_agent import TrafficAgent
from agents.emergency_agent import EmergencyAgent
from api.websocket import ConnectionManager

# ── Global State ─────────────────────────────────────────────────────────────
# These are initialized in api/main.py during the 'lifespan' event

_sumo_running: bool = False
_agent_loaded: bool = False
manager: ConnectionManager = ConnectionManager()

# Environments
traffic_env: Optional[SumoEnvironment] = None
emergency_env: Optional[EmergencyEnv] = None

# Agents
traffic_agent: Optional[TrafficAgent] = None
emergency_agent: Optional[EmergencyAgent] = None

sumo_env: Optional[SumoEnvironment] = None # For backward compatibility or general access
