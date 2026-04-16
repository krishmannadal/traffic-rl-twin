"""
main.py — FastAPI Application Entry Point
==========================================

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

This file is the single central hub for the entire API.  It:
  1. Creates the FastAPI application instance
  2. Adds middleware (CORS)
  3. Mounts all route sub-routers
  4. Manages application-wide global state (SUMO env, agents)
  5. Defines startup / shutdown lifecycle events

WHAT CORS IS AND WHY WE NEED IT FOR LOCAL DEVELOPMENT
───────────────────────────────────────────────────────
CORS (Cross-Origin Resource Sharing) is a browser security mechanism.
When a webpage at origin A (e.g., http://localhost:3000 — your React
dashboard) tries to make an HTTP request to origin B (http://localhost:8000
— this FastAPI server), the browser BLOCKS the request unless origin B
explicitly permits it via CORS headers in its response.

Without CORS middleware your React/Expo frontend would see:
    "Access to fetch at 'http://localhost:8000/health' from origin
     'http://localhost:3000' has been blocked by CORS policy."

We allow ALL origins ("*") during development because:
  • React dev server runs on :3000
  • Expo (React Native) may run on :19006 or a mobile device IP
  • SUMO-GUI and Jupyter notebooks may make ad-hoc requests
  • You don't want to whitelist every port manually while iterating

In PRODUCTION you would replace "*" with specific allowed origins:
    allow_origins=["https://your-dashboard.com"]

CORS is a BROWSER enforcement — Python scripts, curl, and Postman
bypass it entirely.  It only affects browser-side requests.

WHY WE INITIALIZE GLOBALS HERE VS INSIDE ROUTES
─────────────────────────────────────────────────
Each HTTP request to a FastAPI route handler is served by a fresh
function call.  If we created the SumoEnvironment inside a route
handler, it would:
  1. Spawn a new SUMO process on EVERY request.
  2. Never be cleaned up (no persistent reference → garbage collected
     after the request, leaving orphan SUMO processes holding ports).
  3. Have no shared state between requests (two requests can't observe
     the same simulation).

By initialising globals in main.py at startup:
  • ONE SUMO process is shared across all routes and all clients.
  • Routes import and MUTATE the shared state (start sim, step, stop).
  • Cleanup happens in the shutdown event, not scattered across routes.
  • WebSocket clients all see the same simulation step counter.

The pattern is standard FastAPI: global state in main.py, imported
by routes via `from api.main import sumo_env, traffic_agent` etc.

WHAT LIFESPAN EVENTS ARE IN FASTAPI
─────────────────────────────────────
FastAPI (via Starlette) provides lifespan context managers that run
code at application startup and shutdown:

  @asynccontextmanager
  async def lifespan(app):
      # code here runs BEFORE the first request
      yield
      # code here runs AFTER the last request / on server shutdown

This replaced the older @app.on_event("startup") / ("shutdown")
decorators (still available but deprecated since FastAPI 0.93).

Startup:  perfect for connecting to databases, loading ML models,
          warming up caches, registering background tasks.
Shutdown: perfect for closing DB connections, saving state, stopping
          SUMO so it doesn't leak processes on the host.

Unlike middleware, lifespan events run ONCE per server process, not
once per request.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Local imports ──────────────────────────────────────────────────────
from simulation.environment import SumoEnvironment
from simulation.traffic_env import TrafficEnv
from simulation.emergency_env import EmergencyEnv
from agents.traffic_agent import TrafficAgent
from agents.emergency_agent import EmergencyAgent
from api.websocket import ConnectionManager, ws_router

# Route sub-routers
from api.routes import simulation, agents, metrics, vehicles, admin, user


# ──────────────────────────────────────────────────────────────────────
#  Application-wide Globals
# ──────────────────────────────────────────────────────────────────────
# These objects are created ONCE at startup and shared across all routes.
# Routes import them with: from api.main import sumo_env, manager, ...

# WebSocket connection manager — tracks all live browser/mobile clients
# (Already initialized in state.manager)

# Low-level SUMO bridge — not yet started (no SUMO process yet)
from api import state

# ──────────────────────────────────────────────────────────────────────
#  Lifespan Context Manager
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: runs startup code, yields control to FastAPI,
    then runs shutdown code when the server terminates.
    """
    # ── STARTUP ───────────────────────────────────────────────────────
    print("=" * 55)
    print("  [API] Traffic RL Twin API starting up...")
    print("=" * 55)
    print("  Endpoints ready at http://localhost:8000")
    print("  Interactive docs at http://localhost:8000/docs")
    print("=" * 55)

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    print("\n[STOP] Traffic RL Twin API shutting down...")
    
    # 1. Halt Background Simulation
    state._sumo_running = False
    
    # 2. Cleanup Environments
    try:
        # Use emergency_env as our primary handle (it inherits from TrafficEnv/SumoEnvironment)
        if state.emergency_env:
            state.emergency_env.stop()
            print("  [OK] SUMO process terminated.")
        elif state.sumo_env:
            state.sumo_env.stop()
            print("  [OK] SUMO process terminated.")
    except Exception as e:
        print(f"  [WARN] Error during SUMO cleanup: {e}")
    
    print("  [DONE] Shutdown complete")


# ──────────────────────────────────────────────────────────────────────
#  App Instantiation
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Traffic RL Twin API",
    description=(
        "REST + WebSocket API for the Traffic RL Twin project. "
        "Controls SUMO traffic simulation, RL agent training, "
        "and real-time metric streaming for the dashboard."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────
#  CORS Middleware
# ──────────────────────────────────────────────────────────────────────
# Allow all origins during development.
# See module docstring for why "*" is safe for local dev and what to
# change for production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # All origins — see docstring
    allow_credentials=True,        # Allow cookies / auth headers
    allow_methods=["*"],           # GET, POST, PUT, DELETE, OPTIONS...
    allow_headers=["*"],           # Content-Type, Authorization, X-Custom...
)


# ──────────────────────────────────────────────────────────────────────
#  Mount Sub-Routers
# ──────────────────────────────────────────────────────────────────────
# Each router is defined in its own file for separation of concerns.
# Prefixes are added here so endpoint files stay clean.

app.include_router(
    simulation.router,
    prefix="/simulation",
    tags=["Simulation"]
)

app.include_router(
    agents.router,
    prefix="/agents",
    tags=["Agents"]
)

app.include_router(
    metrics.router,
    prefix="/metrics",
    tags=["Metrics"]
)

app.include_router(
    vehicles.router,
    prefix="/vehicles",
    tags=["Vehicles"]
)

app.include_router(
    admin.router,
    prefix="/admin",
    tags=["Admin"]
)

app.include_router(
    user.router,
    prefix="/user",
    tags=["User"]
)

# WebSocket endpoints (Dashboard & Mobile)
app.include_router(ws_router)


# ──────────────────────────────────────────────────────────────────────
#  Root Endpoint
# ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    API index — returns status and a map of available endpoints.

    Useful for the frontend to discover what the API supports
    without hard-coding endpoint lists.
    """
    return {
        "name": "Traffic RL Twin API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "docs":           "/docs",
            "health":         "/health",
            "websocket":      "/ws",
            "simulation": {
                "start":      "POST /simulation/start",
                "stop":       "POST /simulation/stop",
                "step":       "POST /simulation/step",
                "state":      "GET  /simulation/state",
                "reset":      "POST /simulation/reset",
            },
            "agents": {
                "train":      "POST /agents/train",
                "predict":    "POST /agents/predict",
                "status":     "GET  /agents/status",
                "load":       "POST /agents/load",
                "emergency":  "POST /agents/emergency/trigger",
            },
            "metrics": {
                "current":    "GET  /metrics/current",
                "history":    "GET  /metrics/history",
                "summary":    "GET  /metrics/summary",
            },
            "vehicles": {
                "list":       "GET  /vehicles/",
                "info":       "GET  /vehicles/{vehicle_id}",
                "emergency":  "GET  /vehicles/emergency",
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
#  Health Check
# ──────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Root"])
async def health_check() -> Dict[str, Any]:
    """
    Health probe for container orchestration / frontend polling.

    Returns the current operational state of all major components so
    the dashboard can display accurate status indicators:
      • api_status       — always "ok" if this endpoint is reachable
      • sumo_running     — whether a SUMO simulation is active
      • agent_loaded     — whether the DQN model weights are loaded
      • connected_clients — active WebSocket connections (browser tabs)
    """
    return {
        "api_status": "ok",
        "sumo_running": state._sumo_running,
        "agent_loaded": getattr(state, "_agent_loaded", False),
        "connected_clients": state.manager.active_connections_count,
    }


