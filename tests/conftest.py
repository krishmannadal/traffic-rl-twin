"""
conftest.py — Shared pytest fixtures for the Traffic RL Twin project.

Provides mock environments, agents, and test clients so that all tests
can run WITHOUT a live SUMO installation or GPU.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import gymnasium as gym
from gymnasium import spaces

# ──────────────────────────────────────────────────────────────────────
#  Mock SUMO Environment
# ──────────────────────────────────────────────────────────────────────

class MockTrafficEnv(gym.Env):
    """
    A lightweight mock of TrafficEnv that returns valid shapes without SUMO.
    
    Observation space: 17 floats (8 queue lengths + 4 wait times + 4 phase one-hot + 1 emergency)
    Action space: 4 discrete actions (one per phase)
    """
    metadata = {"render_modes": []}
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self._step_count = 0
        self._max_steps = kwargs.get('max_steps', 100)
        self._emergency_active = False
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.random.rand(17).astype(np.float32)
        info = {
            "sim_step": 0,
            "total_waiting_time": 0.0,
            "total_queue_length": 0.0,
            "emergency_active": False,
        }
        return obs, info
    
    def step(self, action):
        self._step_count += 1
        obs = np.random.rand(17).astype(np.float32)
        reward = float(np.random.uniform(-1, 1))
        terminated = False
        truncated = self._step_count >= self._max_steps
        info = {
            "sim_step": self._step_count,
            "total_waiting_time": float(np.random.uniform(0, 50)),
            "total_queue_length": float(np.random.uniform(0, 20)),
            "emergency_active": self._emergency_active,
            "queue_per_direction": {"N": 2.0, "S": 3.0, "E": 1.5, "W": 2.5},
        }
        return obs, reward, terminated, truncated, info
    
    def close(self):
        pass
    
    def render(self):
        return {}


class MockEmergencyEnv(MockTrafficEnv):
    """Extended mock that supports emergency methods."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emergency_route = ["south_to_center", "center_to_north"]
        self.emergency_log = []
    
    @property
    def emergency_active(self):
        return self._emergency_active
    
    def trigger_emergency(self, vehicle_id, origin_edge, destination_edge):
        self._emergency_active = True
        return 45.0  # baseline travel time estimate
    
    def deactivate_emergency(self):
        self._emergency_active = False
        result = {
            "emergency_travel_time": 30.0,
            "baseline_travel_time": 45.0,
            "time_saved": 15.0,
            "disruption": 0.15,
        }
        self.emergency_log.append(result)
        return result
    
    def get_green_corridor(self, route):
        return [{"tl_id": "center", "phase": 0}]
    
    def apply_corridor(self, corridor):
        pass


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_traffic_env():
    """Returns a mock TrafficEnv for testing without SUMO."""
    return MockTrafficEnv(max_steps=50)


@pytest.fixture
def mock_emergency_env():
    """Returns a mock EmergencyEnv for testing without SUMO."""
    return MockEmergencyEnv(max_steps=50)


@pytest.fixture
def sample_observation():
    """Returns a valid 17-dim observation vector."""
    return np.random.rand(17).astype(np.float32)


@pytest.fixture
def sample_config():
    """Returns a minimal config dict for testing."""
    return {
        "sumo": {
            "config_path": "simulation/sumo_configs/simulation.sumocfg",
            "base_port": 8813,
            "step_length": 1.0,
            "max_steps": 100,
        },
        "environment": {
            "sumo_cfg": "simulation/sumo_configs/simulation.sumocfg",
            "base_port": 8813,
            "num_envs": 1,
            "max_steps": 100,
        },
        "dqn": {
            "policy": "MlpPolicy",
            "net_arch": [64, 64],
            "learning_rate": 1e-4,
            "buffer_size": 1000,
            "batch_size": 32,
            "gamma": 0.99,
        },
        "ppo": {
            "policy": "MlpPolicy",
            "net_arch": [64, 64],
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 3,
            "gamma": 0.95,
        },
        "training": {
            "total_timesteps": 100,
            "save_frequency": 50,
        },
        "evaluation": {
            "eval_episodes": 2,
            "baseline_episodes": 2,
            "fixed_timer_phase_duration": 30,
        },
        "paths": {
            "model_dir": "models/saved/",
            "log_dir": "logs/",
            "results_plot": "docs/results.png",
            "training_summary": "docs/training_summary.json",
            "tensorboard_dir": "logs/test",
        },
    }
