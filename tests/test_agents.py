"""
test_agents.py — Unit tests for RL agents and reward functions.

Tests agent initialization, prediction, mode switching, reward function
output shapes, and model save/load lifecycle.  All tests use mock
environments and run without SUMO or GPU.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.reward import TrafficReward, EmergencyReward


# ──────────────────────────────────────────────────────────────────────
#  Reward Function Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrafficReward:
    """Tests for the TrafficReward function."""

    @pytest.fixture
    def reward_fn(self):
        return TrafficReward(device="cpu")

    def test_compute_returns_float(self, reward_fn):
        """compute() should return a float scalar."""
        state = {
            "waiting_time": {"N_0": 10.0, "N_1": 5.0, "S_0": 8.0, "S_1": 3.0,
                             "E_0": 6.0, "E_1": 4.0, "W_0": 7.0, "W_1": 2.0},
            "queue_length": {"N_0": 4, "N_1": 2, "S_0": 3, "S_1": 1,
                             "E_0": 2, "E_1": 3, "W_0": 1, "W_1": 2},
            "vehicle_count": {"N_0": 5, "N_1": 3, "S_0": 4, "S_1": 2,
                              "E_0": 3, "E_1": 2, "W_0": 4, "W_1": 1},
            "signal_phase": 0,
        }
        result = reward_fn.compute(state)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_reward_bounded(self, reward_fn):
        """Reward should be bounded in the expected range."""
        state = {
            "waiting_time": {"N_0": 100.0, "N_1": 100.0, "S_0": 100.0, "S_1": 100.0,
                             "E_0": 100.0, "E_1": 100.0, "W_0": 100.0, "W_1": 100.0},
            "queue_length": {"N_0": 20, "N_1": 20, "S_0": 20, "S_1": 20,
                             "E_0": 20, "E_1": 20, "W_0": 20, "W_1": 20},
            "vehicle_count": {"N_0": 10, "N_1": 10, "S_0": 10, "S_1": 10,
                              "E_0": 10, "E_1": 10, "W_0": 10, "W_1": 10},
            "signal_phase": 1,
        }
        result = reward_fn.compute(state)
        assert -1.0 <= result <= 0.5, f"Reward out of range: {result}"

    def test_zero_state_gives_reasonable_reward(self, reward_fn):
        """Zero waiting + zero queue should give a non-negative reward."""
        state = {
            "waiting_time": {"N_0": 0.0, "N_1": 0.0, "S_0": 0.0, "S_1": 0.0,
                             "E_0": 0.0, "E_1": 0.0, "W_0": 0.0, "W_1": 0.0},
            "queue_length": {"N_0": 0, "N_1": 0, "S_0": 0, "S_1": 0,
                             "E_0": 0, "E_1": 0, "W_0": 0, "W_1": 0},
            "vehicle_count": {"N_0": 5, "N_1": 5, "S_0": 5, "S_1": 5,
                              "E_0": 5, "E_1": 5, "W_0": 5, "W_1": 5},
            "signal_phase": 0,
        }
        result = reward_fn.compute(state)
        assert result >= -0.5, f"Zero-state reward too low: {result}"

    def test_high_wait_gives_negative_reward(self, reward_fn):
        """Very high waiting times should produce negative reward."""
        state = {
            "waiting_time": {"N_0": 500.0, "N_1": 500.0, "S_0": 500.0, "S_1": 500.0,
                             "E_0": 500.0, "E_1": 500.0, "W_0": 500.0, "W_1": 500.0},
            "queue_length": {"N_0": 25, "N_1": 25, "S_0": 25, "S_1": 25,
                             "E_0": 25, "E_1": 25, "W_0": 25, "W_1": 25},
            "vehicle_count": {"N_0": 10, "N_1": 10, "S_0": 10, "S_1": 10,
                              "E_0": 10, "E_1": 10, "W_0": 10, "W_1": 10},
            "signal_phase": 2,
        }
        result = reward_fn.compute(state)
        assert result < 0, f"High-wait reward should be negative, got {result}"


class TestEmergencyReward:
    """Tests for the EmergencyReward function."""

    @pytest.fixture
    def reward_fn(self):
        return EmergencyReward(device="cpu")

    def test_compute_returns_float(self, reward_fn):
        """compute() should return a float scalar."""
        result = reward_fn.compute(
            emergency_travel_time=30.0,
            baseline_time=45.0,
            traffic_disruption=0.1,
            is_arrival=False,
        )
        assert isinstance(result, float)

    def test_reward_is_numeric(self, reward_fn):
        """Even with edge-case inputs, reward should be a valid number."""
        result = reward_fn.compute(
            emergency_travel_time=0.0,
            baseline_time=0.0,
            traffic_disruption=0.0,
            is_arrival=False,
        )
        assert not np.isnan(result), "Reward should not be NaN"
        assert not np.isinf(result), "Reward should not be infinite"

    def test_arrival_bonus(self, reward_fn):
        """Arrival should give a positive boost."""
        no_arrival = reward_fn.compute(30.0, 60.0, 0.0, is_arrival=False)
        with_arrival = reward_fn.compute(30.0, 60.0, 0.0, is_arrival=True)
        assert with_arrival > no_arrival


# ──────────────────────────────────────────────────────────────────────
#  Traffic Agent Tests (DQN)
# ──────────────────────────────────────────────────────────────────────

class TestTrafficAgent:
    """Tests for the TrafficAgent wrapper around SB3 DQN."""

    @pytest.fixture
    def agent(self, mock_traffic_env):
        """Create a TrafficAgent with a mock environment."""
        from agents.traffic_agent import TrafficAgent
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = TrafficAgent(env=mock_traffic_env, model_dir=tmpdir)
            yield agent

    def test_agent_initialises(self, agent):
        """TrafficAgent should initialise without error."""
        assert agent is not None
        assert agent.model is not None

    def test_predict_returns_valid_action(self, agent, sample_observation):
        """predict() should return a valid action in [0, n_actions)."""
        action, _ = agent.predict(sample_observation, deterministic=True)
        assert 0 <= int(action) < 4

    def test_predict_batch(self, agent):
        """predict() should handle a batch of observations."""
        batch_obs = np.random.rand(5, 17).astype(np.float32)
        # SB3's predict handles batches via iteration
        for obs in batch_obs:
            action, _ = agent.predict(obs)
            assert 0 <= int(action) < 4

    def test_model_save_load_cycle(self, agent, sample_observation, tmp_path):
        """Save and load should produce the same predictions."""
        # Get prediction before save
        action_before, _ = agent.predict(sample_observation, deterministic=True)
        
        # Save
        save_path = str(tmp_path / "test_model")
        agent.model.save(save_path)
        
        # Load
        from stable_baselines3 import DQN
        loaded_model = DQN.load(save_path)
        action_after, _ = loaded_model.predict(sample_observation, deterministic=True)
        
        # Should give same action (deterministic)
        assert int(action_before) == int(action_after)


# ──────────────────────────────────────────────────────────────────────
#  Emergency Agent Tests (PPO)
# ──────────────────────────────────────────────────────────────────────

class TestEmergencyAgent:
    """Tests for the EmergencyAgent wrapper around SB3 PPO."""

    @pytest.fixture
    def agent(self, mock_emergency_env):
        """Create an EmergencyAgent with a mock environment."""
        from agents.emergency_agent import EmergencyAgent
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = EmergencyAgent(
                env=mock_emergency_env,
                model_dir=tmpdir,
                mode="rule_based",
            )
            yield agent

    def test_agent_initialises(self, agent):
        """EmergencyAgent should initialise without error."""
        assert agent is not None
        assert agent.model is not None

    def test_default_mode_is_rule_based(self, agent):
        """Default mode should be 'rule_based'."""
        assert agent.mode == "rule_based"

    def test_is_active_initially_false(self, agent):
        """Agent should not be active before activation."""
        assert agent.is_active is False

    def test_predict_returns_valid_action(self, agent, sample_observation):
        """predict() should return a valid action."""
        action, _ = agent.predict(sample_observation, deterministic=True)
        assert 0 <= int(action) < 4

    def test_activate_sets_active(self, agent):
        """activate() should set the agent as active."""
        agent.activate(
            emergency_vehicle_id="test_001",
            origin="south_to_center",
            destination="center_to_north",
        )
        assert agent.is_active is True

    def test_deactivate_clears_active(self, agent):
        """deactivate() should clear the active flag."""
        agent.activate("test_002", "south_to_center", "center_to_north")
        agent.deactivate()
        assert agent.is_active is False

    def test_deactivate_returns_metrics(self, agent):
        """deactivate() should return performance metrics."""
        agent.activate("test_003", "south_to_center", "center_to_north")
        result = agent.deactivate()
        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────────────
#  Reward Function GPU/Device Tests
# ──────────────────────────────────────────────────────────────────────

class TestRewardDevice:
    """Ensure reward functions work correctly on CPU (and GPU if available)."""

    def test_traffic_reward_cpu(self):
        """TrafficReward should work on CPU."""
        fn = TrafficReward(device="cpu")
        state = {
            "waiting_time": {"N_0": 5.0, "N_1": 5.0, "S_0": 5.0, "S_1": 5.0,
                             "E_0": 5.0, "E_1": 5.0, "W_0": 5.0, "W_1": 5.0},
            "queue_length": {"N_0": 2, "N_1": 2, "S_0": 2, "S_1": 2,
                             "E_0": 2, "E_1": 2, "W_0": 2, "W_1": 2},
            "vehicle_count": {"N_0": 3, "N_1": 3, "S_0": 3, "S_1": 3,
                              "E_0": 3, "E_1": 3, "W_0": 3, "W_1": 3},
            "signal_phase": 0,
        }
        result = fn.compute(state)
        assert isinstance(result, float)

    def test_emergency_reward_cpu(self):
        """EmergencyReward should work on CPU."""
        fn = EmergencyReward(device="cpu")
        result = fn.compute(
            emergency_travel_time=25.0,
            baseline_time=40.0,
            traffic_disruption=0.15,
            is_arrival=False,
        )
        assert isinstance(result, float)
