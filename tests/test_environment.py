"""
test_environment.py — Unit tests for the simulation environments.

Tests observation/action space shapes, step return structure, reward ranges,
and episode lifecycle using mock SUMO to avoid requiring SUMO installation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────
#  Observation & Action Space Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrafficEnvSpaces:
    """Tests for TrafficEnv observation and action space shapes."""

    def test_observation_space_shape(self, mock_traffic_env):
        """Observation should be a 17-dimensional float vector."""
        assert mock_traffic_env.observation_space.shape == (17,)

    def test_action_space_size(self, mock_traffic_env):
        """Action space should have 4 discrete actions (4 signal phases)."""
        assert mock_traffic_env.action_space.n == 4

    def test_observation_dtype(self, mock_traffic_env):
        """Observations should be float32 for neural network compatibility."""
        obs, _ = mock_traffic_env.reset()
        assert obs.dtype == np.float32

    def test_observation_values_in_range(self, mock_traffic_env):
        """Observations should be normalised to [0, 1]."""
        obs, _ = mock_traffic_env.reset()
        assert np.all(obs >= 0.0), f"Observation has negative values: {obs}"
        assert np.all(obs <= 1.0), f"Observation exceeds 1.0: {obs}"

    def test_action_space_sample(self, mock_traffic_env):
        """Sampled action should be a valid integer in [0, n_actions)."""
        for _ in range(20):
            action = mock_traffic_env.action_space.sample()
            assert 0 <= action < 4, f"Invalid action: {action}"


# ──────────────────────────────────────────────────────────────────────
#  Reset Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrafficEnvReset:
    """Tests for the reset() method."""

    def test_reset_returns_tuple(self, mock_traffic_env):
        """reset() should return (observation, info)."""
        result = mock_traffic_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self, mock_traffic_env):
        """Observation from reset should match observation_space."""
        obs, _ = mock_traffic_env.reset()
        assert obs.shape == mock_traffic_env.observation_space.shape

    def test_reset_info_contains_keys(self, mock_traffic_env):
        """Info dict from reset should contain expected keys."""
        _, info = mock_traffic_env.reset()
        assert "sim_step" in info
        assert "total_waiting_time" in info


# ──────────────────────────────────────────────────────────────────────
#  Step Tests
# ──────────────────────────────────────────────────────────────────────

class TestTrafficEnvStep:
    """Tests for the step() method."""

    def test_step_returns_five_values(self, mock_traffic_env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        mock_traffic_env.reset()
        result = mock_traffic_env.step(0)
        assert len(result) == 5

    def test_step_observation_shape(self, mock_traffic_env):
        """Observation from step should match observation_space."""
        mock_traffic_env.reset()
        obs, _, _, _, _ = mock_traffic_env.step(0)
        assert obs.shape == (17,)

    def test_step_reward_is_float(self, mock_traffic_env):
        """Reward should be a numeric value."""
        mock_traffic_env.reset()
        _, reward, _, _, _ = mock_traffic_env.step(0)
        assert isinstance(reward, (int, float, np.floating))

    def test_step_reward_bounded(self, mock_traffic_env):
        """Reward should be in a reasonable range [-2, 2]."""
        mock_traffic_env.reset()
        for _ in range(10):
            _, reward, _, _, _ = mock_traffic_env.step(0)
            assert -2.0 <= reward <= 2.0, f"Reward out of range: {reward}"

    def test_step_terminated_is_bool(self, mock_traffic_env):
        """terminated flag should be boolean."""
        mock_traffic_env.reset()
        _, _, terminated, _, _ = mock_traffic_env.step(0)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, mock_traffic_env):
        """truncated flag should be boolean."""
        mock_traffic_env.reset()
        _, _, _, truncated, _ = mock_traffic_env.step(0)
        assert isinstance(truncated, bool)

    def test_step_info_has_waiting_time(self, mock_traffic_env):
        """Info dict from step should include total_waiting_time."""
        mock_traffic_env.reset()
        _, _, _, _, info = mock_traffic_env.step(0)
        assert "total_waiting_time" in info
        assert info["total_waiting_time"] >= 0

    def test_episode_truncation(self, mock_traffic_env):
        """Episode should truncate after max_steps."""
        mock_traffic_env.reset()
        for i in range(mock_traffic_env._max_steps + 5):
            _, _, terminated, truncated, _ = mock_traffic_env.step(0)
            if terminated or truncated:
                assert i < mock_traffic_env._max_steps + 1
                break
        else:
            pytest.fail("Episode never truncated")


# ──────────────────────────────────────────────────────────────────────
#  Emergency Environment Tests
# ──────────────────────────────────────────────────────────────────────

class TestEmergencyEnv:
    """Tests for the EmergencyEnv emergency-specific methods."""

    def test_trigger_emergency_returns_baseline(self, mock_emergency_env):
        """trigger_emergency() should return estimated travel time."""
        baseline = mock_emergency_env.trigger_emergency(
            vehicle_id="emergency_001",
            origin_edge="south_to_center",
            destination_edge="center_to_north",
        )
        assert isinstance(baseline, float)
        assert baseline > 0

    def test_trigger_emergency_activates(self, mock_emergency_env):
        """trigger_emergency() should set emergency_active flag."""
        mock_emergency_env.trigger_emergency(
            "emergency_002", "south_to_center", "center_to_north"
        )
        assert mock_emergency_env._emergency_active is True

    def test_deactivate_emergency(self, mock_emergency_env):
        """deactivate_emergency() should clear the emergency flag."""
        mock_emergency_env.trigger_emergency(
            "emergency_003", "south_to_center", "center_to_north"
        )
        mock_emergency_env.deactivate_emergency()
        assert mock_emergency_env._emergency_active is False

    def test_get_green_corridor(self, mock_emergency_env):
        """get_green_corridor() should return a list of preemption entries."""
        corridor = mock_emergency_env.get_green_corridor(
            mock_emergency_env.emergency_route
        )
        assert isinstance(corridor, list)
        assert len(corridor) > 0

    def test_emergency_info_flag(self, mock_emergency_env):
        """Step info should reflect emergency_active state."""
        mock_emergency_env.reset()
        mock_emergency_env.trigger_emergency(
            "emergency_004", "south_to_center", "center_to_north"
        )
        _, _, _, _, info = mock_emergency_env.step(0)
        assert info["emergency_active"] is True


# ──────────────────────────────────────────────────────────────────────
#  Close/Cleanup Tests
# ──────────────────────────────────────────────────────────────────────

class TestEnvironmentCleanup:
    """Ensure environments clean up resources properly."""

    def test_close_does_not_raise(self, mock_traffic_env):
        """close() should not raise an exception."""
        mock_traffic_env.reset()
        mock_traffic_env.step(0)
        mock_traffic_env.close()  # Should not raise

    def test_double_close(self, mock_traffic_env):
        """Calling close() twice should be safe (idempotent)."""
        mock_traffic_env.close()
        mock_traffic_env.close()  # Should not raise
