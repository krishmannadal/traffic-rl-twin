"""
base_agent.py — Abstract base class for all RL agents in this project.

Every agent (traffic signal, emergency corridor, etc.) should inherit from
BaseAgent so the training harness, API layer, and evaluation scripts can
treat them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class that defines the interface every agent must implement.

    Subclasses must override:
        • train()     — run the learning loop
        • predict()   — choose an action given an observation
        • evaluate()  — measure agent performance over N episodes
        • save()      — persist model weights / parameters
        • load()      — restore model weights / parameters
    """

    def __init__(self, env: Any, config: Dict[str, Any] | None = None):
        """
        Parameters
        ----------
        env : gymnasium.Env
            The environment the agent will interact with.
        config : dict, optional
            A dictionary of hyperparameters / settings.
        """
        self.env = env
        self.config = config or {}

    # ------------------------------------------------------------------
    # Core interface — every concrete agent MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, timesteps: int) -> None:
        """Train the agent for *timesteps* environment steps."""
        ...

    @abstractmethod
    def predict(self, observation: np.ndarray) -> Tuple[Any, Any]:
        """
        Given an observation, return (action, extra_info).

        Parameters
        ----------
        observation : np.ndarray
            Current state/observation from the environment.

        Returns
        -------
        action : int or np.ndarray
            The action chosen by the agent.
        info : dict or None
            Optional metadata (e.g. Q-values, confidence).
        """
        ...

    @abstractmethod
    def evaluate(self, episodes: int) -> Dict[str, float]:
        """
        Roll out the current policy for *episodes* episodes and return
        aggregated metrics (mean reward, mean waiting time, etc.).
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights / parameters to *path*."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights / parameters from *path*."""
        ...
