"""
    Defines properties and methods for agents as expected by source
"""
import abc
import numpy as np
from typing import List, Tuple


class AgentInterface(abc.ABC):
    """Multi agent system interface"""

    @abc.abstractproperty
    def A(self) -> np.ndarray:
        """Advantage."""

    @abc.abstractproperty
    def PI(self) -> List[List[float]]:
        """Advantage."""

    @abc.abstractproperty
    def V(self) -> np.ndarray:
        """Value function"""

    @abc.abstractmethod
    def act(self, state: int) -> Tuple[int]:
        """Act."""

    @abc.abstractmethod
    def update(
        self,
        state: int,
        actions: Tuple[int],
        next_rewards: np.ndarray,
        next_state: int,
        next_action: Tuple[int],
        done: bool,
    ) -> None:
        """Update."""
