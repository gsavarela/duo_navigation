"""One-step ActorCritic For Episodic Tasks.

    * Episodic tasks.
    * V function approximation.
    * Linear function approximation.
    * Semi-gradient actor critic.

    References:
    -----------
    * Sutton and Barto, 2018:
        `Introduction to Reinforcement Learning 2nd Edition` (pg 333).
    * Zhang, et al. 2018:
        `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`
"""
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from numpy.random import choice

from features import get, label
from utils import softmax

from agents.common import Serializable
from agents.interfaces import AgentInterface

N_PLAYERS = 2

class ActorCriticCentral(Serializable, AgentInterface):
    def __init__(
        self,
        env,
        alpha=0.3,
        beta=0.2,
        gamma=0.98,
        episodes=20,
        explore=False,
        decay=True,
    ):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_states = env.n_states
        assert N_PLAYERS == 2
        n_features = self.n_states

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        # LFA
        self.omega = np.zeros(n_features)
        self.theta = np.zeros((len(self.action_set), n_features))

        # Loop control
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.decay = decay
        self.decay_count = 1
        self.gamma = gamma
        self.explore = explore
        self.epsilon = 1.0
        self.epsilon_step = float(1.2 * (1 - 1e-1) / episodes)
        self.reset(seed=0, first=True)

    # TODO: those methods should go to the interface.
    def reset(self, seed=None, first=False):
        self.discount = 1.0

        np.random.seed(seed)
        if not first:
            self.epsilon = max(1e-1, self.epsilon - self.epsilon_step)
            # For each episode
            if self.decay:
                self.decay_count += 1
                self.alpha = np.power(self.decay_count, -0.85)
                self.beta = np.power(self.decay_count, -0.65)

    @property
    def label(self):
        return f"ActorCritic SG ({label()})"

    @property
    def task(self):
        return "episodic"

    @property
    def tau(self):
        return float(10 * self.epsilon if self.explore else 1.0)

    """
        AgentInterface: Implementation Methods and Properties.
    """
    @property
    def A(self) -> np.ndarray:
        ret = []
        for state in range(self.n_states):
            ret.append(get(state) @ self.theta.T / self.tau)
        return np.vstack(ret)

    @property
    def PI(self) -> List[List[float]]:
        ret = []
        for state in range(self.n_states):
            ret.append(self._PI(state).tolist())
        return ret

    @property
    def V(self) -> np.ndarray:
        return self._V(self.step_count)

    def act(self, state: int) -> Tuple[int]:
        cur = choice(len(self.action_set), p=self._PI(state))
        return self.action_set[cur]

    def update(
        self,
        state: int,
        actions: Tuple[int],
        next_rewards: np.ndarray,
        next_state: int,
        next_actions: Tuple[int],
        done: bool,
    ) -> None:
        cur = self.action_set.index(actions)

        if done:
            self.delta = np.mean(next_rewards) - (get(state) @ self.omega)

        else:
            self.delta = (
                np.mean(next_rewards)
                + ((self.gamma * get(next_state)) - get(state)) @ self.omega
            )

        # Actor update
        self.omega += self.alpha * self.delta * get(state)

        # Critic update
        self.theta += self.beta * self.discount * self.delta * self.psi(state, cur)
        self.discount *= self.gamma
        self.step_count += 1

    def psi(self, state, action):
        X = np.tile(get(state) / self.tau, (len(self.action_set), 1))
        P = -np.tile(self._PI(state), (self.theta.shape[0], 1)).T

        P[action] += 1
        return P * X

    @lru_cache(maxsize=1)
    def _V(self, step_count: int) -> np.ndarray:
        return np.array([get(state) @ self.omega for state in range(self.n_states)])

    def _PI(self, state: int) -> np.ndarray:
        return self._pi(state, self.step_count)

    @lru_cache(maxsize=1)
    def _pi(self, state: int, step_count: int) -> np.ndarray:
        return softmax(self.theta @ (get(state) / self.tau))
