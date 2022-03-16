"""One-step ActorCritic For Episodic Tasks.

    * JALs: Joint Action Learners
    * Episodic tasks
    * Fully cooperative learners with full state observability
    * Each agent estimates the value of its' own actions.
    * Linear function approximation

    References:
    -----------
    * Sutton and Barto, 2018:
        `Introduction to Reinforcement Learning 2nd Edition` (pg 333).
    * Zhang, et al. 2018:
        `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`
"""
from functools import lru_cache
from cached_property import cached_property
from typing import List, Tuple

import numpy as np
from numpy.random import choice

from features import get, label
from utils import softmax

from agents.common import Serializable
from agents.interfaces import AgentInterface


class ActorCriticJALs(Serializable, AgentInterface):
    def __init__(
        self,
        env,
        alpha=0.3,
        beta=0.2,
        gamma=0.98,
        episodes=20,
        explore=False,
        decay=False,
    ):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.gamma = gamma
        self.explore = explore
        self.decay = decay

        assert self.n_agents < 3
        self.n_features = self.n_states

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        # LFA
        self.omega = np.zeros((self.n_agents, self.n_features))
        self.theta = np.zeros((self.n_agents, self.n_actions, self.n_features))

        # Loop control
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.decay_count = 1
        self.epsilon = 1.0
        self.epsilon_step = float(1.2 * (1 - 1e-1) / episodes)
        self.reset(seed=0, first=True)

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

    @cached_property
    def n_actions(self):
        """number of actions per agent"""
        return int(len(self.action_set) ** (1 / self.n_agents))

    @cached_property
    def label(self):
        return f"ActorCriticJALs ({label()})"

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
            x_s = get(state) / self.tau
            if self.n_agents == 1:
                a_s = self.theta[0] @ x_s  # yields 4
            else:  # n_agent == 2
                a_s = []  # yields 16
                for u in range(self.n_actions):
                    for v in range(self.n_actions):
                        a_s.append(
                            self.theta[0, v, :] @ x_s + self.theta[1, u, :] @ x_s
                        )
            ret.append(a_s)
        return np.stack(ret)

    # Do not use this property within this class
    @property
    def V(self) -> np.ndarray:
        return self._V(self.step_count)

    @property
    def PI(self) -> List[List[float]]:
        ret = []
        for state in range(self.n_states):
            ret.append(self._PI(state))
        return ret

    def act(self, state: int) -> Tuple[int]:
        # Tuple
        ret = []
        for i in range(self.n_agents):
            prob = self._pi(state, i, self.step_count)
            ret.append(choice(self.n_actions, p=prob))
        return tuple(ret)

    def update(
        self,
        state: int,
        actions: Tuple[int],
        next_rewards: np.ndarray,
        next_state: int,
        next_actions: Tuple[int],
        done: bool,
    ) -> None:
        # get arguments
        self.delta = next_rewards.tolist()

        x = get(state)
        y = get(next_state)

        # performs update loop
        for i in range(self.n_agents):
            if done:
                self.delta[i] -= self.omega[i] @ x
            else:
                self.delta[i] += self.omega[i] @ ((self.gamma * y) - x)

            # Actor update
            self.omega[i] += self.alpha * self.delta[i] * get(state)

            # Critic update
            self.theta[i] += (
                self.beta
                * self.discount
                * self.delta[i]
                * self.psi(state, actions[i], i)
            )
        self.discount *= self.gamma
        self.step_count += 1

    def psi(self, state, action, i):
        x = get(state) / self.tau
        X = np.tile(x, (self.n_actions, 1))
        P = -np.tile(self._pi(state, i, self.step_count), (self.theta[i].shape[0], 1)).T
        P[action] += 1
        return P @ X

    @lru_cache(maxsize=1)
    def _V(self, step_count: int) -> np.ndarray:
        ret = []
        for state in range(self.n_states):
            vs = np.mean(self.omega @ get(state), axis=0)
            ret.append(vs)
        return np.array(ret)

    def _PI(self, state: int) -> List[float]:
        if self.n_agents == 1:
            self._pi(state, 0, self.step_count)
        # For two agents
        ret = []
        for p in self._pi(state, 1, self.step_count):
            for q in self._pi(state, 0, self.step_count):
                ret.append(q * p)
        return ret

    @lru_cache(maxsize=1)
    def _pi(self, state: int, i: int, step_count: int) -> np.ndarray:
        x = get(state) / self.tau
        return softmax(self.theta[i] @ x)
