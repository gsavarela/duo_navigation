'''One-step ActorCritic For Episodic Tasks.

    * Episodic tasks
    * V function approximation.
    * Linear function approximation
    
    References:
    -----------
    * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 333).
    * Zhang, et al. 2018 `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`
''' 
from pathlib import Path
from functools import lru_cache

import dill
import numpy as np
from numpy.random import rand, choice

from features import get, label
from utils import softmax

class ActorCriticSemiGradient(object):
    def __init__(self, env, alpha=0.3, beta=0.2, gamma=0.98 ,episodes=20, explore=False):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        assert self.n_agents < 3
        # n_features =  self.n_states // self.n_agents
        n_features =  self.n_states
        self.n_joint_actions = len(env.action_set)

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros(n_features)
        self.theta = np.zeros((len(self.action_set), n_features))

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.explore = explore
        self.epsilon = 1.0
        self.epsilon_step = float(1.2  * (1 - 1e-1) / episodes)
        self.reset(seed=0, first=True)

    def reset(self, seed=0, first=False):
        self.discount = 1.0
        if first:
            np.random.seed(seed)
        else:
            self.epsilon = max(1e-1, self.epsilon - self.epsilon_step)

    @property
    def label(self):
        return f'ActorCritic SG ({label()})'

    @property
    def task(self):
        return 'episodic'

    @property
    def tau(self):
        return float(10 * self.epsilon if self.explore else 1.0)

    @property
    def V(self):
        return self._cache_V(self.step_count)
    
    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        return np.array([
            get(state) @ self.omega for state in range(self.n_states)
        ])

    def PI(self, state):
        return self._cache_PIS(state, self.step_count).tolist() 

    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        return softmax(get(state) @ self.theta.T / self.tau)

    def act(self, state):
        cur = choice(len(self.action_set), p=self.PI(state))
        return self.action_set[cur]

    def update(self, state, actions, next_rewards, next_state, next_actions, done):
        cur = self.action_set.index(actions)

        if done:
            self.delta = np.mean(next_rewards) + (get(state) @ self.omega)
        else:
            self.delta = np.mean(next_rewards) + \
                        ((self.gamma * get(next_state)) - get(state)) @ self.omega

        self.delta = np.clip(self.delta, -1, 1)
        self.omega += self.alpha * self.delta * get(state)
        self.theta[cur][:] += self.beta *  self.discount * self.delta * self.psi(state, cur)
        self.discount *= self.gamma
        self.step_count += 1

        
    def psi(self, state, action):
        return (1 - self.PI(state)[action]) * (get(state) / self.tau)

    def save_checkpoints(self, chkpt_dir_path, chkpt_num):
        class_name = type(self).__name__.lower()
        file_path = Path(chkpt_dir_path) / chkpt_num / f'{class_name}.chkpt'  
        file_path.parent.mkdir(exist_ok=True)
        with open(file_path, mode='wb') as f:
            dill.dump(self, f)
        
    @classmethod
    def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
        class_name = cls.__name__.lower()
        file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'  
        with file_path.open(mode='rb') as f:
            new_instance = dill.load(f)

        return new_instance
