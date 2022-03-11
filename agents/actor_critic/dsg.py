'''One-step ActorCritic For Continuing tasks. 

    * Continuing tasks
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
from numpy.random import choice

from features import get, get_sa, label
from utils import softmax

class ActorCriticDifferentialSemiGradient(object):
    def __init__(self, env, alpha=0.3, beta=0.2, zeta=0.1,episodes=20, explore=False, decay=False):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        assert self.n_agents < 3
        n_features =  self.n_states
        self.n_joint_actions = len(env.action_set)

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros(n_features)
        self.theta = np.zeros(n_features)
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.zeta = zeta
        self.explore = explore
        self.decay = decay
        self.epsilon = 1.0
        self.decay_count = 1
        self.epsilon_step = float(1.1  * (1 - 1e-1) / (env.max_steps * episodes))
        self.reset(seed=0)

    @property
    def label(self):
        return f'ActorCritic ({label()})'

    @property
    def task(self):
        return 'continuing'

    @property
    def tau(self):
        return float(10 * self.epsilon if self.explore else 1.0)

    @property
    def A(self):
        res = np.stack([
            (get(state) @ self.theta.T / self.tau) for state in range(self.n_states)
        ])
        return res
        
    @property
    def V(self):
        return self._V(self.step_count)
    
    @lru_cache(maxsize=1)
    def _V(self, step_count):
        return get() @ self.omega


    def PI(self, state):
        return self.pi(state, explore=False).tolist()

    def pi(self, state, explore=False):
        tau = self.tau if explore else 1 
        return softmax(self._thetaX(state, self.step_count) / tau)

    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        return softmax(get(state) @ self.theta.T / self.tau)

    def reset(self, seed=0):
        np.random.seed(seed)

    def act(self, state):
        cur = choice(len(self.action_set), p=self.PI(state))
        return self.action_set[cur]

    def update(self, state, actions, next_rewards, next_state, next_actions):

        cur = self.action_set.index(actions)

        self.delta = np.mean(next_rewards) - self.mu  + \
                    (get(next_state) - get(state)) @ self.omega

        self.delta = np.clip(self.delta, -1, 1)
        self.mu += self.zeta * self.delta
        self.omega += self.alpha * self.delta * get(state)
        self.theta += self.beta * self.delta * self.psi(state, cur)
        self.step_count += 1
        self.epsilon = float(max(1e-1, self.epsilon - self.epsilon_step))


        if self.decay:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count, -0.85)
            self.beta = np.power(self.decay_count, -0.65)

        
    def psi(self, state, action):
        X = np.tile(get(state) / self.tau, (len(self.action_set), 1))
        P = -np.tile(self.PI(state), (self.theta.shape[0], 1)).T
        P[action] += 1
        return P * X

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
