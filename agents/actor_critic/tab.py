'''One-step ActorCritic For Continuing tasks. 

    * Continuing tasks
    * Critic Tabular V function approximation.
    * Actor boltzman polict
    
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

class ActorCriticTabular(object):
    def __init__(self, env, alpha=0.3, beta=0.2, gamma=0.98 ,episodes=20, explore=False, decay=True):

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
        # self.omega = np.zeros(n_features)
        self.V = np.zeros(self.n_states)
        self.theta = np.zeros((len(self.action_set), n_features))
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.gamma = gamma
        self.decay = decay
        self.decay_count = 1
        self.reset(seed=0, first=True)
        self.explore = explore
        self.epsilon = 1.0
        self.epsilon_step = float((1.1 - 1e-2) / (env.max_steps * episodes))

    @property
    def label(self):
        return f'ActorCriticTabular ({label()})'

    @property
    def task(self):
        return 'episodic'

    def PI(self, state):
        return self._cache_PIS(state, self.step_count).tolist() 


    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        return softmax(get(state) @ self.theta.T / self.tau)
    
    def reset(self, seed=0, first=False):
        self.discount = 1.0

        if seed is not None:
            np.random.seed(seed)
        if not first:
            self.epsilon = max(1e-1, self.epsilon - self.epsilon_step)
            # For each episode
            if self.decay:
                self.decay_count += 1
                self.alpha = np.power(self.decay_count, -0.85)
                self.beta = np.power(self.decay_count, -0.65)

    @property
    def A(self):
        return np.stack([
            (get(state) @ self.theta.T / self.tau) for state in range(self.n_states)
        ])

    def act(self, state):
        cur = choice(len(self.action_set), p=self.PI(state))
        return self.action_set[cur]


    @property
    def tau(self):
        return float(10 * self.epsilon if self.explore else 1.0)

    def update(self, state, actions, next_rewards, next_state, next_actions, done):
        cur = self.action_set.index(actions)

        if done:
            self.delta = np.mean(next_rewards) - self.V[state]
                
        else:
            self.delta = np.mean(next_rewards) + \
                    (self.gamma * self.V[next_state] - self.V[state]) 

        self.V[state] += self.alpha * self.delta
        self.theta += self.beta * self.discount * self.delta * self.psi(state, cur)

        self.discount *= self.gamma
        self.step_count += 1
        self.epsilon = float(max(1e-2, self.epsilon - self.epsilon_step))
        
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
