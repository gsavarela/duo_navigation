'''One-step ActorCritic For Continuing tasks. 

    * Continuing tasks
    * Q function approximation.
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

class ActorCritic(object):
    def __init__(self, env, alpha=0.3, beta=0.2, zeta=0.1,episodes=20):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        assert self.n_agents < 3
        n_features =  self.n_states // self.n_agents
        self.n_joint_actions = len(env.action_set)

        # Parameters
        # The feature are state-value function features, i.e,
        # the generalize w.r.t the actions.
        self.omega = np.zeros((len(self.action_set), n_features))
        self.theta = np.zeros((len(self.action_set), n_features))
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / (episodes * env.max_steps)
        self.reset(seed=0)

    @property
    def label(self):
        return f'ActorCritic ({label()})'

    @property
    def task(self):
        return 'continuing'

    @property
    def V(self):
        return self._cache_V(self.step_count)
    
    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        return np.array([
            self.Q[state] @ self._cache_PIS(state, self.step_count) for state in range(self.n_states)
        ])

    @property
    def Q(self):
        return self._cache_Q(self.step_count)
    
    @lru_cache(maxsize=1)
    def _cache_Q(self, step_count):
        q_values = []
        for state in range(self.n_states):
            qs = self._cache_QS(state, step_count)
            q_values.append(qs)
        return np.array(q_values)

    @lru_cache(maxsize=1)
    def _cache_QS(self, state, step_count):
        # TODO: Test straight forward multiplication.
        qs = []
        for ind in range(len(self.action_set)):
            qs.append((get(state) @ self.omega[ind][:]).tolist())
        return qs

    def PI(self, state):
        return self._cache_PIS(state, self.step_count).tolist() 

    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        return softmax(get(state) @ self.theta.T)

    def reset(self, seed=0):
        np.random.seed(seed)
        self.mu = 0


    def act(self, state):
        if rand() < self.epsilon:
            cur = choice(len(self.action_set))
        else:
            cur = choice(len(self.action_set), p=self.PI(state))
        return self.action_set[cur]

    def update(self, state, actions, next_rewards, next_state, next_actions):
        cur = self.action_set.index(actions)
        nxt = self.action_set.index(next_actions)

        delta = np.mean(next_rewards) - self.mu  + \
                ((get(next_state) @ self.omega[nxt]) - get(state) @ self.omega[cur])

        self.mu += self.beta * delta
        self.omega[cur] += self.alpha * delta * get(state)
        self.theta[cur] += self.zeta * self.A(state, cur) * self.psi(state, cur)
        self.epsilon = max(1e-2, self.epsilon - self.epsilon_step)
        self.step_count += 1
        
    def psi(self, state, action):
        p_action = self._cache_PIS(state, self.step_count)[action]
        return (1 - p_action) * get(state)

    def A(self, state, action):
        return get(state) @ (self.omega[action] - self.PI(state) @ self.omega)

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
