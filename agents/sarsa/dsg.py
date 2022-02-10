'''SARSA Differencial Semi-gradient: On policy TD(0) policy control.

    * Continuing tasks
    * Linear function approximation
    
    References:
    -----------
    * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 251).
''' 
from pathlib import Path
from functools import lru_cache

import dill
import numpy as np
from numpy.random import rand, choice

from features import get, label

class SARSADifferentialSemiGradient(object):
    def __init__(self, env, alpha=0.2, beta=0.8, episodes=20):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        assert self.n_agents < 3
        n_features =  self.n_states // self.n_agents
        self.n_joint_actions = len(env.action_set)

        # Parameters
        # The feature are related to the state.
        self.omega = np.zeros((len(self.action_set), n_features))
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / (episodes * env.max_steps)
        self.reset(seed=0, first=True)

    @property
    def label(self):
        return f'Sarsa DSG ({label()})'

    @property
    def task(self):
        return 'continuing'

    @property
    def V(self):
        return self._cache_V(self.step_count)
    
    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        return np.max(self.Q, axis=1)

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
        qs = []
        for ind in range(len(self.action_set)):
            qs.append((get(state) @ self.omega[ind][:]).tolist())
        return qs

    def PI(self, state):
        # TODO: investigate why keepdims is not an option here.
        # res = np.argmax(self.Q[state, :], axis=1, keepdims=True)
        max_action = int(np.argmax(self._cache_QS(state, self.step_count)))
        res = [int(i == max_action) for i in range(self.n_joint_actions)]
        return res

    def reset(self, seed=0, first=False):
        np.random.seed(seed)
        self.mu = 0


    def act(self, state):
        if rand() < self.epsilon:
            cur = choice(len(self.action_set))
        else:
            qs = self._cache_QS(state, self.step_count)
            cur = np.argmax(qs)
        return self.action_set[cur]

    def update(self, state, actions, next_rewards, next_state, next_actions):
        cur = self.action_set.index(actions)
        nxt = self.action_set.index(next_actions)

        delta = np.mean(next_rewards) - self.mu  + \
                ((get(next_state) @ self.omega[nxt]) - get(state) @ self.omega[cur])

        self.mu += self.beta * delta
        self.omega[cur][:] += self.alpha * delta * get(state)
        self.epsilon = max(1e-2, self.epsilon - self.epsilon_step)
        self.step_count += 1
        
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
