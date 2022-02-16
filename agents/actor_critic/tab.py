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
# DEBUG ALTERNATIVE 1: Use a new generator
from numpy.random import Generator

# DEBUG ALTERNATIVE 2: Select actions from POLICY_SET.
from agents.optimal import POLICY_SET

from features import get, label
from utils import softmax

class ActorCriticTabular(object):
    def __init__(self, env, alpha=0.3, beta=0.2, zeta=0.1,episodes=20):

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
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.reset(seed=0)
        self.explore = False
        self.epsilon = 1.0
        self.epsilon_step = float(2 * (1 - 1e-2) / env.max_steps * episodes)

    @property
    def label(self):
        return f'ActorCriticTabular ({label()})'

    @property
    def task(self):
        return 'continuing'

    def PI(self, state):
        return self._cache_PIS(state, self.step_count).tolist() 

    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        return softmax(get(state) @ self.theta.T)

    def reset(self, seed=0):
        np.random.seed(seed)


    def act(self, state):
        if self.explore and rand() < self.epsilon:
            cur = choice(len(self.action_set), p=self.PI(state))
        cur = choice(len(self.action_set), p=self.PI(state))
        return self.action_set[cur]

    def update(self, state, actions, next_rewards, next_state, next_actions):
        cur = self.action_set.index(actions)

        self.delta = np.mean(next_rewards) - self.mu  + \
                    self.V[next_state] - self.V[state]

        self.delta = np.clip(self.delta, -1, 1)
        self.mu += self.beta * self.delta
        self.V[state] += self.alpha * self.delta
        self.theta[cur] += self.zeta * self.delta * self.psi(state, cur)
        self.step_count += 1
        self.epsilon = float(max(0, self.epsilon - self.epsilon_step))

        
    def psi(self, state, action):
        return (1 - self.PI(state)[action]) * get(state)


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
