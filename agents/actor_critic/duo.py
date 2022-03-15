'''One-step ActorCritic For Episodic Tasks.

    * Episodic tasks
    * Fully cooperative learners with full state observability
    * Each agent learns how to act independently.
    * Linear function approximation
    
    References:
    -----------
    * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 333).
    * Zhang, et al. 2018 `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`
''' 
from pathlib import Path
from functools import lru_cache
from cached_property import cached_property

import dill
import numpy as np
from numpy.random import rand, choice

from features import get, label
from utils import softmax

class ActorCriticSemiGradientDuo(object):
    def __init__(self, env, alpha=0.3, beta=0.2, gamma=0.98,
                 episodes=20, explore=False, decay=False,
                 cooperative=True, partial_observability=False):

        # The environment
        self.action_set = env.action_set

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.cooperative = cooperative
        self.partial_observability = partial_observability
        self.gamma = gamma
        self.explore = explore
        self.decay = decay

        assert self.n_agents < 3
        if self.partial_observability:
            self.n_features = ((env.width - 2) * (env.height - 2))
        else:
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
        self.epsilon_step = float(1.2  * (1 - 1e-1) / episodes)
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
        '''number of actions per agent'''
        return int(len(self.action_set) ** (1 / self.n_agents))

    @cached_property
    def label(self):
        prefix = 'coop.' if self.cooperative else 'indep.'
        if self.partial_observability:
            prefix = f'{prefix}+partial_observability'
        return f'ActorCritic Duo ({prefix}, {label()})'

    @property
    def task(self):
        return 'episodic'

    @property
    def tau(self):
        return float(10 * self.epsilon if self.explore else 1.0)

    # Do not use this property within this class
    @property
    def V(self):
        return self._cache_V(self.step_count)
    
    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        if self.partial_observability:
            ret = np.array([
                np.mean(np.sum(self.omega * get(state), axis=1), axis=0)  for state in range(self.n_states)
            ])
        else:
            ret = np.array([
                np.mean(self.omega @ get(state), axis=0)  for state in range(self.n_states)
            ])
        return ret
    def PI(self, state):
        _PI = self._cache_PIS(state, self.step_count)
        return _PI

    def _pi(self, state, i):
        x = (get(state)/ self.tau)
        if self.partial_observability:
            return softmax(self.theta[i] @ x[i])
        else:
            return softmax(self.theta[i] @ x)

    @lru_cache(maxsize=1)
    def _cache_PIS(self, state, step_count):
        if self.n_agents == 1: self._pi(state, 0)
        # For two agents
        ret = []
        for p in self._pi(state, 1):
            for q in self._pi(state, 0):
                ret.append(q * p)
        return ret 

    @property
    def A(self):
        ret = []
        for state in range(self.n_states):
            x_s = (get(state) / self.tau)
            if self.n_agents == 1:
                a_s = self.theta[0] @ x_s # yields 4
            else: # n_agent == 2
                a_s = [] # yields 16
                for u in range(self.n_actions):
                    for v in range(self.n_actions):
                        if self.partial_observability:
                            a_s.append(self.theta[0, v, :] @ x_s[0] + self.theta[1, u, :] @ x_s[1])
                        else:
                            a_s.append(self.theta[0, v, :] @ x_s + self.theta[1, u, :] @ x_s)
            ret.append(a_s)
        return np.stack(ret)

    
    def act(self, state):
        cur = [choice(self.n_actions, p=self._pi(state, i)) for i in range(self.n_agents)]
        return tuple(cur)

    def update(self, state, actions, next_rewards, next_state, next_actions, done):
        # get arguments 
        self.delta = next_rewards.tolist()
    
        x = get(state) 
        y = get(next_state)

        # performs update loop
        for i in range(self.n_agents):
            if self.partial_observability:
                if done:
                    self.delta[i] -= self.omega[i] @ x[i]  
                else:
                    self.delta[i] += self.omega[i] @ ((self.gamma * y[i]) - x[i])
            else:
                if done:
                    self.delta[i] -= self.omega[i] @ x  
                else:
                    self.delta[i] += self.omega[i] @ ((self.gamma * y) - x)

            if self.partial_observability:
                # Actor update
                self.omega[i] += self.alpha * self.delta[i] * x[i]

                # Critic update
                self.theta[i] += self.beta *  self.discount * self.delta[i] * self.psi(state, actions[i], i)
            else:
                # Actor update
                self.omega[i] += self.alpha * self.delta[i] * get(state)

                # Critic update
                self.theta[i] += self.beta *  self.discount * self.delta[i] * self.psi(state, actions[i], i)
        self.discount *= self.gamma
        self.step_count += 1

    def psi(self, state, action, i):
        x = get(state) / self.tau
        if self.partial_observability:
            x = x[i]
        X = np.tile(x, (self.n_actions, 1))
        P = -np.tile(self._pi(state, i), (self.theta[i].shape[0], 1)).T
        P[action] += 1
        return P @ X

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
