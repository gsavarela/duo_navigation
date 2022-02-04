'''The optimal agent always moves according to the best action.

    * Performs policy evalution (critic)

'''
import dill
from pathlib import Path
from functools import lru_cache
from itertools import product

import numpy as np
from numpy.random import choice

from utils import best_actions
from decorators import int2act


class Optimal:

    def __init__(self, env, alpha=0.2, decay=True):
        # Inputs
        self.goal_pos = env.goal_pos
        self.action_set = env.action_set

        # TODO: optimize this.
        self.phi = env.features.get_phi # use this for update.
        def PHI(x): # use this for acting.
            return np.array([self.phi(x, u) for u in self.action_set])
        self.PHI = PHI 

        # Constants.
        self.n_agents = len(env.agents)
        self.n_states = env.state.n_states
        self.n_actions = 4
        self.n_phi = 10
        self.decay = decay

        # Parameters.
        self.omega = np.zeros(self.n_phi)

        # Loop control.
        self.mu = 0 
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.decay_count = 0
        self.reset()

        # Build the optimal policy
        self._build_policy(env)


    @property
    def V(self):
        return self._cache_V(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_V(self, t):
        values = []
        for state in range(self.n_states):
            values.append(self.pi(state) @ self.Q[state, :])
        return np.array(values)

    @property
    def Q(self):
        return self._cache_Q(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_Q(self, t):
        q_values = []
        for state in range(self.n_states):
            q_values.append((self.PHI(state) @ self.omega).tolist())
        return np.array(q_values)

    def reset(self, seed=0):
        np.random.seed(seed)

    @int2act
    def act(self, state):
        res = choice(np.arange(len(self.action_set)), p=self.pi(state))
        return res

    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.phi(next_state, next_actions) - self.phi(state, actions)) @ self.omega 

        # Critic step.
        self.omega += self.alpha * delta * self.phi(state, actions)

        # Update loop control.
        self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(next_rewards)
        self.step_count += 1
        if self.decay and np.abs(delta) > 1e-6:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count + 1, -0.85)

    def _build_policy(self, env):
        states_positions_gen = env.next_states()
        def fn(x):
            return best_actions(x, self.goal_pos, env.state.width, env.state.height)

        try:
            res = []
            while True:
                state, positions = next(states_positions_gen)
                bas = [*map(fn, positions)]
                jas = [*product(*bas, repeat=1)]
                res.append([a in jas for a in self.action_set])

        except StopIteration:
            self._pi = np.array(res)
            self._pi = self._pi / self._pi.sum(keepdims=True, axis=1)
        return self._pi

    # Joint action from all agents.
    def pi(self, state):
        return self._pi[state]

    # Joint action from all agents in list format.
    # Interface compatibility between A-C.
    def PI(self, state):
        return self.pi(state).tolist()

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
