'''SARSA: On-policy TD(0) policy control.

    References:
    -----------
    * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 133).
''' 
from pathlib import Path
from functools import lru_cache

import dill
import numpy as np
from numpy.random import rand, choice

from decorators import int2act, act2int
from utils import i2q, q2i
from features import Features

class SARSATabular(object):
    '''SARSA: On policy TD(0) policy control.

        References:
        -----------
        * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 133).
    ''' 
    def __init__(self, env, alpha=0.2, gamma=0.98, episodes=1000):

        # Constants
        self.action_set = env.action_set
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.n_joint_actions = len(env.action_set)

        # Parameters
        self.Q = np.zeros((self.n_states, self.n_joint_actions))

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.gamma = 0.98
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / episodes
        self.reset(first=True)

    # TODO: Move label and task to a model metaclass 
    @property
    def label(self):
        return f'Sarsa Tabular'

    @property
    def task(self):
        return 'episodic'

    @property
    def V(self):
        return np.max(self.Q, axis=1)
    
    def PI(self, state):
        # TODO: investigate why keepdims is not an option here.
        # res = np.argmax(self.Q[state, :], axis=1, keepdims=True)
        max_action = int(np.argmax(self.Q[state, :]))
        res = [int(i == max_action) for i in range(self.n_joint_actions)]
        return res

    def reset(self, seed=0, first=False):
        # np.random.seed(seed)
        if first:
            np.random.seed(seed)
        else:
            self.epsilon = max(1e-2, self.epsilon - self.epsilon_step)
        

    @int2act
    def act(self, state):
        if rand() < self.epsilon:
            ind = choice(self.n_joint_actions)
        else:
            ind = np.argmax(self.Q[state, :])
        return ind

    @act2int
    def update(self, state, actions, next_rewards, next_state, next_actions, *args, **kwargs):
        self.Q[state, actions] += self.alpha * (np.mean(next_rewards) + self.gamma * \
                self.Q[next_state, next_actions] - self.Q[state, actions])
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


class SARSASemiGradient(object):
    '''Episodic Semi-gradient SARSA : On policy TD(0) policy control.

        * Episodic
        * Linear function approximation
        * TODO: Decorate features as an argument.
        
        References:
        -----------
        * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 251).
    ''' 
    def __init__(self, env, alpha=0.2, episodes=20):

        # The environment
        self.action_set = env.action_set
        # self.phi = env.features.get_phi
        self.phi = Features().get
        # def PHI(x): # use this for acting.
        #     return np.array([self.phi(x, u) for u in self.action_set])
        # self.PHI = PHI 

        # Constants
        # self.n_phi = Features().n_phi[-1]
        self.n_agents = len(env.agents)
        self.n_states = env.n_states

        assert self.n_agents < 3
        n_features =  self.n_states // self.n_agents
        self.n_joint_actions = len(env.action_set)

        # Parameters
        # The feature are related to the state.
        self.omega = np.zeros((len(self.action_set), n_features))
        # self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        # self.beta = beta
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / episodes
        self.reset(first=True)

    # TODO: Move label and task to a model metaclass 
    @property
    def label(self):
        return f'Sarsa SG ({Features().label})'

    @property
    def task(self):
        return 'episodic'

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
            qa_values = []
            for ind in range(len(self.action_set)):
                qa_values.append((self.phi(state) @ self.omega[ind][:]).tolist())
            q_values.append(qa_values)
        return np.array(q_values)

    def PI(self, state):
        # TODO: investigate why keepdims is not an option here.
        # res = np.argmax(self.Q[state, :], axis=1, keepdims=True)
        max_action = int(np.argmax(self.Q[state, :]))
        res = [int(i == max_action) for i in range(len(self.action_set))]
        return res

    def reset(self, seed=0, first=False):
        if first:
            np.random.seed(seed)
        else:
            self.epsilon = max(1e-2, self.epsilon - self.epsilon_step)

    def act(self, state):
        if rand() < self.epsilon:
            ind = choice(len(self.action_set))
        else:
            q_values = []
            for ind in range(len(self.action_set)):
                q_values.append(self.phi(state) @ self.omega[ind][:])
            ind = np.argmax(q_values)
        return self.action_set[ind]

    def update(self, state, actions, next_rewards, next_state, next_actions, done):
        ind = self.action_set.index(actions)
        if done:
            delta = np.mean(next_rewards) - self.phi(state) @ self.omega[ind][:]
        else:
            delta = np.mean(next_rewards) + \
                    (self.phi(next_state) -  self.phi(state)) @ self.omega[ind][:]
            
        self.omega[ind][:] += self.alpha * delta * self.phi(state)
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

class SARSADiffentialSemiGradient(object):
    '''SARSA Semi-gradient: On policy TD(0) policy control.

        * Linear function approximation
        
        References:
        -----------
        * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 251).
    ''' 
    def __init__(self, env, alpha=0.2, beta=0.8, episodes=20):

        # The environment
        self.action_set = env.action_set
        # self.phi = env.features.get_phi
        self.phi = Features().get_phi
        def PHI(x): # use this for acting.
            return np.array([self.phi(x, u) for u in self.action_set])
        self.PHI = PHI 

        # Constants
        self.n_phi = Features().n_phi[-1]
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.n_joint_actions = len(env.action_set)

        # Parameters
        self.omega = np.zeros(self.n_phi)
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / (episodes * env.max_steps)
        self.reset(seed=0)

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
            q_values.append((self.phi(state) @ self.omega).tolist())
        return np.array(q_values)

    def PI(self, state):
        # TODO: investigate why keepdims is not an option here.
        # res = np.argmax(self.Q[state, :], axis=1, keepdims=True)
        max_action = int(np.argmax(self.Q[state, :]))
        res = [int(i == max_action) for i in range(self.n_joint_actions)]
        return res

    def reset(self, seed=0):
        np.random.seed(seed)

    def act(self, state):
        if rand() < self.epsilon:
            ind = choice(self.n_joint_actions)
        else:
            q_values = []
            for a in range(self.n_joint_actions):
                q_values.append(self.phi(state, i2q(a, self.n_agents)) @ self.omega)
            ind = np.argmax(q_values)
        return self.action_set[ind]

    def update(self, state, actions, next_rewards, next_state, next_actions):
        delta = np.mean(next_rewards) - self.mu  + \
                (self.phi(next_state, next_actions) -  self.phi(state, actions)) @ self.omega
            
        self.mu += self.beta * delta
        self.omega += self.alpha * delta * self.phi(state, actions)
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
