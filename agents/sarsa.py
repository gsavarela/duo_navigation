'''SARSA: On-policy TD(0) policy control.

    References:
    -----------
    * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 133).
''' 
import dill
from pathlib import Path
import numpy as np
from numpy.random import rand, choice

from decorators import int2act, act2int
from utils import i2q, q2i

class SARSATabular(object):
    '''SARSA: On policy TD(0) policy control.

        References:
        -----------
        * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 133).
    ''' 
    def __init__(self, env, alpha=0.2, gamma=0.98, max_episodes=1000):

        # Constants
        self.n_agents = len(env.agents)
        self.n_states = env.state.n_states
        self.n_joint_actions = env.team_actions.n_team_actions

        # Parameters
        self.Q = np.zeros((self.n_states, self.n_joint_actions))

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.gamma = 0.98
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / 1000
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)
        if self.step_count > 0: self.epsilon = max(self.epsilon - self.epsilon_step, 1e-2)

    @int2act
    def act(self, state):
        if rand() < self.epsilon:
            ind = choice(self.n_joint_actions)
        else:
            ind = np.argmax(self.Q[state, :])
        return ind

    @act2int
    def update(self, state, actions, next_rewards, next_state, next_actions):
        self.Q[state, actions] += self.alpha * (next_rewards + self.gamma * \
                self.Q[next_state, next_actions] - self.Q[state, actions])
        self.epsilon = max(1e-2, self.epsilon - 1e-4)
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
    '''SARSA Semi-gradient: On policy TD(0) policy control.

        * Linear function approximation
        
        References:
        -----------
        * Sutton and Barto `Introduction to Reinforcement Learning 2nd Edition` (pg 251).
    ''' 
    def __init__(self, env, alpha=0.2, beta=0.8, max_episodes=1000):

        # The environment
        self.env = env
        self.phi = env.features.get_phi

        # Constants
        self.n_phi = env.features.n_phi[-1]
        self.n_agents = len(env.agents)
        self.n_states = env.state.n_states
        self.n_joint_actions = env.team_actions.n_team_actions

        # Parameters
        self.omega = np.zeros(self.n_phi)
        self.mu = 0

        # Loop control
        self.step_count = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1
        self.epsilon_step = (1 - 1e-2) / 1000
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)

    @int2act
    def act(self, state):
        if rand() < self.epsilon:
            ind = choice(self.n_joint_actions)
        else:
            q_values = [self.Q(state, i2q(a, self.n_agents)) for a in range(self.n_joint_actions)]
            ind = np.argmax(q_values)
        return ind

    def update(self, state, actions, next_rewards, next_state, next_actions):
        delta = np.mean(next_rewards) - self.mu  + \
                self.Q(next_state, next_actions) - self.Q(state, actions)
        self.mu += self.beta * delta
        self.omega += self.alpha * delta * self.phi(state, actions)
        self.epsilon = max(1e-2, self.epsilon - 1e-4)
        self.step_count += 1
        
    def Q(self, state, actions):
        res = self.phi(state, actions) @ self.omega
        return res

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
