from pathlib import Path
from decorators import int2act, act2int

import dill
import numpy as np
from numpy.random import rand, choice

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
