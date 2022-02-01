from pathlib import Path
import dill

import numpy as np
from numpy.random import choice

from utils import best_actions

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(keepdims=True)


class CentralizedActorCritic:
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):
        # Inputs.
        self.env = env

        # Constants.
        self.n_agents = len(env.agents)
        self.n_actions = 4
        self.n_phi = 10
        self.n_varphi = 5
        self.decay = decay

        # Parameters.
        self.omega = np.zeros(self.n_phi)
        self.theta = np.zeros((self.n_agents, self.n_varphi))

        # Loop control.
        self.mu = 0 
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.decay_count = 0
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)

    def act(self, state):
        varphi = self.env.features.get_varphi(state)
        probs = [self.pi(varphi, i) for i in range(self.n_agents)]
        return [choice(self.n_actions, p=prob) for prob in probs]

    def update_mu(self, rewards):
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(rewards)
        return self.next_mu

    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Gather features from environment.
        phi = self.env.features.get_phi(state, actions)
        next_phi = self.env.features.get_phi(next_state, next_actions)
        varphi = self.env.features.get_varphi(state)

        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (next_phi - phi) @ self.omega 

        # Critic step.
        self.omega += self.alpha * delta * phi
        for i in range(self.n_agents):
            self.theta[i, :] += self.beta * delta * self.psi(actions, varphi, i)

        # Update loop control.
        self.step_count += 1
        self.mu = self.next_mu
        if self.decay and np.abs(delta) > 1e-6:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count + 1, -0.85)
            self.beta = np.power(self.decay_count + 1, -0.65)

    def pi(self, varphi, i):
        return softmax(varphi[:, i, :] @ self.theta[i, :])

    def psi(self, actions, varphi, i):
        return varphi[actions[i], i, :] - self.pi(varphi, i) @ varphi[:, i, :]

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
