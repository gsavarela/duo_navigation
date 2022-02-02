from pathlib import Path
import dill

import numpy as np
from numpy.random import choice

from utils import best_actions, i2q

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(keepdims=True)


class CentralizedActorCritic:
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):
        # Inputs.
        self.env = env
        self.phi = env.features.get_phi
        self.varphi = env.features.get_varphi

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

    @property
    def V(self):
        values = []
        for state in range(self.env.state.n_states):
            varphi = self.varphi(state)
            val = 0
            for actions in self.env.action_set:
                phi = self.phi(state, actions)
                prob = np.prod([self.pi(varphi, i)[act] for i, act in enumerate(actions)])
                val += prob * (phi @ self.omega)
            values.append(val)
        return np.array(values)
        
    def reset(self, seed=0):
        np.random.seed(seed)

    def act(self, state):
        probs = [self.pi(self.varphi(state), i) for i in range(self.n_agents)]
        return [choice(self.n_actions, p=prob) for prob in probs]

    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Gather features from environment.
        # phi = self.env.features.get_phi(state, actions)
        # next_phi = self.env.features.get_phi(next_state, next_actions)
        # varphi = self.env.features.get_varphi(state)

        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.phi(next_state, next_actions) - self.phi(state, actions)) @ self.omega 

        # Critic step.
        self.omega += self.alpha * delta * self.phi(state, actions)
        for i in range(self.n_agents):
            self.theta[i, :] += self.beta * delta * self.psi(actions, self.varphi(state), i)

        # Update loop control.
        self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(next_rewards)
        self.step_count += 1
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
