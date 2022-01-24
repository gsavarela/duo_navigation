import numpy as np
from numpy.random import choice

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(keepdims=True)

class ActorCritic:
    def __init__(self, env):
        # Inputs
        self.env = env

        # Constants
        self.n_agents = len(env.agents)
        self.n_actions = 4
        self.n_phi = 10
        self.n_varphi = 5

        # Parameters
        self.omega = np.ones((self.n_phi,)) * 0.1 
        self.theta = np.ones((self.n_agents, self.n_varphi)) * 0.2

        # Loop control
        self.mu = 0 
        self.step_count = 0
        self.alpha = 1
        self.beta = 1
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
        # Gather features from environment
        phi = self.env.features.get_phi(state, actions)
        next_phi = self.env.features.get_phi(next_state, next_actions)
        varphi = self.env.features.get_varphi(state)
        next_varphi = self.env.features.get_varphi(next_state)

        delta = np.mean(next_rewards) - self.mu + \
                (next_phi - phi) @ self.omega 

        # Critic step.
        self.omega += self.alpha * delta * phi

        # Actor step.
        for i in range(self.n_agents):
            self.theta[i, :] += self.beta * delta * self.psi(actions, varphi, i)

        # Update loop control
        self.step_count += 1
        self.mu = self.next_mu
        self.alpha = np.power(self.step_count + 1, -0.85)
        self.beta = np.power(self.step_count + 1, -0.65)

    def pi(self, varphi, i):
        return softmax(varphi[:, i, :] @ self.theta[i, :])

    def psi(self, actions, varphi, i):
        return varphi[actions[i], i, :] - self.pi(varphi, i) @ varphi[:, i, :]
