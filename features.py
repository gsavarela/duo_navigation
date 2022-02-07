import numpy as np
from numpy.random import uniform
from utils import action_set
# TODO: extend this to accept new parameters,
# features as a singleton pattern

WIDTH = 2
HEIGHT = 2
N_AGENTS = 2
N_ACTIONS = 4
class Features:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Features, cls).__new__(cls)
            np.random.seed(0)
            self = cls._instance
            self.action_set = action_set(N_AGENTS)

            n_joint_actions = N_ACTIONS ** N_AGENTS
            self.n_states = (WIDTH * HEIGHT) ** N_AGENTS


            self.n_phi = (self.n_states, n_joint_actions, 10)
            self.n_critic = 10

            self.phi = uniform(size=self.n_phi) 
            self.phi_test = self.phi.copy().reshape((self.n_states * n_joint_actions, self.n_critic))
            # self.phi = self.phi / np.abs(self.phi).sum(keepdims=True, axis=-1)


            self.n_actor = 5
            self.n_varphi = (self.n_states, N_ACTIONS, N_AGENTS, self.n_actor)
            self.varphi = uniform(size=self.n_varphi) 
            # self.varphi = self.varphi / np.abs(self.varphi).sum(keepdims=True, axis=-1)
        return cls._instance

    def get_phi(self, state, actions): 
        u = self.action_set.index(tuple(actions))
        res = self.phi[state][u][:]
        return res

    def get_varphi(self, state):
        val = self.varphi[state, ...]
        return val
