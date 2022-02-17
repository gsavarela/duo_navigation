import numpy as np
from numpy.random import uniform
from utils import action_set, state2pos, coord2index, generate_transitions_from_moves
from deprecated import deprecated
# TODO: extend this to accept new parameters,
# features as a singleton pattern


def get_sa(state, actions=None):
    '''
    Parameters:
    -----------
    * state: int
    * actions: None, int, or list

    Returns:
    -------
    * features: np.array or matrix
    '''
    return Features().get_state_actions(state, actions=actions)

def get_s(state=None):
    '''
    Parameters:
    -----------
    * state: int
    if None return all states.

    Returns:
    -------
    * features: np.array or matrix
    '''
    return Features().get_state(state)
get = get_s


def label():
    return Features().label

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
            # tr: state, action, next_state
            self.transitions = {
                tuple(tr[:2]): tr[-1]for tr in generate_transitions_from_moves()
            }
            
        return cls._instance


    @deprecated
    def get_phi(self, state, actions): 
        u = self.action_set.index(tuple(actions))
        res = self.phi[state][u][:]
        return res

    @deprecated
    def get_varphi(self, state):
        val = self.varphi[state, ...]
        return val

    @deprecated
    def get_onehot(self, state):
        pos = state2pos(state)
        indexes = [coord2index(p) for p in pos]
        indicators = [self.onehot[i, :, j] for j, i in enumerate(indexes)]
        onehot = np.hstack(indicators)
        return onehot

    def set(self, features, n_features=None, n_agents=2, width=2, height=2, **kwargs):
        self.label = features
        self.n_states = (width * height) ** n_agents
        self.n_features = self.n_states
        

        # THis is here for phi and varphi properties.
        self.action_set = action_set(n_agents)
        self.width = width
        self.height = height


        column_rank = 0
        row_rank = 0
        # Change this for tests
        while column_rank != self.n_states and row_rank != self.n_features:
            self.features = np.zeros((self.n_features, self.n_features), dtype=float)
            if 'onehot' in features:
                # this features belong to the state only
                # self.features += np.tile(np.eye(self.n_features), (n_agents, 1, 1)).T
                self.features += np.eye(self.n_features)

            if 'uniform' in features:
                # this features belong to the state only
                self.features += uniform(size=self.features.shape)
            self.features = l2_norm(self.features)
            column_rank = np.linalg.matrix_rank(self.features)
            row_rank = np.linalg.matrix_rank(self.features.T)

    def get_state_actions(self, state, actions=None):
        tr = self.transitions 
        # actions = j return the state after action j is taken (use: update)
        if isinstance(actions, int):
            res = self.features[tr[(state, actions)]]
        # actions is array-like the states[0, ...j..|S|] after actions[0,.. j.. |A|]
        if actions is None:
            next_states = [tr[(state, j)] for j in range(len(self.action_set))]
            res = self.features[next_states, :]
        return res

    def get_state(self, state=None):
        if state is None: return self.features
        return self.features[state]

        
def l2_norm(features):
    # get original shape
    orig_shape = features.shape

    # vector norm over axis 1
    features = features / np.linalg.norm(features, keepdims=True, axis=1)

    return features.reshape(orig_shape)
