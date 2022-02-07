''' Centralized versions of actor-critic algorithm

    TODO:
    ----
    * Remove loop from PHI by using advanced broadcasting on env.features.get_phi
    * Cache PHI for faster computations.
    * Create SerializableAgent for saving and loading.

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`

    Zhang, et al. 2018
'''
from itertools import product
from functools import lru_cache
from pathlib import Path
import dill

import numpy as np
from numpy.random import choice

from utils import best_actions, i2q, q2i
from decorators import int2act, act2int
from features import Features

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(keepdims=True)


class TabularCentralizedActorCritic:
    '''Tabular Centralized Actor Critic: A single Actor-Critic

        * Critic evaluates the policy.
        * Actor generates actions w.r.t the full action set.
        * Differs from Centralized actor critic 
        * Extended from Centralized Algorithm-1 Zhang, et al. 2018

    '''
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):

        # Inputs.
        self.action_set = env.action_set

        # TODO: optimize this.
        def phi(x, i):
            return Features().get_phi(x, self.action_set[i])
        self.phi = phi # use this for update.
        def PHI(x): # use this for acting.
            return np.array([self.phi(x, i) for i in range(len(self.action_set))])
        self.PHI = PHI 

        # Constants.
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.n_actions = 4
        self.n_phi = 10
        self.n_varphi = 5
        self.decay = decay

        # Parameters.
        self.Q = np.zeros((self.n_states, len(self.action_set)))
        self.theta = np.zeros(self.n_phi)

        # Loop control.
        self.mu = 0 
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.decay_count = 0
        self.reset()

    @property
    def V(self):
        return self._cache_V(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_V(self, t):
        values = []
        for state in range(self.n_states):
            values.append(self.pi(state) @ self.Q[state, :])
        return np.array(values)

    def reset(self, seed=0):
        np.random.seed(seed)

    @int2act
    def act(self, state):
        return choice(len(self.action_set), p=self.pi(state))

    @act2int
    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.Q[next_state, next_actions] - self.Q[state, actions])

        # Actor step.
        self.Q[state, actions] += self.alpha * delta
        self.theta += self.beta * delta * self.psi(state, actions)

        # Update loop control.
        self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(next_rewards)
        self.step_count += 1
        if self.decay and np.abs(delta) > 1e-6:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count + 1, -0.85)
            self.beta = np.power(self.decay_count + 1, -0.65)

    # Joint action from all agents.
    def pi(self, state):
        return softmax(self.PHI(state) @ self.theta)

    # Joint action from all agents in list format.
    # Interface compatibility between A-C.
    def PI(self, state):
        return self.pi(state).tolist()
    
    def psi(self, state, actions):
        return self.phi(state, actions) - self.pi(state) @ self.PHI(state)

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

class FullyCentralizedActorCriticV2:
    '''Fully Centralized Actor Critic: A single Actor-Critic

        * Critic evaluates w.r.t the full action set.
        * Actor generates actions w.r.t the full action set.
        * Reference: Sutton and Barto, ActorCritic with eligibility traces.
          ---------
    '''
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):
        # Inputs.
        self.phi = Features().get_phi
        self.varphi = Features().get_varphi
        self.action_set = env.action_set
        # TODO: make features#get_phi to get all actions.
        def PHI(x): # use this for Q
            return np.array([self.phi(x, u) for u in self.action_set])
        self.PHI = PHI 

        # Constants.
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.n_actions = 4
        self.n_phi = 10
        self.n_varphi = 5
        self.decay = decay

        # Parameters.
        self.omega = np.zeros(self.n_phi)
        self.omega_z = np.zeros(self.n_phi)
        self.theta = np.zeros((self.n_agents, self.n_varphi))
        self.theta_z = np.zeros((self.n_agents, self.n_varphi))


        # Loop control.
        # TODO: Insert this as parameters.
        self.omega_lambda = 0.0
        self.theta_lambda = 0.0
        self.mu = 0 
        self.step_count = 0
        self.gamma = 0.01
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)
        self.mu = 0
        self.omega_z = np.zeros(self.n_phi)
        self.theta_z = np.zeros((self.n_agents, self.n_varphi))
    
    @property
    def V(self):
        values = []
        for state in range(self.n_states):
            values.append(self.PI(state) @ self.Q[state, :])
        return np.array(values)

    @property
    def V(self):
        return self._cache_V(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        values = []
        for state in range(self.n_states):
            values.append(self.PI(state) @ self.Q[state, :])
        return np.array(values)

    @property
    def Q(self):
        return self._cache_Q(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_Q(self, step_count):
        q_values = []
        for state in range(self.n_states):
            q_values.append(self.PHI(state) @ self.omega)
        return np.array(q_values)


    def act(self, state):
        probs = self.pi(state)
        return [choice(self.n_actions, p=prob) for prob in probs]


    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.phi(next_state, next_actions) - self.phi(state, actions)) @ self.omega 

        # Eligibility Traces.
        self.omega_z = self.omega_lambda * self.omega_z + self.phi(state, actions)
        for i in range(self.n_agents):
            self.theta_z[i, :] = self.theta_lambda * self.theta_z[i, :] + self.psi(state, actions, i)


        # Critic step
        self.omega += self.alpha * delta * self.omega_z
        for i in range(self.n_agents):
            self.theta[i, :] += self.beta * delta * self.theta_z[i, :]

        # Update loop control.
        self.mu += self.gamma * delta
        self.step_count += 1

    # A list of individual actions.
    def pi(self, state, i=None):
        if i is None:
            return [self.pi(state, i) for i in range(self.n_agents)]
        varphi = self.varphi(state)
        return softmax(varphi[:, i, :] @ self.theta[i, :])

    # Joint actions from all agents.
    def PI(self, state):
        pi = self.pi(state) # individual actions
        pi = pi[-1::-1] # reverse order only if n_agents == 2
        return [np.prod(p) for p in product(*pi)]
        
    def psi(self, state, actions,  i):
        varphi = self.varphi(state)
        return varphi[actions[i], i, :] - self.pi(state, i) @ varphi[:, i, :]

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

class FullyCentralizedActorCriticV1:
    '''Fully Centralized Actor Critic: A single Actor-Critic

        * Critic evaluates w.r.t the full action set.
        * Actor generates actions w.r.t the full action set.
        * Differs from Centralized actor critic 
        * Extended from Centralized Algorithm-1 Zhang, et al. 2018

    '''
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):

        # Inputs.
        self.action_set = env.action_set

        # TODO: optimize this.
        self.phi = Features().get_phi # use this for update.
        def PHI(x): # use this for acting.
            return np.array([self.phi(x, u) for u in self.action_set])
        self.PHI = PHI 

        # Constants.
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
        self.n_phi = 10
        self.decay = decay

        # Parameters.
        self.omega = np.zeros(self.n_phi)
        self.theta = np.zeros(self.n_phi)

        # Loop control.
        self.mu = 0 
        self.step_count = 0
        self.alpha = 1 if decay else alpha
        self.beta = 1 if decay else beta
        self.decay_count = 0
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)
        if self.decay:
            self.alpha = 1
            self.beta = 1
            self.decay_count = 0

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

    def act(self, state):
        ind = choice(len(self.action_set), p=self.pi(state))
        return list(self.action_set[ind])

    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.phi(next_state, next_actions) - self.phi(state, actions)) @ self.omega 

        # print(state, actions, np.mean(next_rewards), delta, self.PI(state))
        # import ipdb; ipdb.set_trace()
        # Critic step.
        self.omega += self.alpha * delta * self.phi(state, actions)
        self.theta += self.beta * delta * self.psi(state, actions)

        # Update loop control.
        self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(next_rewards)
        self.step_count += 1
        if self.decay and np.abs(delta) > 1e-6:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count + 1, -0.85)
            self.beta = np.power(self.decay_count + 1, -0.65)

    # Joint action from all agents.
    def pi(self, state):
        return softmax(self.PHI(state) @ self.theta)

    # Joint action from all agents in list format.
    # Interface compatibility between A-C.
    def PI(self, state):
        return self.pi(state).tolist()
    
    def psi(self, state, actions):
        return self.phi(state, actions) - self.pi(state) @ self.PHI(state)

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

class CentralizedActorCritic:
    '''Centralized Actor Critic: A single Actor-Critic

        * Critic evaluates w.r.t the full action set.
        * Actor generates actions w.r.t node by node.
        * Actor once trained acts fully decentralized.
        * Differs from Centralized actor critic 
        * Centralized Algorithm-1 Zhang, et al. 2018

    '''
    def __init__(self, env, alpha=0.5, beta=0.3, decay=True):
        # Inputs.
        self.phi = Features().get_phi
        self.varphi = Features().get_varphi
        self.action_set = env.action_set
        # TODO: make features#get_phi to get all actions.
        def PHI(x): # use this for Q
            return np.array([self.phi(x, u) for u in self.action_set])
        self.PHI = PHI 

        # Constants.
        self.n_agents = len(env.agents)
        self.n_states = env.n_states
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
        self.decay = decay
        self.reset()

    def reset(self, seed=0):
        np.random.seed(seed)
        if self.decay:
            self.alpha = 1
            self.beta = 1
            self.decay_count = 0
    @property
    def V(self):
        values = []
        for state in range(self.n_states):
            values.append(self.PI(state) @ self.Q[state, :])
        return np.array(values)

    @property
    def V(self):
        return self._cache_V(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_V(self, step_count):
        values = []
        for state in range(self.n_states):
            values.append(self.PI(state) @ self.Q[state, :])
        return np.array(values)

    @property
    def Q(self):
        return self._cache_Q(self.step_count)

    @lru_cache(maxsize=1)
    def _cache_Q(self, step_count):
        q_values = []
        for state in range(self.n_states):
            q_values.append(self.PHI(state) @ self.omega)
        return np.array(q_values)


    def act(self, state):
        probs = self.pi(state)
        return [choice(self.n_actions, p=prob) for prob in probs]


    def update(self, state, actions, next_rewards, next_state, next_actions):
        # Time-difference error.
        delta = np.mean(next_rewards) - self.mu + \
                (self.phi(next_state, next_actions) - self.phi(state, actions)) @ self.omega 

        # Critic step.
        self.omega += self.alpha * delta * self.phi(state, actions)
        for i in range(self.n_agents):
            self.theta[i, :] += self.beta * delta * self.psi(state, actions, i)

        # Update loop control.
        self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(next_rewards)
        self.step_count += 1
        if self.decay and np.abs(delta) > 1e-6:
            self.decay_count += 1
            self.alpha = np.power(self.decay_count + 1, -0.85)
            self.beta = np.power(self.decay_count + 1, -0.65)

    # A list of individual actions.
    def pi(self, state, i=None):
        if i is None:
            return [self.pi(state, i) for i in range(self.n_agents)]
        varphi = self.varphi(state)
        return softmax(varphi[:, i, :] @ self.theta[i, :])

    # Joint actions from all agents.
    def PI(self, state):
        pi = self.pi(state) # individual actions
        pi = pi[-1::-1] # reverse order only if n_agents == 2
        return [np.prod(p) for p in product(*pi)]
        
    def psi(self, state, actions,  i):
        varphi = self.varphi(state)
        return varphi[actions[i], i, :] - self.pi(state, i) @ varphi[:, i, :]

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
