'''Auxiliary function'''
import numpy as np
import pandas as pd

# `right`, `down`, `left`, `up`
MOVES = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

def pos2str(pos):
    return f'{tuple(pos.tolist())}'

def act2str(act):
    if act == 0: return '>'
    if act == 1: return 'V'
    if act == 2: return '<'
    if act == 3: return '^'
    raise KeyError(f'{act} is not a valid action')

def acts2str(acts):
    moves = ', '.join([act2str(act) for act in acts])
    return f'({moves})'

def pi2str(probs):
    strprobs = [f'{prob:0.2f}' for prob in probs]
    strprobs = ', '.join(strprobs)
    return f'({strprobs})'
q2str = pi2str

def str2bool(v, exception=None):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        if exception is None:
            raise ValueError('boolean value expected')
        else:
            raise exception

# converts integer to a quaternary list.
def i2q(ind, n_agents):
    if n_agents == 1: return [ind % 4]
    if n_agents == 2: return [ind % 4, ind // 4]
    raise KeyError(f'i2q: 0 < {ind} < {4 ** n_agents - 1}')

# converts a quaternary list to an array.
def q2i(acts):
    return [act * (4 ** i) for i, act in enumerate(acts)]


# The best action is the one that brings the agent closest
# to the goal.
def best_actions(agent_pos, goal_pos):

    # Manhattan distance
    def dist(x):
        return np.abs(goal_pos - x).sum()
    res = []
    prev_dist = dist(agent_pos)
    for i, move in enumerate(MOVES):
        next_dist  = dist(agent_pos + move)
        if next_dist < prev_dist: res.append(i) 
    return res

