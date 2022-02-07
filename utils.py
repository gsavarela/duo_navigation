'''Auxiliary function'''
from itertools import product
from operator import itemgetter

import numpy as np
import pandas as pd

# `right`, `down`, `left`, `up`
MOVES = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

def pos2str(pos):
    return f'{tuple(pos.tolist())}'

def pos2state(positions, n_agents=2, width=2, height=2):
    return sum([self.coord2index(x) * y for x, y in zip(positions, self.pow)])

# width, height = grid.width-2, grid.height-2
def coord2index(coord, width=2, height=2):
    # | 00 | 01 | 02 | ... | 07 |
    # + -- + -- + -- + ... + -- +
    # | 08 | 09 | 10 | ... | 15 |
    # + -- + -- + -- + ... + -- +
    #           ...
    # | 56 | 57 | 58 | ... | 63 |
    # pos[0] from 1 ... grid.width -1
    # pos[1] from 1 ... grid.width -1
    rows, cols = (np.array(coord) - 1) 
    return rows * self.width + cols

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
    return int(sum([act * (4 ** i) for i, act in enumerate(acts)]))


# The best action is the one that brings the agent closest
# to the goal.
def best_actions(agent_pos, goal_pos, width=None, height=None):

    if not ((width is None) or (height is None)):
        # Handles grid boundaries
        def next_position(pos, move):
            npos = pos + move
            if np.min(npos) < 1 or npos[0] > width or npos[1] > height:
                return pos  # don't make the move.
            return npos
    else:
        # Handles only lower grid boundaries 
        def next_position(pos, move):
            npos = pos + move
            return pos if np.min(npos) < 1 else npos


    # Manhattan distance
    def dist(x):
        return np.abs(goal_pos - x).sum()

    prev_dist = dist(agent_pos)
    res = []
    for i, move in enumerate(MOVES):
        next_dist  = dist(next_position(agent_pos, move))
        if next_dist < prev_dist or \
            (next_dist == prev_dist and prev_dist == 0):
            res.append(i) 
    return res

def action_set(n_agents):
    res = product(np.arange(4).tolist(), repeat=n_agents) 
    if n_agents== 2: res = sorted(res, key=itemgetter(1))
    return [*res] # no generators.
