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
    w_h = width, height
    b = (width * height)
    return sum([coord2index(x, *w_h) * b ** y for y, x in enumerate(positions)])

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
    rows, cols = (coord - 1) 
    return rows * width + cols

def state2pos(state, n_agents=2, width=2, height=2):
    b = (width * height)
    rem = state
    positions = []
    # if state == 4: import ipdb; ipdb.set_trace()
    for i in range(n_agents - 1, -1, -1):
        index = rem // (b ** i)
        positions.append(index2coord(index))
        rem = rem - index * (b ** i)
    # corrects reverse coordinates.
    return positions[-1::-1]
        
        
def index2coord(index, width=2, height=2):
    # | 00 | 01 | 02 | ... | 07 |
    # + -- + -- + -- + ... + -- +
    # | 08 | 09 | 10 | ... | 15 |
    # + -- + -- + -- + ... + -- +
    #           ...
    # | 56 | 57 | 58 | ... | 63 |
    # pos[0] from 1 ... grid.width -1
    # pos[1] from 1 ... grid.width -1
    cols = index // width + 1
    rows = index % width + 1
    return np.array([cols, rows])

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
    if n_agents == 1: return [int(ind % 4)]
    if n_agents == 2: return [int(ind % 4), int(ind // 4)]
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

if __name__ == '__main__':
    n_agents = 2
    states_positions = [
        (0,       [(1, 1), (1, 1)]),
        (1,       [(1, 2), (1, 1)]),
        (2,       [(2, 1), (1, 1)]),
        (3,       [(2, 2), (1, 1)]),
        (4,       [(1, 1), (1, 2)]),
        (5,       [(1, 2), (1, 2)]),
        (6,       [(2, 1), (1, 2)]),
        (7,       [(2, 2), (1, 2)]),
        (8,       [(1, 1), (2, 1)]),
        (9,       [(1, 2), (2, 1)]),
        (10,      [(2, 1), (2, 1)]),
        (11,      [(2, 2), (2, 1)]),
        (12,      [(1, 1), (2, 2)]),
        (13,      [(1, 2), (2, 2)]),
        (14,      [(2, 1), (2, 2)]),
        (15,      [(2, 2), (2, 2)])

    ]
    for state, position in states_positions:
        ag1, ag2 = state2pos(state, n_agents)
     #   print(state, ag1, ag2, position)
        assert [tuple(ag1), tuple(ag2)] == position

