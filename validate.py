import json
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd

import utils


def sort(x):
    x = sorted(x, key=itemgetter(1))
    x = sorted(x, key=itemgetter(0))
    return x

def generate_transitions_from_csv():
    path = Path('data/AC-CHALLENGE/tabular/tr.csv')

    df = pd.read_csv(path.as_posix())
    

    df['next_state'] = np.insert(df['state'].values[1:], -1, 0)
    transitions = []
    for episode in range(20):
        # Use group by to select transitions
        episode_df = df[df['episode'] == episode].\
                        groupby(by=['state', 'action', 'next_state']).\
                        count()
        # Drop first and last
        episode_df = episode_df[1:-1]
        transitions += episode_df.index.tolist()

    transitions_unique = sort(set(transitions))
    for state in range(16):
        state_transitions = [tr for tr in transitions_unique if tr[0] == state]
        if len(state_transitions) > 16:
            print(state, state_transitions)
    
    tr_path = path.parent / 'tr.json'
    with tr_path.open('w') as f:
        json.dump(transitions_unique, f)

def generate_transitions_from_moves(width=2, height=2):
    action_set = utils.action_set(2)

    # Get pos from state for this grid.
    def getpos(x):
        return utils.state2pos(x, n_agents=2, width=width, height=height)

    # Get state from positions
    def getstate(x):
        return utils.pos2state(x, n_agents=2, width=width, height=height)

    # Correct move on grid.
    def safemove(x):
        return np.clip(x, [1, 1], [width, height]) 

    moves = utils.MOVES
    tr = []
    for state in range(16):
        ag1_pos, ag2_pos = getpos(state)
        for ii, ag_moves in enumerate(action_set):
            ag1_move, ag2_move = ag_moves
            next_ag1_pos = safemove(ag1_pos + moves[ag1_move])
            next_ag2_pos = safemove(ag2_pos + moves[ag2_move])
            next_state = getstate([next_ag1_pos, next_ag2_pos])
            tr.append((state, ii, next_state))

    return tr

if __name__ == '__main__':
    tr = generate_transitions_from_moves()
    for state in range(16):
         sl = slice(state * 16, (state + 1) * 16)
         print(state, tr[sl])

