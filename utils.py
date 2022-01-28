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

def transition_update_df(tr_dict, experiment_dir):
    # Convert single dataframe into multiple
    df = pd.DataFrame.from_dict(tr_dict). \
            set_index('timestep', inplace=False)

    columns = df.columns.values.tolist()
    # sort key for (state, action) 
    def srtk1(x): return eval(x[-6:])
    # sort key for (state) 
    def srtk2(x): return eval(x[-3:])

    # base transitions.
    trdf = df[['state', 'actions', 'next_rewards', 'next_state', 'next_actions']]

    # q-functions, a-functions, policies
    for k, fn in [('Q', srtk1), ('A', srtk1), ('pi', srtk1), ('V', srtk2)]:
        cix = [k in col for col in columns]
        kdf = df.loc[:, cix]

        kcols = sorted(kdf.columns.values.tolist(), key=fn)
        kdf = kdf.loc[:, kcols]

        kdf = pd.concat((trdf, kdf), axis=1)
        kdf.to_csv(experiment_dir / f'{k}.csv')

    return df

