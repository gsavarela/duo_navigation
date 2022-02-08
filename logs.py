from collections import defaultdict
import re
import numpy as np
import pandas as pd

from utils import act2str, acts2str, pi2str, best_actions
from features import Features

def snapshot_log(episode, env, agent, tr, log_dict, debug=True):

    actions = env.action_set
    log_dict['state'].append(int(tr[0]))
    log_dict['action'].append(actions.index(tuple(tr[1])))
    log_dict['reward'].append(np.mean(tr[2]))
    log_dict['step_count'].append(agent.step_count)
    log_dict['episode'].append(episode)

    step_log = (f'TRAIN: Episode: {episode}\t'
                f'Steps: {env.step_count}\t'
                f'Average Reward: {np.mean(log_dict["reward"]):0.4f}\t')
    
    if hasattr(agent, 'mu'):
        log_dict['mu'].append(np.mean(agent.mu))
        step_log += f'Globally Averaged J: {agent.mu:0.4f}\t' 

    if debug:
        n_actions = len(actions)
        states_positions_gen = env.next_states()
        try:
            while True:
                state, _ = next(states_positions_gen)
                log_dict[f'V({state})'].append(agent.V[state])
                PI = agent.PI(state)
                log_dict[f'Q({state})'].append(agent.Q[state, :].tolist())
                log_dict[f'A({state})'].append((agent.Q[state, :] - agent.V[state]).tolist())
                log_dict[f'PI({state})'].append(agent.PI(state))

        except StopIteration:
            pass

    return step_log
