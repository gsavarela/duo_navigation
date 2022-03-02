from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

def snapshot_log(episode, env, agent, tr, log_dict, debug=True):
    # Read step variables
    actions = env.action_set
    action_index = actions.index(tuple(tr[1]))
    step_count = env.step_count
    total_step_count = agent.step_count
    reward = np.mean(tr[2])
    

    log_dict['state'].append(int(tr[0]))
    log_dict['action'].append(action_index)
    log_dict['reward'].append(reward)
    log_dict['step_count'].append(step_count)
    log_dict['episode'].append(episode)

    if 'task' not in log_dict: log_dict['task'] = agent.task
    if 'label' not in log_dict: log_dict['label'] = agent.label


    if 'average_reward' not in log_dict:
        avg_reward = reward
    else:
        prev = log_dict['average_reward'][-1]
        avg_reward = (prev * (total_step_count - 1) + reward) \
                        / total_step_count
    log_dict['average_reward'].append(avg_reward)

    

    step_log = (f'TRAIN: Episode: {episode}\t'
                f'Steps: {step_count}\t'
                f'Average Reward: {avg_reward:0.4f}\t')

    if hasattr(agent, 'epsilon'):
        log_dict['epsilon'].append(np.round(agent.epsilon, 4))

    if hasattr(agent, 'tau'):
        log_dict['tau'].append(np.round(agent.tau, 4))
        step_log += f'tau: {log_dict["tau"][-1]:0.4f}\t'

    if hasattr(agent, 'mu'):
        log_dict['mu'].append(np.mean(agent.mu))
        step_log += f'Globally Averaged J: {agent.mu:0.4f}\t' 

    if hasattr(agent, 'delta'):
        log_dict['delta'].append(float(np.round(agent.delta, 4)))

    if debug:
        n_actions = len(actions)
        states_positions_gen = env.next_states()
        try:
            while True:
                state, _ = next(states_positions_gen)
                log_dict[f'V({state})'].append(agent.V[state])
                PI = agent.PI(state)

                log_dict[f'PI({state})'].append(agent.PI(state))

                if hasattr(agent, 'Q'):
                    log_dict[f'Q({state})'].append(agent.Q[state, :].tolist())

                if hasattr(agent, 'A'):
                    log_dict[f'A({state})'].append(agent.A[state, :].tolist())

        except StopIteration:
            pass

    return step_log

def snapshot_state_actions_log(log, log_dir=None):
    sa = list(zip(log['state'], log['action']))

    # TODO: Ideal build a dataframe from dictionary
    sac = [k + (v,) for k, v in Counter(sa).items()]

    df = pd.DataFrame.from_records(sac,columns=['state', 'action', 'count']). \
            pivot(index='state', columns='action', values='count')

    sa_path = Path(log_dir) / 'state_action.csv'
    if log_dir is not None:
        df.to_csv(sa_path)
    return df


