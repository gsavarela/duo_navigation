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

def display_critic(env, agent):
    '''Displays Q-value'''
    actions_scores = {}
    actions_display = {}
    actions = [(v, u) for v in range(3) for u in range(3)]

    # rows (coordinate y) 
    for i in range(1, env.height - 1): 
        actions_scores[i] = []
        actions_display[i] = []
        # columns (coordinate x) 
        for j in range(1, env.width - 1): 
            positions = [np.array([j, i]), np.array([j, i])] 
            state = env.state
            # scores = [env.features.get_phi(state, ac) @ agent.omega for ac in actions]
            scores = [features().get_phi(state, ac) @ agent.omega for ac in actions]
            disact = acts2str(actions[np.argmax(scores)])

            actions_scores[i].append(np.argmax(scores))
            actions_display[i].append(disact)

    df_scores = pd.DataFrame.from_dict(actions_scores)
    pretty_print_df(df_scores)

    df_display = pd.DataFrame.from_dict(actions_display)
    pretty_print_df(df_display)
    
    return df_scores, df_display

def display_actor(env, agent):
    '''Displays policy'''
    pi_display = {}
    actions_display = {}
    n_agents = len(env.agents)

    for k in range(n_agents):

        pi_display[k] = {}
        actions_display[k] = {}
        for i in range(1, env.width - 1): 
            pi_display[k][i] = []
            actions_display[k][i] = []
            for j in range(1, env.height - 1): 
                positions = [np.array([i, j]), np.array([i, j])] 
                state = env.state
                # varphi = env.features.get_varphi(state)
                varphi = features().get_varphi(state)
                pi_k = agent.pi(varphi, k)
                pi_display[k][i].append(pi2str(pi_k))
                actions_display[k][i].append(act2str(np.argmax(pi_k)))

    pis_dataframes = []
    for k, pidis in pi_display.items():
        pis_dataframes.append(pd.DataFrame.from_dict(pidis))

    actions_dataframes = []
    for k, actdis in actions_display.items():
        actions_dataframes.append(pd.DataFrame.from_dict(actdis))

    margin = '#'*33 
    for k in range(n_agents):
        print(f'{margin}\tAGENT {k + 1}\t{margin}')
        pretty_print_df(pis_dataframes[k])
        pretty_print_df(actions_dataframes[k])

    return pis_dataframes, actions_dataframes
            

