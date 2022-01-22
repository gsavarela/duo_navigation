import time
import argparse
from collections import defaultdict
from copy import deepcopy

import gym
from gym.envs.registration import register
import pandas as pd
import numpy as np

from ac import ActorCritic


def pretty_print_df(df):
    # Pandas display options.
    with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', 1000,
                            'display.colheader_justify', 'center',
                            'display.precision', 3):
        print(df)

def act2str(act):
    if act == 0: return '<'
    if act == 1: return '>'
    if act == 2: return '^'
    raise KeyError(f'{act} is not a valid action')

def act2str2(acts):
    moves = ', '.join([act2str(act) for act in acts])
    return f'({moves})'

def pi2str(probs):
    strprobs = [f'{prob:0.2f}' for prob in probs]
    strprobs = ', '.join(strprobs)
    return f'({strprobs})'

def display_critic(env, agent):
    '''Displays Q-value'''
    actions_scores = {}
    actions_display = {}
    actions = [(v, u) for v in range(3) for u in range(3)]
    for i in range(1, env.width - 1): 
        actions_scores[i] = []
        actions_display[i] = []
        for j in range(1, env.height - 1): 
            positions = [np.array([i, j]), np.array([i, j])] 
            state = env.state.get(positions)
            scores = [env.features.get_phi(state, ac) @ agent.omega for ac in actions]
            disact = act2str2(actions[np.argmax(scores)])

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
    # actions = [(v, u) for v in range(3) for u in range(3)]

    for k in range(2):

        pi_display[k] = {}
        for i in range(1, env.width - 1): 
            pi_display[k][i] = []
            for j in range(1, env.height - 1): 
                positions = [np.array([i, j]), np.array([i, j])] 
                state = env.state.get(positions)
                varphi = env.features.get_varphi(state)
                pi_k = agent.pi(varphi, k)
                pi_display[k][i].append(pi2str(pi_k))

    dataframes = []
    for k, pidis in pi_display.items():
        dataframes.append(pd.DataFrame.from_dict(pidis))

    margin = '#'*33 
    print(f'{margin}\tAGENT 0\t{margin}')
    pretty_print_df(dataframes[0])
    print(f'{margin}\tAGENT 1\t{margin}')
    pretty_print_df(dataframes[1])

    return dataframes

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

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--episodes', default=100, type=int)
parser.add_argument('-r', '--render', default=False, type=str2bool)

def main(episodes, render):

    # Instanciate
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
    )

        
    first = True
    duration = 0
    min_step_count = 1e5
    global_average_returns = []
    for episode in range(episodes):
        env = gym.make('duo-navigation-v0')
        if first:
            agent = ActorCritic(env) 
            first = False
        else:
            agent.env = env
        state = env.reset()
        agent.reset()
        actions = agent.act(state)

        while True:
            if render:
                env.render(mode='human', highlight=True)
                time.sleep(0.1)

            next_state, next_reward, done, _ = env.step(actions)

            agent.update_mu(next_reward)
            next_actions = agent.act(next_state)
            agent.update(state, actions, next_reward, next_state, next_actions)

            state = next_state
            actions = next_actions

            if done:
                break

        if env.step_count < min_step_count:
            min_step_count = env.step_count
            agent_best = deepcopy(agent)
        duration = 0.95 * duration + 0.05 * env.step_count
        global_average_returns.append(agent.mu)
        print(f'TRAIN: Episode {episode}:\t{env.step_count}\t:{agent.mu}: duration:{duration}')

    env = gym.make('duo-navigation-v0')
    state = env.reset()
    agent_test = agent_best
    agent_test.env = env
    agent_test.reset()
    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        actions = agent_test.act(state)
        next_state, next_reward, done, _ = env.step(actions)

        state = next_state
        if done:
            break

    print(f'TEST:\t{env.step_count}\t:{agent_test.mu}')

    # display_critic(env, agent)
    display_actor(env, agent)

if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags.episodes, flags.render)
