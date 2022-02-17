from collections import defaultdict
import json
from operator import itemgetter
from pathlib import Path

import gym
from gym.envs.registration import register

import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import pi2str, pos2str, act2str, act2str2, best_actions, q2str

import numpy as np
import pandas as pd


plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0
CENTRALIZED_AGENT_COLOR = (0.2, 1.0, 0.2)
SMOOTHING_CURVE_COLOR = (0.33,0.33,0.33)

def snapshot_plot(snapshot_log, img_path):

    episodic = 'mu' not in snapshot_log
    episodes = snapshot_log['episode']
    rewards = snapshot_log['reward']

    # TODO: remove get and make default __itemgetter__. 
    label = snapshot_log.get('label', None)
    task = snapshot_log.get('task', 'episodic')

    if task == 'episodic':
        epsilons = snapshot_log['epsilon']
        cumulative_rewards_plot(rewards, img_path, label)
        episode_duration_plot(episodes, epsilons, img_path, label=label)
        episode_rewards_plot(episodes, rewards, img_path, label=label)
    else:
        # For continous tasks
        globally_averaged_plot(snapshot_log['mu'], img_path, episodes)

    
    snapshot_path = img_path / 'snapshot.json'
    with snapshot_path.open('w') as f:
        json.dump(snapshot_log, f)
# use this only for continuing tasks.
# episodes is a series with the episode numbers
def globally_averaged_plot(mus, img_path, episodes):
    
    globally_averaged_return = np.array(mus)
    
    episodes = np.array(episodes)
    globally_averaged_episodes = []
    episodes_to_plot = (int(np.min(episodes)), int(np.mean(episodes)), int(np.max(episodes)))
    for episode in episodes_to_plot: 
        globally_averaged_episodes.append(globally_averaged_return[episodes == episode])
    Y = np.vstack(globally_averaged_episodes).T

    n_steps = Y.shape[0]
    X = np.linspace(1, n_steps, n_steps)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    labels = [f'episode {epis}' for epis in episodes_to_plot]
    plt.plot(X, Y, label=labels)
    plt.xlabel('Time')
    plt.ylabel('Globally Averaged Return J')
    plt.legend(loc=4)

    file_name = img_path / 'globally_averaged_return_per_episode.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'globally_averaged_return_per_episode.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def cumulative_rewards_plot(rewards, img_path, label=None):
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    X = np.linspace(1, len(rewards), len(rewards))
    Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    suptitle = 'Team Return' 
    y_label = 'Cumulative Averaged Reward'
    x_label = 'Timestep'


    plt.suptitle(suptitle)
    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label=label)
    plt.plot(X, Y_smooth[:, 1], label=f'Smoothed {label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)

    file_name = img_path / 'cumulative_averaged_rewards.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'cumulative_averaged_rewards.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def episode_rewards_plot(episodes, rewards, img_path, label=None):
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    target = 1

    rewards = np.array(rewards)
    episodes = np.array(episodes)
    rewards_episodes = []
    episodes_to_plot = np.arange(np.max(episodes))
    X = np.linspace(1, len(episodes_to_plot), len(episodes_to_plot))
    for episode in episodes_to_plot: 
        rewards_episodes.append(np.sum(rewards[episodes == episode]))

    Y = np.array(rewards_episodes)
    Y_smooth = sm.nonparametric.lowess(Y, X, frac=0.10)

    suptitle = 'Episode Return vs Target' 
    y_label = 'Smoothed Return Per Episode'
    x_label = 'Episodes'


    plt.suptitle(suptitle)
    plt.axhline(y=target, c='red', label='target')
    plt.plot(X, Y_smooth[:, 1], c=SMOOTHING_CURVE_COLOR, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)
    file_name = img_path / 'return_per_episode.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'return_per_episode.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# Training 2-axes plot of episode length and vs episilon.
def episode_duration_plot(episodes, epsilons, img_path, label=None):
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    epsilons = np.array(epsilons)
    episodes = np.array(episodes)
    episodes_to_plot = np.arange(np.max(episodes))
    X = np.linspace(1, len(episodes_to_plot), len(episodes_to_plot))

    episodes_duration = []
    episodes_epsilon = []
    for episode in episodes_to_plot: 
        episodes_duration.append(np.sum(episodes == episode))
        episodes_epsilon.append(np.mean(epsilons[episodes == episode]))

    Y1 = np.array(episodes_duration)
    Y1_smooth = sm.nonparametric.lowess(Y1, X, frac=0.10)
    Y2 = np.array(episodes_epsilon)

    suptitle = 'Duration vs. Epsilon' 
    if label is not None:
        suptitle += f': {label}'

    y1_label = 'Duration'
    y2_label = 'Epsilon'
    x_label = 'Episodes'

    #define colors to use
    c1 = 'steelblue'
    c2 = 'red'

    #define subplots
    fig, ax = plt.subplots()

    #add first line to plot
    ax.plot(X, Y1_smooth[:, 1], color=SMOOTHING_CURVE_COLOR)

    #add x-axis label
    ax.set_xlabel(x_label)

    #add y-axis label
    ax.set_ylabel(f'Smoothed {y1_label}', color=SMOOTHING_CURVE_COLOR)

    #define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    #add second line to plot
    ax2.plot(X, Y2, color=c2)

    #add second y-axis label
    ax2.set_ylabel(y2_label, color=c2)

    plt.suptitle(suptitle)
    file_name = img_path / 'duration_vs_epsilon.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'duration_vs_epsilon.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def advantages_plot(advantages, results_path, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    
    n_steps = len(advantages)
    # Makes a list of dicts.
    ld = [dict(adv) for adv in advantages if adv[-1] is not None]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
        if x in dl:
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)
            X = np.linspace(1, n_steps, n_steps)
            Y = np.array(dl[x])
            labels = tuple([f'Best action {act2str(y)}' for y in ys])

            plt.suptitle(f'Advantages State {x}')
            plt.plot(X,Y, label=labels)
            plt.xlabel('Timesteps')
            plt.ylabel('Advantages')
            plt.legend(loc='center right')

            file_name = (results_path / f'advantages_state_{x}.pdf').as_posix()
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            file_name = (results_path / f'advantages_state_{x}.png').as_posix()
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            plt.close()

def q_values_plot(q_values, results_path, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    
    n_steps = len(q_values)
    # Makes a list of dicts.
    ld = [dict(qval) for qval in q_values if qval[-1] is not None]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
        if x in dl:
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)
            X = np.linspace(1, n_steps, n_steps)
            Y = np.array(dl[x])
            labels = tuple([f'Best action {act2str(y)}' for y in ys])

            plt.suptitle(f'Q-values State {x}')
            plt.plot(X,Y, label=labels)
            plt.xlabel('Timesteps')
            plt.ylabel('Relative Q-values')
            plt.legend(loc='center right')

            file_name = (results_path / f'q_values_state_{x}.pdf').as_posix()
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            file_name = (results_path / f'q_values_state_{x}.png').as_posix()
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            plt.close()


# TODO: port display_ac
def display_policy(env, agent):
    #if hasattr(agent, 'PI'):
    return display_ac(env, agent)
    # else:
    #     display_Q(env, agent)

        
def display_Q(env, agent):
    goal_pos = env.goal_pos
    def bact(x): return best_actions(x, np.array(list(goal_pos)))
    print(env)

    margin = '#' * 45
    print(f'{margin} GOAL {margin}')
    print(f'GOAL: {goal_pos}')
    print(f'{margin} SARSA {margin}')

    action_set = env.action_set
    states_positions_gen = env.next_states()
    while True:
        try:
            state, pos = next(states_positions_gen)
            actions_log = act2str2(action_set[np.argmax(agent.Q[state, :])])

            best_log = ', '.join([act2str2(bact(p)) for p in pos])
            pos_log = ', '.join([pos2str(p) for p in pos])
            msg = (f'\t{state}'
                   f'\t{agent.V[state]:0.3f}'
                   f'\t{pos_log}'
                   f'\t{q2str(agent.Q[state, :])}'
                   f'\t{actions_log}'
                   f'\t{best_log}')
            print(msg)
        except StopIteration:
            break

def display_ac(env, agent):
    goal_pos = env.goal_pos
    def bact(x):
        return best_actions(
            x,
            np.array(list(goal_pos)),
            width=env.width - 2,
            height= env.height - 2
        )
    print(env)


    action_set = env.action_set
    states_positions_gen = env.next_states()
    margin = '#' * 30
    data = defaultdict(list)
    print(f'{margin} GOAL {margin}')
    print(f'GOAL: {goal_pos}')
    print(f'{margin} ACTOR {margin}')
    while True:
        try:
            state, pos = next(states_positions_gen)
            pi_log = pi2str(agent.PI(state))
            max_action = np.argmax(agent.PI(state))
            actions_log = act2str2([max_action])
            actions_optimal = bact(pos)
            best_log = act2str2(actions_optimal)
            pos_log = ', '.join([pos2str(p) for p in pos])
            msg = (f'\t{state}\t{pos_log}'
                   f'\t{agent.V[state]:0.2f}'
                   f'\t{pi_log}\n'
                   f'\t{state}\t{pos_log}'
                   f'\t{agent.V[state]:0.2f}'
                   f'\t{actions_log}\tin\t{best_log}: {max_action in actions_optimal}\n'
                   f'{"-" * 150}')
            print(msg)
            data['state'].append(state)
            data['Coord 1'].append(tuple(pos[0]))
            data['Coord 2'].append(tuple(pos[1]))
            data['V'].append(np.round(agent.V[state], 2))
            data['move_most_likely'].append(actions_log)
            data['move_optimal'].append(best_log)
            pr_success = 0
            for i, pi in enumerate(agent.PI(state)):  
                data[f'PI(state, {i})'].append(np.round(pi, 2))
                if i in actions_optimal: pr_success += pi
            data[f'PI(state, success)'].append(np.round(pr_success, 2))
        except StopIteration:
            break

    if hasattr(agent, 'Q'):
        print(f'{margin} CRITIC {margin}')
        states_positions_gen = env.next_states()
        while True:
            try:
                state, pos = next(states_positions_gen)

                max_action = np.argmax(agent.Q[state, :])

                actions_log = act2str2([max_action])
                actions_optimal = bact(pos)
                best_log = act2str2(actions_optimal)
                pos_log = ', '.join([pos2str(p) for p in pos])
                msg = (f'\t{state}'
                       f'\t{agent.V[state]:0.2f}'
                       f'\t{pos_log}'
                       f'\t{q2str(agent.Q[state, :])}'
                       f'\t{actions_log}'
                       f'\t{best_log}')
                print(msg)

                for i, q in enumerate(agent.Q[state, :]):  
                    data[f'Q(state, {i})'].append(np.round(q, 2))
            except StopIteration:
                break
    df = pd.DataFrame.from_dict(data). \
            set_index('state')
    
    return df

def validation_plot(rewards):

    Y = np.cumsum(rewards)
    n_steps = len(rewards)
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    X = np.linspace(1, n_steps, n_steps)
    plt.suptitle(f'Validation Round')
    plt.plot(X,Y)
    plt.xlabel('Timesteps')
    plt.ylabel('Accumulated Averaged Rewards.')
    plt.show()

if __name__ == '__main__':
    import json
    from pathlib import Path
    from argparse import Namespace
    from agents import get_agent

    # path = Path('data/20220205155519')
    # path = Path('data/20220205164323')
    # SARSA Tabular data/20220205175246 
    # path = Path('data/20220205175246')
    # SARSA SG 20220205195417
    # FullyCentralizedAC 20220205171328

    EXPERIMENTS = [
        # ('SarsaTabular', '20220207043611', True),
        ('SarsaTabular', 'data/20220205175246', True),
        ('Centralized A/C', 'data/20220205171612', False),
        ('SARSA SG', 'data/20220205195417', False),
        ('FullyCentralized A/C', 'data/20220205171328', False),
        ('TabularFullyCentralized A/C', 'data/20220205182447', False),
    ]

    labels = []
    Ys = []
    for label, path, episodic in EXPERIMENTS:

        print(f'Experiment ({label}): {path}')

        labels.append(label)
        path = Path(path)
        snapshot_path = Path(path) / 'snapshot.json'
        with snapshot_path.open('r') as f:
            snapshot = json.load(f)

        config_path = Path(path) / 'config.json'
        with config_path.open('r') as f: flags = json.load(f)

        # change from training to validation
        flags['max_steps'] = 100
        flags['episodic'] = False
        flags = Namespace(**flags)

        # Instanciate
        register(
            id='duo-navigation-v0',
            entry_point='env:DuoNavigationGameEnv',
            kwargs={'flags': flags}
        )
        # TODO: Verify why env != agent.env
        chkpt_num = max([int(p.parent.stem) for p in path.rglob('*chkpt')])
        env = gym.make('duo-navigation-v0')
        agent = get_agent(env, flags).load_checkpoint(path, str(chkpt_num))
        agent.env = env
        if episodic: agent.epsilon = 0

        #df = display_policy(env, agent)
        #df.to_csv(path / 'policy.csv', sep='\t')
        rewards = []
        for episode in range(100):
            state = env.reset()
            agent.reset()
            actions = agent.act(state)

            episode_rewards = []
            while True:
                next_state, next_reward, done, _ = env.step(agent.act(state))
                
                episode_rewards.append(np.mean(next_reward))
                state = next_state
                if done: break
            rewards.append(np.cumsum(episode_rewards) / np.arange(1, 101))

        # TODO: LOOP AND ADD LABELS.
        Y = np.mean(np.stack(rewards), axis=0)
        Ys.append(Y)

    Y = np.stack(Ys).T
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    X = np.linspace(1, Y.shape[0], Y.shape[0])
    plt.suptitle(f'Validation Round (N=100)')
    plt.plot(X,Y, label=labels)
    plt.xlabel('Timesteps')
    plt.ylabel('Validation Averaged Rewards.')
    plt.legend(loc=4)
    plt.show()
    

