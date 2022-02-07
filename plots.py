from collections import defaultdict
import json
from operator import itemgetter
from pathlib import Path

import gym
from gym.envs.registration import register

import matplotlib
import matplotlib.pyplot as plt

from utils import pi2str, pos2str, act2str, acts2str, best_actions, q2str

import numpy as np
import pandas as pd


plt.style.use('ggplot')

FIGURE_X = 6.0
FIGURE_Y = 4.0
CENTRALIZED_AGENT_COLOR = (0.2, 1.0, 0.2)

def snapshot_plot(snapshot_log, img_path):
    cumulative_rewards_plot(snapshot_log['reward'], img_path)

    # For continous tasks
    if 'mu' in snapshot_log:
        globally_averaged_plot(snapshot_log['mu'], img_path)

    
    snapshot_path = img_path / 'snapshot.json'
    with snapshot_path.open('w') as f:
        json.dump(snapshot_log, f)

def globally_averaged_plot(mus, img_path):
    
    globally_averaged_return = np.array(mus)
    n_steps = globally_averaged_return.shape[0]


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    X = np.linspace(1, n_steps, n_steps)
    Y = globally_averaged_return

    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label='Centralized')
    plt.xlabel('Time')
    plt.ylabel('Globally Averaged Return J')
    plt.legend(loc=4)

    file_name = img_path / 'globally_averaged_return.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'globally_averaged_return.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# use this only for continuing tasks.
# episodes is a series with the episode numbers
def globally_averaged_plot2(mus, img_path, episodes):
    
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

def cumulative_rewards_plot(rewards, img_path):
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    X = np.linspace(1, Y.shape[0], Y.shape[0])

    plt.plot(X, Y, c=CENTRALIZED_AGENT_COLOR, label='Centralized')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Averaged Reward')
    plt.legend(loc=4)

    file_name = img_path / 'cumulative_averaged_reward.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'cumulative_averaged_reward.png'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def cumulative_rewards_plot2(rewards, img_path, episodes, episodic=False):
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    rewards = np.array(rewards)
    episodes = np.array(episodes)
    rewards_episodes = []
    if episodic:
        episodes_to_plot = np.arange(np.max(episodes))
        for episode in episodes_to_plot: 
            rewards_episodes.append(np.mean(rewards[episodes == episode]))
        Y = np.array(rewards_episodes)
        y_label = 'Average Reward Per Episode'
        x_label = 'Episodes'
        labels = 'SARSATabular'

    else:
        episodes_to_plot = (int(np.min(episodes)), int(np.mean(episodes)), int(np.max(episodes)))
        for episode in episodes_to_plot: 
            rewards_episodes.append(rewards[episodes == episode])
        Y = np.vstack(rewards_episodes).T

        n = np.tile(np.arange(1, Y.shape[0] + 1).reshape((-1, 1)), len(episodes_to_plot))

        Y = np.cumsum(Y, axis=0) / n
        y_label = 'Cumulative Averages Reward Per Episode'
        x_label = 'Timesteps'
        labels = [f'episode {epis}' for epis in episodes_to_plot]

    X = np.linspace(1, Y.shape[0], Y.shape[0])

    plt.plot(X, Y, label=labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=4)

    file_name = img_path / 'cumulative_averaged_reward_per_episode.pdf'
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = img_path / 'cumulative_averaged_reward_per_episode.png'
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
            actions_log = acts2str(action_set[np.argmax(agent.Q[state, :])])

            best_log = ', '.join([acts2str(bact(p)) for p in pos])
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
    def bact(x): return best_actions(x, np.array(list(goal_pos)))
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
            max_actions = np.argmax(agent.PI(state))
            actions_log = acts2str(action_set[max_actions])
            best_log = ', '.join([acts2str(bact(p)) for p in pos])
            pos_log = ', '.join([pos2str(p) for p in pos])
            msg = (f'\t{state}\t{pos_log}'
                   f'\t{agent.V[state]:0.2f}'
                   f'\t{pi_log}\t{actions_log}'
                   f'\t{best_log}')
            print(msg)
            data['state'].append(state)
            data['Coord 1'].append(tuple(pos[0]))
            data['Coord 2'].append(tuple(pos[1]))
            data['V'].append(np.round(agent.V[state], 2))
            data['move'].append(actions_log)
            for i, pi in enumerate(agent.PI(state)):  
                data[f'PI(state, {i})'].append(np.round(pi, 2))
        except StopIteration:
            break

    print(f'{margin} CRITIC {margin}')
    states_positions_gen = env.next_states()
    while True:
        try:
            state, pos = next(states_positions_gen)

            max_action = np.argmax(agent.Q[state, :])
            actions_log = acts2str(action_set[max_action])

            best_log = ', '.join([acts2str(bact(p)) for p in pos])
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

        # if not episodic:
        #     globally_averaged_plot2(snapshot['mu'], path, snapshot['episode'])
        # cumulative_rewards_plot2(snapshot['reward'], path, snapshot['episode'], episodic=episodic)

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
    

    # main(env, agent, path)
