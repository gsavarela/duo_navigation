import json
from operator import itemgetter
from pathlib import Path

import gym
from gym.envs.registration import register

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import pi2str, pos2str, act2str, acts2str, best_actions, q2str


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
    if hasattr(agent, 'PI'):
        display_ac(env, agent)
    else:
        display_Q(env, agent)

        
def display_Q(env, agent):
    goal_pos = env.goal_pos
    def bact(x): return best_actions(x, np.array(list(goal_pos)))
    print(env)

    margin = '#' * 45
    print(f'{margin} GOAL {margin}')
    print(f'GOAL: {goal_pos}')
    print(f'{margin} SARSA {margin}')
    import timeit

    print(timeit.timeit(stmt='agent.Q', number=1000, globals=locals()))
    # print(timeit.timeit(stmt='agent.new_Q', number=1000, globals=locals()))

    # np.testing.assert_almost_equal(agent.Q, agent.new_Q)

    print(timeit.timeit(stmt='agent.V', number=1000, globals=locals()))
    # print(timeit.timeit(stmt='agent.new_V', number=1000, globals=locals()))
    # np.testing.assert_almost_equal(agent.V, agent.new_V)

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
        except StopIteration:
            break

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
    from rl import RLAgent
    from pathlib import Path
    from argparse import Namespace

    path = Path('data/20220128152550')
    config_path = Path(path) / 'config.json'
    with config_path.open('r') as f:
        d = json.load(f)
    flags = Namespace(**d)

    # Instanciate
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
        kwargs={'flags': flags}
    )
    # TODO: Verify why env != agent.env
    env = gym.make('duo-navigation-v0')
    agent = RLAgent.load_checkpoint(path, flags.episodes)
    agent.env = env

    main(env, agent, path)
