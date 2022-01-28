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

def advantages_plot(advantages, results_path, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    
    n_steps = len(advantages)
    # Makes a list of dicts.
    ld = [dict(adv) for adv in advantages]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
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

def q_values_plot(q_values, results_path, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    
    n_steps = len(q_values)
    # Makes a list of dicts.
    ld = [dict(adv) for adv in q_values]
    # Converts a list of dicts into dictionary of lists.
    dl = {k: [d[k] for d in ld] for k in ld[0]}
    for x, ys in state_actions:
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

def display_ac(env, agent):
    
    goal_pos = env.goals_pos[0]
    def phi(x , y):
        return env.features.get_phi(x, y)
    print(env)
    margin = '#' * 30
    print(f'{margin} ACTOR {margin}')
    for k in range(agent.n_agents):
        for i in range(1, env.width - 1):
            for j in range(1, env.height - 1):
                agent_pos = np.array([i, j])
                if not np.array_equal(agent_pos, goal_pos):
                    # position to state
                    state = env.state.lin(agent_pos)
                    varphi = env.features.get_varphi(state)
                    pi_k = agent.pi(varphi, k)
                    act = np.argmax(pi_k)
                    msg = (f'\t{state}\t{pos2str(agent_pos)}'
                           f'\t{pi2str(pi_k)}\t{act2str(act)}'
                           f'\t{acts2str(best_actions(agent_pos, goal_pos))}')
                    print(msg)

    print(f'{margin} CRITIC {margin}')
    actions = [[a] for a in range(agent.n_actions)]
    for k in range(agent.n_agents):
        for i in range(1, env.width - 1):
            for j in range(1, env.height - 1):
                agent_pos = np.array([i, j])
                qs = []
                if not np.array_equal(agent_pos, goal_pos):
                    # position to state
                    state = env.state.lin(agent_pos)
                    for act in actions:
                        qs.append(phi(state, act) @ agent.omega)
                    msg = (f'\t{state}\t{pos2str(agent_pos)}'
                           f'\t{q2str(qs)}\t{act2str(np.argmax(qs))}'
                           f'\t{acts2str(best_actions(agent_pos, goal_pos))}')
                    print(msg)

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
