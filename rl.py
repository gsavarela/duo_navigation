import argparse
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

import gym
from gym.envs.registration import register
import numpy as np

from plots import globally_averaged_plot
from logs import display_actor, display_critic  
from utils import str2bool

from ac import ActorCritic

parser = argparse.ArgumentParser(description='''
    This script trains an Actor-Critic DuoNavigation Gym Experiment.

    Usage:
    -----
    > python -a 0.7 -b 0.2 -d 0 -e 100 -n 2 -s 4 -r 1 -v 0

    Where
''')
parser.add_argument('-a', '--alpha', default=0.5, type=float,
        help='''Alpha is the critic parameter:
                Only active if `decay` is False.''')

parser.add_argument('-b', '--beta', default=0.3, type=float,
        help='''Beta is the actor parameter:
                Only active if `decay` is False.''')

parser.add_argument('-d', '--decay', default=True, type=str2bool,
        help='''Exponential decay of actor and critic parameters:
                Replaces `alpha` and `beta` parameters.''')

parser.add_argument('-e', '--episodes', default=100, type=int,
        help='''Regulates the number of re-starts after the
                GOAL has been reached.''')

parser.add_argument('-n', '--n_agents', default=2, choices=[1, 2], type=int,
        help='''The number of agents on the grid:
                Should be either `1` or `2`. Use `1` for debugging.''')

parser.add_argument('-s', '--size', default=3, type=int,
        help='''The side of the visible square grid.''')

parser.add_argument('-S', '--seed', default=47, type=int,
        help='''An integer to be used as a random seed.''')

parser.add_argument('-r', '--random_starts', default=True, type=str2bool,
        help='''The number of agents on the grid:
                Should be either `1` or `2`. Use `1` for debugging.''')

parser.add_argument('-R', '--render', default=False, type=str2bool,
        help='''Shows the grid during the training.''')

def print_arguments(opts, timestamp):

    print('Arguments Duo Navigation Game:')
    print(f'\tTimestamp: {timestamp}')
    for k, v in vars(opts).items():
        print(f'\t{k}: {v}')

def main(flags):

    # Instanciate
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
        kwargs={'flags': flags}
    )

    # Loop control and execution flags.
    render = flags.render
    episodes = flags.episodes

    # Actor-Critic parameters. 
    env = gym.make('duo-navigation-v0')
    agent = ActorCritic(env, alpha=flags.alpha, beta=flags.beta, decay=flags.decay) 

    # Notify the user what's going on.
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print_arguments(flags, timestamp)

    # Save parameters
    experiment_dir = Path('data') / timestamp
    experiment_dir.mkdir(exist_ok=True)
    config_path = experiment_dir / 'config.json'
    with config_path.open('w') as f:
        json.dump(vars(flags), f)

    duration = 0
    min_step_count = 1e5
    n_agents = len(env.agents)
    globally_averaged_returns = []
    for episode in range(episodes):
        state = env.reset()
        actions = agent.act(state)

        while True:
            if render:
                env.render(mode='human', highlight=True)
                time.sleep(0.1)
                # for i in range(n_agents):
                #     print(f'Agent {i+1}:{env.agents[i].pos}')

            next_state, next_reward, done, _ = env.step(actions)

            agent.update_mu(next_reward)
            next_actions = agent.act(next_state)
            agent.update(state, actions, next_reward, next_state, next_actions)

            state = next_state
            actions = next_actions
            globally_averaged_returns.append(np.mean(agent.mu))
            if done:
                break

        if env.step_count <= min_step_count:
            min_step_count = env.step_count

        duration += env.step_count
        msg = (f'TRAIN: Episode {episode}\t'
                f'steps:{str(env.step_count)}\t'
                f'mu:{agent.mu:0.5f}:\t'
                f'duration:{duration / (episode + 1):0.2f}'
                f'\t:best {min_step_count}.')
        print(msg)

    state = env.reset()
    agent_test = agent
    agent_test.env = env
    agent.save_checkpoints(experiment_dir,str(episodes))

    print(str(env))
    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        actions = agent_test.act(state)
        next_state, next_reward, done, _ = env.step(actions)
        state = next_state
        if done:
            break
    print(f'TEST:\t{env.step_count}\t:{agent_test.mu}')

    globally_averaged_plot(globally_averaged_returns, experiment_dir)
    # display_critic(env, agent)
    # display_actor(env, agent)
    print(str(env))

if __name__ == '__main__':
    # Gather parameters.
    flags = parser.parse_args()

    main(flags)
