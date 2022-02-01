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

from plots import (globally_averaged_plot, advantages_plot,
        q_values_plot, display_ac, display_ac2, validation_plot)
from logs import logger, transition_update_log, best_actions_log
from utils import str2bool

# from centralized import CentralizedActorCritic as RLAgent
from sarsa import SARSATabular as RLAgent
# from ac import OptimalAgent as RLAgent

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

parser.add_argument('-e', '--episodes', default=1, type=int,
        help='''Regulates the number of re-starts after the
                GOAL has been reached. Keep in mind that the
                task is continuing''')

parser.add_argument('-E', '--episodic', default=True, type=str2bool,
        help='''Controls the nature of the task as continuing or episodic.''')

parser.add_argument('-m', '--max_steps', default=10000, type=int,
        help='''Regulates the maximum number of steps.''')

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

parser.add_argument('-R', '--render', default=False, type=str2bool, help='''Shows the grid during the training.''')

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
    max_steps = flags.max_steps
    n_agents = flags.n_agents

    # Actor-Critic parameters. 
    env = gym.make('duo-navigation-v0').unwrapped
    # Actor-Critic Agent
    # agent = RLAgent(env, alpha=flags.alpha, beta=flags.beta, decay=flags.decay) 
    agent = RLAgent(env) 

    # Notify the user what's going on.
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print_arguments(flags, timestamp)

    # Save parameters
    experiment_dir = Path('data') / timestamp
    experiment_dir.mkdir(exist_ok=True)
    config_path = experiment_dir / 'config.json'
    with config_path.open('w') as f:
        json.dump(vars(flags), f)
    # state_actions 
    state_actions = best_actions_log(env, experiment_dir)
    
    n_agents = len(env.agents)
    globally_averaged_returns = []
    rewards = []
    q_values = []
    advantages = []
    tr_dict = defaultdict(list)
    # logger_log = logger(env, agent, state_actions)  
    for episode in range(episodes):
        state = env.reset()
        actions = agent.act(state)

        while True:
            if render:
                env.render(mode='human', highlight=True)
                time.sleep(0.05)
            next_state, next_reward, done, _ = env.step(actions)
            # agent.update_mu(next_reward)
            next_actions = agent.act(next_state)
            tr = (state, actions, next_reward, next_state, next_actions)
            # Log.
            # globally_averaged_returns.append(np.mean(agent.mu))
            rewards.append(np.mean(next_reward))
            # TODO: Context manager should handle this use case.
            # logger_log(advantages, q_values, tr, tr_dict, updated=False)
            agent.update(*tr)

            # logger_log(advantages, q_values, tr, tr_dict, updated=True)

            print(f'TRAIN: Episode: {episode}\t'
                  f'Steps: {env.step_count}\t'
                  f'Mean reward: {np.mean(rewards)}')

            # print(f'TRAIN: Episode: {episode}\t'
            #       f'Steps: {env.step_count}\t'
            #       f'Globally Averaged J: {agent.mu}')
            state = next_state 
            actions = next_actions
            if done:
                break
    agent.save_checkpoints(experiment_dir,str(episodes))
    globally_averaged_plot(np.cumsum(rewards) / np.arange(1, len(rewards) + 1), experiment_dir)
    # advantages_plot(advantages, experiment_dir, state_actions)
    # q_values_plot(q_values, experiment_dir, state_actions)
    # transition_update_log(tr_dict, experiment_dir)

    # if agent.n_agents == 1:
    #     display_ac(env, agent)
    # else:
    #     display_ac2(env, agent)
    import ipdb; ipdb.set_trace()
    print(f'Experiment path:\t{experiment_dir.as_posix()}')


    state = env.reset()
    validation_rewards = []
    for _ in range(100):
       if render:
           env.render(mode='human', highlight=True)
           time.sleep(0.1)

       next_state, next_reward, done, _ = env.step(agent.act(state))
       validation_rewards.append(np.mean(next_reward))
       state = next_state

       if done:
           break
    validation_plot(validation_rewards)

if __name__ == '__main__':
    # Gather parameters.
    flags = parser.parse_args()

    main(flags)
