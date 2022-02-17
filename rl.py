import argparse
from datetime import datetime
from collections import defaultdict, Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from env import make, register
import numpy as np

from plots import (globally_averaged_plot, advantages_plot,
        q_values_plot, display_policy, validation_plot, snapshot_plot)
from logs import snapshot_log
from utils import str2bool

from features import Features


from agents import get_agent
from agents import __all__ as AGENT_TYPES

parser = argparse.ArgumentParser(description='''
    This script trains an Actor-Critic DuoNavigation Gym Experiment.

    Usage:
    -----
    > python rl.py -s 2 -n 2 -e 2000 -m 100 -R 0 -A SARSATabular
    Where
''')

parser.add_argument('-A', '--agent_type', default='SARSATabular', type=str,
        choices=AGENT_TYPES, help='''A Reinforcement Learning Agent.''')

parser.add_argument('-a', '--alpha', default=0.5, type=float,
        help='''Alpha is the critic parameter:
                Only active if `decay` is False.''')

parser.add_argument('-b', '--beta', default=0.3, type=float,
        help='''Beta is the actor parameter:
                Reward attenuation for continuing tasks.''')

parser.add_argument('-z', '--zeta', default=0.1, type=float,
        help='''Zeta is the actor parameter:
                Actor learning rate for continuing tasks.''')

parser.add_argument('-d', '--decay', default=True, type=str2bool,
        help='''Exponential decay of actor and critic parameters:
                Replaces `alpha` and `beta` parameters.''')

parser.add_argument('-D', '--debug', default=True, type=str2bool,
        help='''Makes step_count accesses to hard to approximate functions, e.g, 
                V, Q and PI. Set to false for considerably less computing time.''')

parser.add_argument('-e', '--episodes', default=1, type=int,
        help='''Regulates the number of re-starts after the
                GOAL has been reached. Keep in mind that the
                task is continuing''')

parser.add_argument('-f', '--features', default='onehot', type=str,
        choices=['onehot', 'onehot+uniform', 'uniform'], 
        help='''Valid For agents with function approximation.''')

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

parser.add_argument('-R', '--render', default=False, type=str2bool,
        help='''Shows the grid during the training.''')

def print_arguments(opts, timestamp):

    print('Arguments Duo Navigation Game:')
    print(f'\tTimestamp: {timestamp}')
    for k, v in vars(opts).items():
        print(f'\t{k}: {v}')

def validate_arguments(opts):
    assert (opts.agent_type in ('SARSATabular', 'SARSASemiGradient') and opts.episodic) or \
        (opts.agent_type in ('SARSADifferentialSemiGradient', 'ActorCritic', 'ActorCriticTabular') and not opts.episodic)
    # or \ (opts.agent_type in ('CentralizedActorCritic', 'Optimal','FullyCentralizedActorCriticV1', 'FullyCentralizedActorCriticV2', 'SARSASemiGradient', 'TabularCentralizedActorCritic') and not opts.episodic)

def main(flags, timestamp):

    # Instanciate environment and agent
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
        kwargs={'flags': flags}
    )
    env = make('duo-navigation-v0').unwrapped

    features = Features().set(
        flags.features,
        n_agents=len(env.agents),
        width=env.width - 2,
        height=env.height - 2
    )
    agent = get_agent(env, flags) 

    # Loop control and execution flags.
    render = flags.render
    episodes = flags.episodes
    n_agents = flags.n_agents
    episodic = flags.episodic
    debug = flags.debug

    # Save parameters
    experiment_dir = Path('data') / timestamp
    experiment_dir.mkdir(exist_ok=True)
    config_path = experiment_dir / 'config.json'
    with config_path.open('w') as f:
        json.dump(vars(flags), f)
    
    log = defaultdict(list)
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        actions = agent.act(state)

        while True:
            if render:
                env.render(mode='human', highlight=True)
                time.sleep(0.1)
            next_state, next_reward, done, _ = env.step(actions)

            next_actions = agent.act(next_state)
            tr = [state, actions, next_reward, next_state, next_actions]
            if episodic: tr.append(done)

            agent.update(*tr)
            step_log = snapshot_log(episode, env, agent, tr, log, debug=debug)

            print(step_log)
            state = next_state 
            actions = next_actions
            if done:
                break

    agent.save_checkpoints(experiment_dir, str(episodes))
    snapshot_plot(log, experiment_dir)
    print(f'Experiment path:\t{experiment_dir.as_posix()}')
    print('Visited states', Counter(log['state']))

    df = display_policy(env, agent)
    df.to_csv((experiment_dir / 'policy.csv').as_posix(), sep='\t')
    

    validation_rewards = []
    state = env.reset()
    for _ in range(100):
        if render:
           env.render(mode='human', highlight=True)
           time.sleep(0.1)

        next_state, next_reward, done, _ = env.step(agent.act(state))
        validation_rewards.append(np.mean(next_reward))
        state = next_state

    validation_plot(validation_rewards)

if __name__ == '__main__':
    # Gather parameters.
    flags = parser.parse_args()

    # Notify the user what's going on.
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print_arguments(flags, timestamp)

    validate_arguments(flags)
    main(flags, timestamp)
