import argparse
from datetime import datetime
from collections import defaultdict, Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from env import make, register
import numpy as np

from plots import display_policy, snapshot_plot
from logs import snapshot_log, snapshot_state_actions_log
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
                Actor learning rate. ''')

parser.add_argument('-c', '--cooperative', default=True, type=str2bool,
        help='''Reward type:
                if True reward is the same for both players.
                if False each player earns it\'s own reward''')

parser.add_argument('-z', '--zeta', default=0.1, type=float,
        help='''Reward attenuation for continuing tasks.''')

parser.add_argument('-d', '--decay', default=False, type=str2bool,
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
        help='''Regulates hte initial position of agents on the grid:
                Should be either `0` or `1`. Use `0` for debugging.''')

parser.add_argument('-R', '--render', default=False, type=str2bool,
        help='''Shows the grid during the training.''')

parser.add_argument('-x', '--explore', default=True, type=str2bool,
        help='''Use temperature for continuing tasks or epsilon-greedy.''')

def print_arguments(opts, timestamp):

    print('Arguments Duo Navigation Game:')
    print(f'\tTimestamp: {timestamp}')
    for k, v in vars(opts).items():
        print(f'\t{k}: {v}')

def validate_arguments(opts):
    assert (opts.agent_type in ('SARSATabular', 'SARSASemiGradient', 'ActorCriticSemiGradient', 'ActorCriticSemiGradientDuo') and opts.episodic) or \
        (opts.agent_type in ('SARSADifferentialSemiGradient', 'ActorCriticDifferentialSemiGradient', 'ActorCriticTabular') and not opts.episodic)
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
        agent.reset(seed=episode)
        actions = agent.act(state)

        while True:
            if render:
                env.render(mode='human', highlight=True)
                time.sleep(0.1)
            next_state, next_reward, done, timeout = env.step(actions)
            
            # Uncomment to test individual reward structure.:
            # if next_state in (0, 1, 3, 4, 5, 7, 12, 13, 15):
            #     test_reward = np.array([-0.1, -0.1])
            # elif next_state in (2, 6, 14):
            #     test_reward = np.array([0.1, -0.1])
            # elif next_state in (8, 9, 11):
            #     test_reward = np.array([-0.1, 0.1])
            # else:
            #     test_reward = np.array([0.1, 0.1])
            # np.testing.assert_array_equal(next_reward, test_reward)
            next_actions = agent.act(next_state)
            tr = [state, actions, next_reward, next_state, next_actions]
            if episodic: tr.append(done)

            agent.update(*tr)
            step_log = snapshot_log(episode, env, agent, tr, log, debug=debug)

            print(step_log)
            state = next_state 
            actions = next_actions
            if done or timeout:
                break

    agent.save_checkpoints(experiment_dir, str(episodes))

    snapshot_plot(log, experiment_dir)
    print(f'Experiment path:\t{experiment_dir.as_posix()}')

    state_counter = Counter(log['state']) 
    print('Visited states', state_counter)
    df = snapshot_state_actions_log(log, experiment_dir)
    print(df)
    

    df = display_policy(env, agent)

    # Make two files -- policy and advantages
    # policy.csv
    def fn(x): return 'A(state,' not in x
    policy_path = experiment_dir / 'policy.csv'
    policy_columns = [*filter(fn, df.columns.tolist())]
    df[policy_columns].to_csv(policy_path)

    def gn(x): return 'PI(state,' not in x
    advantage_path = experiment_dir / 'advantage.csv'
    advantages_columns = [*filter(gn, df.columns.tolist())]
    df[advantages_columns].to_csv(advantage_path)

    with (experiment_dir / 'snapshot.json').open('w') as f:
        json.dump(log, f)
    
    
if __name__ == '__main__':
    # Gather parameters.
    flags = parser.parse_args()

    # Notify the user what's going on.
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print_arguments(flags, timestamp)

    validate_arguments(flags)
    main(flags, timestamp)
