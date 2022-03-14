import json
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt

import utils


STATES_POSITIONS = [
    (0,       [(1, 1), (1, 1)]),
    (1,       [(1, 2), (1, 1)]),
    (2,       [(2, 1), (1, 1)]),
    (3,       [(2, 2), (1, 1)]),
    (4,       [(1, 1), (1, 2)]),
    (5,       [(1, 2), (1, 2)]),
    (6,       [(2, 1), (1, 2)]),
    (7,       [(2, 2), (1, 2)]),
    (8,       [(1, 1), (2, 1)]),
    (9,       [(1, 2), (2, 1)]),
    (10,      [(2, 1), (2, 1)]),
    (11,      [(2, 2), (2, 1)]),
    (12,      [(1, 1), (2, 2)]),
    (13,      [(1, 2), (2, 2)]),
    (14,      [(2, 1), (2, 2)]),
    (15,      [(2, 2), (2, 2)])
]

def sort(x):
    x = sorted(x, key=itemgetter(1))
    x = sorted(x, key=itemgetter(0))
    return x

def generate_transitions_from_moves(width=2, height=2):
    action_set = utils.action_set(2)

    # Get pos from state for this grid.
    def getpos(x):
        return utils.state2pos(x, n_agents=2, width=width, height=height)

    # Get state from positions
    def getstate(x):
        return utils.pos2state(x, n_agents=2, width=width, height=height)

    # Correct move on grid.
    def safemove(x):
        return np.clip(x, [1, 1], [width, height]) 

    moves = utils.MOVES
    tr = []
    for state in range(16):
        ag1_pos, ag2_pos = getpos(state)
        for ii, ag_moves in enumerate(action_set):
            ag1_move, ag2_move = ag_moves
            next_ag1_pos = safemove(ag1_pos + moves[ag1_move])
            next_ag2_pos = safemove(ag2_pos + moves[ag2_move])
            next_state = getstate([next_ag1_pos, next_ag2_pos])
            tr.append((state, ii, next_state))

    return tr


def config2fields(config_path):
    with config_path.open('r') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    print(df)


def main():
    '''Validation script'''
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


if __name__ == '__main__':
    from pathlib import Path
    from agents import ActorCriticSemiGradient
    from collections import defaultdict
    import features
    from features import get
    import pandas as pd
    n_agents = 2
    width, height = 2, 2
    goal_pos = np.array([2, 1])

    path = Path('data/AC-ONEHOT/01_baseline/')
    # path = Path('data/AC-ONEHOT/02_episodes_100000/') 
    # path = Path('data/AC-ONEHOT/04_decay/') 
    # path = Path('data/AC-ONEHOT/05_x1_r1/') 
    # path = Path('data/AC-ONEHOT/06_x1_r0/') 
    # path = Path('data/AC-ONEHOT/07_x0_r0/') 
    
    

    chkpt_num = [*path.rglob('*chkpt')][0].parent.stem
    agent = ActorCriticSemiGradient.load_checkpoint(path, chkpt_num)

    X = features.Features()
    X.set('onehot', n_agents=2, width=2, height=2)

    data = defaultdict(list) 
    for state, position in STATES_POSITIONS:
        if path is not None:
            data['state'].append(state)
            data['Coord 1'].append(position[0])
            data['Coord 2'].append(position[-1])
            data['V'].append(f'{agent.V[state]:0.2f}')
            for j, adv in enumerate(agent.A[state, :]):
                data[f'A(state, {j})'].append(f'{adv:0.2f}')
        df = pd.DataFrame.from_dict(data). \
                set_index('state')
        
    df.to_csv(path / 'advantage.csv')

    config_path = path / 'config.json'
    config2fields(config_path)


