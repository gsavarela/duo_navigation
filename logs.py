from collections import defaultdict
import re
import numpy as np
import pandas as pd
from utils import act2str, acts2str, pi2str, best_actions
from functionals import calculate_advantage as advantage_functional
from functionals import calculate_q_function as q_functional
from functionals import calculate_transitions as transitions_functional

def pretty_print_df(df):
    # Pandas display options.
    with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', 1000,
                            'display.colheader_justify', 'center',
                            'display.precision', 3):
        print(df)

def display_critic(env, agent):
    '''Displays Q-value'''
    actions_scores = {}
    actions_display = {}
    actions = [(v, u) for v in range(3) for u in range(3)]

    # rows (coordinate y) 
    for i in range(1, env.height - 1): 
        actions_scores[i] = []
        actions_display[i] = []
        # columns (coordinate x) 
        for j in range(1, env.width - 1): 
            positions = [np.array([j, i]), np.array([j, i])] 
            state = env.state.get(positions)
            scores = [env.features.get_phi(state, ac) @ agent.omega for ac in actions]
            disact = acts2str(actions[np.argmax(scores)])

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
    actions_display = {}
    n_agents = len(env.agents)

    for k in range(n_agents):

        pi_display[k] = {}
        actions_display[k] = {}
        for i in range(1, env.width - 1): 
            pi_display[k][i] = []
            actions_display[k][i] = []
            for j in range(1, env.height - 1): 
                positions = [np.array([i, j]), np.array([i, j])] 
                state = env.state.get(positions)
                varphi = env.features.get_varphi(state)
                pi_k = agent.pi(varphi, k)
                pi_display[k][i].append(pi2str(pi_k))
                actions_display[k][i].append(act2str(np.argmax(pi_k)))

    pis_dataframes = []
    for k, pidis in pi_display.items():
        pis_dataframes.append(pd.DataFrame.from_dict(pidis))

    actions_dataframes = []
    for k, actdis in actions_display.items():
        actions_dataframes.append(pd.DataFrame.from_dict(actdis))

    margin = '#'*33 
    for k in range(n_agents):
        print(f'{margin}\tAGENT {k + 1}\t{margin}')
        pretty_print_df(pis_dataframes[k])
        pretty_print_df(actions_dataframes[k])

    return pis_dataframes, actions_dataframes

def logger(env, agent, state_actions):
    advfn = advantage_functional(env, agent)
    advlogger = advantage_log(advfn, state_actions)

    qfn = q_functional(env, agent)
    qlogger = q_log(qfn, state_actions)

    tr_logger = transitions_functional(env, agent)

    def log(advantages, q_values, tr, tr_dict, updated):
        if not updated:
            advlogger(advantages)
            qlogger(q_values)
        tr_logger(tr, tr_dict, updated)
    return log

def advantage_log(fn_a, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    def fn_log(advantages):
        res = []
        for x, ys in state_actions:
            advs = fn_a(x)
            res += [(x, [advs[y] for y in ys])]
        advantages.append(res)
    return fn_log

def q_log(fn_q, state_actions=[(0, [0]),(1, [0, 3]),(3, [3])]):
    def fn_log(q_values):
        res = []
        for x, ys in state_actions:
            res += [(x, [fn_q(x, [y]) for y in ys])]
        q_values.append(res)
    return fn_log

def transition_update_log(tr_dict, experiment_dir):
    # Convert single dataframe into multiple
    df = pd.DataFrame.from_dict(tr_dict). \
            set_index('timestep', inplace=False)

    columns = df.columns.values.tolist()
    # sort key for (state, action) 
    def srt(x): return eval(re.search(r'\((.*?)\)',x).group(1))

    # base transitions.
    trdf = df[['state', 'actions', 'next_rewards', 'next_state', 'next_actions']]

    # q-functions, a-functions, policies
    for k in ('Q', 'A', 'pi', 'V'):
        cix = [k in col for col in columns]
        kdf = df.loc[:, cix]

        kcols = sorted(kdf.columns.values.tolist(), key=srt)
        kdf = kdf.loc[:, kcols]

        kdf = pd.concat((trdf, kdf), axis=1)
        kdf.to_csv(experiment_dir / f'{k}.csv')

    return df

def best_actions_log(env, exp_dir=None):
    """Paginates the positions on the environment."""
    rows, cols = range(1, env.height - 1), range(1, env.width - 1)

    data = defaultdict(list)
    for c in cols:
        for r in rows:
            agent_pos = np.array([c, r])
            state = env.state.lin(agent_pos)
            if not np.array_equal(agent_pos, env.goal_pos):

                data['state'].append(state)
                data['pos'].append(agent_pos)
                bas = best_actions(agent_pos, env.goal_pos)
                data['actions'].append(bas)
                data['moves'].append(acts2str(bas))

    df = pd.DataFrame.from_dict(data). \
             set_index('state', inplace=False)

    if exp_dir is not None: df.to_csv(exp_dir / 'best_actions.csv')
    return [*df['actions'].items()]

                

            

