import numpy as np
import pandas as pd

def pretty_print_df(df):
    # Pandas display options.
    with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', 1000,
                            'display.colheader_justify', 'center',
                            'display.precision', 3):
        print(df)

def act2str(act):
    if act == 0: return '>'
    if act == 1: return 'V'
    if act == 2: return '<'
    if act == 3: return '^'
    raise KeyError(f'{act} is not a valid action')

def act2str2(acts):
    moves = ', '.join([act2str(act) for act in acts])
    return f'({moves})'

def pi2str(probs):
    strprobs = [f'{prob:0.2f}' for prob in probs]
    strprobs = ', '.join(strprobs)
    return f'({strprobs})'

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
            disact = act2str2(actions[np.argmax(scores)])

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
                positions = [np.array([j, i]), np.array([i, j])] 
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
