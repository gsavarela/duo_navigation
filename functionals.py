'''Functional module to compute common variables.'''
import numpy as np

def pretty_print_df(df):
    # Pandas display options.
    with pd.option_context('display.max_rows', None,
                            'display.max_columns', None,
                            'display.width', 1000,
                            'display.colheader_justify', 'center',
                            'display.precision', 3):
        print(df)

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
def calculate_phi(env):
    def fn_phi(x, y):
        return env.features.get_phi(x, y)
    return fn_phi
    
def calculate_pi(env, agent):
    def fn_pi(x, z=0):
        return agent.pi(env.features.get_varphi(x), z)
    return fn_pi

def calculate_q_function(env, agent):
    n_actions = agent.n_actions
    phi = calculate_phi(env)
    def fn_q(x, y=None):
        if y is None:
           return np.array([fn_q(x, [u]) for u in range(n_actions)])
        return phi(x, y) @ agent.omega
    return fn_q

def calculate_v_function(env, agent):
    q = calculate_q_function(env, agent)
    pi = calculate_pi(env, agent)
    def fn_v(x, z=0):
        return pi(x, z) @ q(x)
    return fn_v

def calculate_advantage(env, agent):
    q = calculate_q_function(env, agent)
    v = calculate_v_function(env, agent)
    def fn_a(x, y=None, z=0):
        return q(x, y) - v(x)
    return fn_a

def calculate_transitions(env, agent):
    # Logs the following quantities for (state, action)
    rows, cols = range(1, env.height - 1),  range(1, env.width - 1)
    gs = list(env.goal_pos)
    # Coordinate axis to states.
    def p2s(x, y): return env.state.lin(np.array([y, x])) 
    # Individual actions to joint actions.
    def i2j(u): return env.team_actions.get(u)

    # Those are visitable states.
    if agent.n_agents == 1:
        states = [p2s(r, c) for r in rows for c in cols if [c, r] != gs]
    else:
        positions = [np.array([c, r])
                for r in rows for c in cols if [c, r] != gs]
        states = [env.state.get([p1, p2]) for p2 in positions for p1 in positions]
    # All possible actions.
    n_joint_actions = env.team_actions.n_team_actions
    joint_actions = [*range(n_joint_actions)]
    n_agents = agent.n_agents 
    individual_actions = [*range(agent.n_actions)]

    q = calculate_q_function(env, agent)
    v = calculate_v_function(env, agent)
    a = calculate_advantage(env, agent)
    pi = calculate_pi(env, agent)
    def fn_tr(tr, trdict, updated=False):
        q_key = 'Q'
        v_key = 'V'
        a_key = 'A'
        pi_key = 'pi'
        if not updated: 
            trdict['timestep'].append(env.step_count)
            trdict['state'].append(tr[0]) # state
            trdict['actions'].append(i2j(tr[1])) # actions
            trdict['next_rewards'].append(np.mean(tr[2])) # reward
            trdict['next_state'].append(tr[3]) # next_state
            trdict['next_actions'].append(tr[4][0]) # next_actions
        else:
            q_key += "'"
            v_key += "'"
            a_key += "'"
            pi_key += "'"

        # v-function, q-function before the update.
        for ag in range(n_agents):
            for st in states:
                key = f'{v_key}_{ag}({st})'
                trdict[key].append(v(st)) # next_actions
                for ac in joint_actions:
                    key = f'{q_key}_{ag}({st}, {ac})'
                    trdict[key].append(q(st, [ac])) # next_actions

                    key = f'{a_key}_{ag}({st}, {ac})'
                    trdict[key].append(a(st, [ac])) # next_actions

                for ac in individual_actions:
                    key = f'{pi_key}_{ag}({st}, {ac})'
                    trdict[key].append(pi(st, ag)[ac]) # next_actions

    return fn_tr
