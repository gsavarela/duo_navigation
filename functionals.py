'''Functional module to compute common variables.'''
import numpy as np

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
