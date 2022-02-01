from inspect import getargspec

from utils import i2q, q2i

# Use this to convert a joint action into a list of actions.
def int2act(act):
    def wrapper(*args, **kwargs):
        n_agents = args[0].n_agents
        return i2q(act(*args, **kwargs), n_agents)
    return wrapper

# Use this to convert a list of actions into a joint action.
def act2int(update):
    argspecs = getargspec(update).args
    actions_index = argspecs.index('actions') 
    next_actions_index = argspecs.index('next_actions') 
    def wrapper(*args, **kwargs):
        # modify arguments before passing
        args = list(args) 
        args[actions_index] = q2i(args[actions_index])
        args[next_actions_index] = q2i(args[next_actions_index])
        return update(*args, **kwargs)
    return wrapper

