# FIXME: I wonder if we could use GYM register here.
# Common interface for different variations of a model.
from agents.sarsa import SARSATabular, SARSASemiGradient
from agents.centralized import CentralizedActorCritic

def get_agent(env, flags):
    agent_cls = eval(flags.agent_type)
    if flags.agent_type == 'CentralizedActorCritic':
        return agent_cls(env, alpha=flags.alpha, beta=flags.beta, decay=flags.decay) 
    if flags.agent_type == 'SARSATabular':
        return agent_cls(env, alpha=flags.alpha, episodes=flags.episodes) 
    if flags.agent_type == 'SARSASemiGradient':
        return agent_cls(env, alpha=flags.alpha, beta=flags.beta, episodes=flags.episodes)
__all__ = ['CentralizedActorCritic', 'SARSATabular', 'SARSASemiGradient']
