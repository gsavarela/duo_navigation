from agents.sarsa import SARSATabular
from agents.centralized import CentralizedActorCritic

def get_agent(env, flags):
    agent_cls = eval(flags.agent_type)
    if flags.agent_type == 'CentralizedActorCritic':
        return agent_cls(env, alpha=flags.alpha, beta=flags.beta, decay=flags.decay) 
    if flags.agent_type == 'SARSATabular':
        return agent_cls(env, alpha=flags.alpha, max_episodes=flags.episodes) 

__all__ = ['CentralizedActorCritic', 'SARSATabular']
