# FIXME: I wonder if we could use GYM register here.
# Common interface for different variations of a model.
from agents.sarsa import SARSATabular, SARSASemiGradient, SARSADifferentialSemiGradient
from agents.actor_critic import ActorCritic, ActorCriticTabular
# from agents.centralized import (CentralizedActorCritic,
#         FullyCentralizedActorCriticV1, FullyCentralizedActorCriticV2,
#         TabularCentralizedActorCritic)
from agents.optimal import Optimal

def get_agent(env, flags):
    agent_cls = eval(flags.agent_type)
    # if flags.agent_type in ('CentralizedActorCritic', 'FullyCentralizedActorCriticV1',
    #         'FullyCentralizedActorCriticV2', 'TabularCentralizedActorCritic'):
    #     return agent_cls(env, alpha=flags.alpha, beta=flags.beta, decay=flags.decay) 
    if flags.agent_type in ('SARSATabular', 'SARSASemiGradient'):
        return agent_cls(env, alpha=flags.alpha, episodes=flags.episodes) 
    if flags.agent_type == 'SARSADifferentialSemiGradient':
        return agent_cls(env, alpha=flags.alpha, beta=flags.beta, episodes=flags.episodes)
    if flags.agent_type in ('ActorCritic', 'ActorCriticTabular'):
        return agent_cls(env, alpha=flags.alpha, beta=flags.beta, zeta=flags.zeta, episodes=flags.episodes, explore=flags.explore)
    # if flags.agent_type == 'SARSASemiGradient':
    #     return agent_cls(env, alpha=flags.alpha, beta=flags.beta, episodes=flags.episodes)
    if flags.agent_type == 'Optimal':
        return agent_cls(env, alpha=flags.alpha, decay=flags.decay)
__all__ = ['Optimal', 'ActorCritic', 'ActorCriticTabular', 'SARSATabular', 'SARSASemiGradient','SARSADifferentialSemiGradient']
# __all__ = ['CentralizedActorCritic', 'FullyCentralizedActorCriticV1', 'FullyCentralizedActorCriticV2', 'Optimal', 'SARSATabular', 'SARSASemiGradient', 'TabularCentralizedActorCritic']

