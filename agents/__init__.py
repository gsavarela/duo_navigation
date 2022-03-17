from agents.central import ActorCriticCentral
from agents.jals import ActorCriticJALs
from agents.ils import ActorCriticILs


def get_agent(env, flags):
    # agent_cls = eval(flags.agent_type)
    if flags.agent_type in ("Central", "JALs", "ILs"):
        agent_cls = eval(f'ActorCritic{flags.agent_type}')
        return agent_cls(
            env,
            alpha=flags.alpha,
            beta=flags.beta,
            episodes=flags.episodes,
            explore=flags.explore,
            decay=flags.decay,
        )

    else:
        raise ValueError(f"{flags.agent_type} is not supported.")

AGENT_TYPES = ['Central', "JALs", "ILs"]
__all__ = ["ActorCriticCentral", "ActorCriticJALs", "ActorCriticILs"]
