# TODO: add **kwargs to all agents and standartize interface
# Common interface for different variations of a model.
from agents.sg import ActorCriticSemiGradient
from agents.duo import ActorCriticSemiGradientDuo


def get_agent(env, flags):
    agent_cls = eval(flags.agent_type)
    if flags.agent_type in ("ActorCriticSemiGradient",):
        return agent_cls(
            env,
            alpha=flags.alpha,
            beta=flags.beta,
            episodes=flags.episodes,
            explore=flags.explore,
            decay=flags.decay,
        )

    elif flags.agent_type in ("ActorCriticSemiGradientDuo",):
        return agent_cls(
            env,
            alpha=flags.alpha,
            beta=flags.beta,
            episodes=flags.episodes,
            explore=flags.explore,
            decay=flags.decay,
            cooperative=flags.cooperative,
            partial_observability=flags.partial_observability,
        )
    else:
        raise ValueError(f"{flags.agent_type} is not supported.")


__all__ = ["ActorCriticSemiGradient", "ActorCriticSemiGradientDuo"]
