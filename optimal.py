'''The optimal agent always moves according to the best action.'''
from numpy.random import choice
from centralized import CentralizedActorCritic
from utils import best_actions

class OptimalAgent(CentralizedActorCritic):
    def __init__(self, *args, **kwargs):
        super(OptimalAgent, self).__init__(*args, **kwargs)
        # Optimal Knows where the goal is and selects the best action
        self.goal_pos = self.env.goal_pos

    def act(self, state):
        return [choice(best_actions(ag.pos, self.goal_pos))
                for ag in self.env.agents]
