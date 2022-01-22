"""DuoNavigationGame: Team navigation game for Reinforcement Learning

   MDP:
   ====
    * N = 2
    * |S| = (width * height - (N - 2)) ** |direction * N|
    * A[i] = {`left`, `right`, `forward`} --> A = A[0] x ... x A[N-1]
    * r = {r[i] = 1 if ag[i].terminated else 0}

"""
import time

import numpy as np
from numpy.random import uniform

import gym
from gym.envs.registration import register
from gym_multigrid.multigrid import *

from ac import ActorCritic

class State:
    '''Fully observable state

    Where agents cannot occupy the same tile.

    All states:
    |S| = (width * height - 1) ** N

    where:

        * width is the actual width (w/o boarders). 
        * height is the actual height (w/o boarders). 
        * N is the number o agents.
        * direction = {0, 1, 2, 3}

    '''
    def __init__(self, width, height, n_agents):
        assert n_agents ==  2
        self.n_agents = n_agents
        self.width = (width - 2)
        self.height = (height - 2)
        self.n_base = (self.width * self.height)
        self.n_states = self.n_base ** self.n_agents
        self.pow = [pow(self.n_base, i) for i in range(n_agents)]

    def get(self, positions): 
        return sum([self.lin(x) * y for x, y in zip(positions, self.pow)])

    def lin(self, pos):
        # env: width = 10, height = 10
        # tbl: width = 8, height = 8
        # | 00 | 01 | 02 | ... | 07 |
        # + -- + -- + -- + ... + -- +
        # | 08 | 09 | 10 | ... | 15 |
        # + -- + -- + -- + ... + -- +
        #           ...
        # | 56 | 57 | 58 | ... | 63 |
        rows, cols = (pos - 1) # position is not zero-based
        return rows * self.width + cols

class TeamActions:
    '''Joint actions for the team '''
    def __init__(self, n_actions, n_agents):
        self.n_actions = n_actions
        self.n_team_actions = n_actions ** n_agents
        self.pow = [n_actions ** i  for i in range(n_agents)]

    def get(self, actions):
        return sum([x * y for x, y in zip(actions, self.pow)])

class Features:
    def __init__(self, state, team_actions, n_critic=10, n_actor=5):
        self.state = state
        self.team_actions = team_actions

        self.n_phi = (state.n_states * team_actions.n_team_actions, n_critic)
        self.n_critic = n_critic
        self.phi = uniform(size=self.n_phi) 

        self.n_varphi = (state.n_states, team_actions.n_actions, state.n_agents, n_actor)
        self.n_actor = n_actor
        self.varphi = uniform(size=self.n_varphi) 

    def get_phi(self, state, actions): 
        k = self.team_actions.n_team_actions
        u = self.team_actions.get(actions)
        return self.phi[state * k + u, :]

    def get_varphi(self, state):
        val = self.varphi[state, ...]
        return val

class NavigationActions:
    available=['left', 'right', 'forward']
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

class DuoNavigationGameEnv(MultiGridEnv):
    """
    Environment in which both agents must reach a goal.
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[],
        num_goals=[],
        goals_index=[],
        goals_rewards=[],
        zero_sum = False,
        view_size=1
    ):
        self.num_goals  = num_goals
        self.goals_index = goals_index
        self.goals_reward = goals_rewards
        self.zero_sum = zero_sum
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        # RL stuff
        self.state = State(size, size, len(agents))
        self.team_actions = TeamActions(len(NavigationActions.available), len(agents))
        self.features = Features(self.state, self.team_actions)

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set=NavigationActions,
            partial_obs=False
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_goals, self.goals_index, self.goals_reward):
            for i in range(number):
                self.place_obj(Goal(self.world, index, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def step(self, actions):
        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        for i in order:

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started:
                if self.agents[i].terminated: self._reward(i, rewards, 1)
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        # done = True
                        self.agents[i].terminated = True
                        self._reward(i, rewards, 1)
                    elif fwd_cell.type == 'switch':
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        # Timeout
        if self.step_count >= self.max_steps:
            done = True

        # Victory
        if all([ag.terminated for ag in self.agents]):
            done = True
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            # obs = [self.grid.encode_for_agents(self.agents[i].pos) for i in range(len(actions))]
            positions = [ag.pos for ag in self.agents]
            s_t = self.state.get(positions)
        return s_t, rewards, done, {}

    
    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            # obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(self.agents))]
            #obs=[self.objects.normalize_obs*ob for ob in obs]
            positions = [ag.pos for ag in self.agents]
        return self.state.get(positions)

class DuoNavigationGameEnv(DuoNavigationGameEnv):

    def __init__(self):
        super(DuoNavigationGameEnv, self).__init__(
            size=7,
            agents_index=[1, 2],
            num_goals=[1],
            goals_index=[0],
            goals_rewards=[1],
            zero_sum=False
        )

def main():

    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
    )
    env = gym.make('duo-navigation-v0')

    state = env.reset()

    nb_agents = len(env.agents)
    
    if isrl:
        agent = ActorCritic(env) 
        actions = agent.act(state)
        

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        if not isrl:
            actions = [env.action_space.sample() for _ in range(nb_agents)]

        next_state, next_reward, done, _ = env.step(actions)

        if isrl:
            agent.update_mu(next_reward)
            next_actions = agent.act(next_state)
            agent.update(state, actions, next_reward, next_state, next_actions)

            state = next_state
            actions = next_actions

        if done:
            break

if __name__ == "__main__":
    main()
