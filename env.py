'''DuoNavigationGame: Team navigation game for Reinforcement Learning.'''
import time

import numpy as np
from numpy.random import uniform

import gym
from gym.envs.registration import register
from gym_multigrid.multigrid import *

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
        # assert n_agents ==  2
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
        # tbl: width = 1...8, height = 1..8
        # | 00 | 01 | 02 | ... | 07 |
        # + -- + -- + -- + ... + -- +
        # | 08 | 09 | 10 | ... | 15 |
        # + -- + -- + -- + ... + -- +
        #           ...
        # | 56 | 57 | 58 | ... | 63 |
        # position is not zero-based
        rows, cols = (pos - 1) 
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
        np.random.seed(0)
        self.state = state
        self.team_actions = team_actions
        n_joint_actions = team_actions.n_team_actions

        self.n_phi = (state.n_states * n_joint_actions, n_critic)
        self.n_critic = n_critic
        self.phi = uniform(low=-0.5, high=0.5, size=self.n_phi) 
        self.phi = self.phi / np.linalg.norm(self.phi)

        self.n_varphi = (state.n_states, team_actions.n_actions, state.n_agents, n_actor)
        self.n_actor = n_actor
        self.varphi = uniform(low=-0.5, high=0.5, size=self.n_varphi) 
        self.varphi = self.varphi / np.linalg.norm(self.phi) 

    def get_phi(self, state, actions): 
        k = self.team_actions.n_team_actions
        u = self.team_actions.get(actions)
        return self.phi[state * k + u, :]

    def get_varphi(self, state):
        val = self.varphi[state, ...]
        return val

class NavigationActions:
    available=['right', 'down', 'left', 'up']
    right = 0
    down = 1
    left = 2
    up = 3

class DuoNavigationEnv(MultiGridEnv):
    """
    Environment in which both agents must reach a goal.
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        agents_index=[],
        num_goals=[1],
        goals_index=[0],
        goals_rewards=[1],
        random_starts=True,
        max_steps=10000,
        seed=47,
        zero_sum = False,
        view_size=1
    ):
        self.num_goals  = num_goals
        self.goals_index = goals_index
        self.goals_reward = goals_rewards
        self.zero_sum = zero_sum
        self.world = World
        self.random_starts = random_starts
        self.random_seed = seed

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
        self.goals_pos = [None] * len(num_goals)

        # RL stuff
        self.state = State(size, size, len(agents))
        self.team_actions = TeamActions(len(NavigationActions.available), len(agents))
        self.features = Features(self.state, self.team_actions)

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set=NavigationActions,
            partial_obs=False
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        if self.grid.get(*top) is None:
            pos = top
        else:
            num_tries = 0

            while True:
                # This is to handle with rare cases where rejection sampling
                # gets stuck in an infinite loop
                if num_tries > max_tries:
                    raise RecursionError('rejection sampling failed in place_obj')

                num_tries += 1

                pos = np.array((
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
                ))

                # Don't place the object on top of another object
                if self.grid.get(*pos) != None:
                    continue

                # Check if there is a filtering criterion
                if reject_fn and reject_fn(self, pos):
                    continue

                break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Never overwrite goal.
        g = zip(self.num_goals, self.goals_index, self.goals_reward, self.goals_pos)
        for number, index, reward, pos in g:
            for i in range(number):
                goal_pos = self.place_obj(Goal(self.world, index, reward), top=pos)

                if pos is None: self.goals_pos[i] = goal_pos

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
        goals = [False for _ in range(len(self.agents))]

        for i in order:

            # uses terminated as indicator that it has reached the goal.
            if self.agents[i].terminated  or self.agents[i].paused or not self.agents[i].started:
                continue

            # Align according to current orientation
            if actions[i] == self.actions.right:
                # Face right
                self.agents[i].dir = 0 
            elif actions[i] == self.actions.down:
                # Face down
                self.agents[i].dir = 1
            elif actions[i] == self.actions.left:
                # Face left
                self.agents[i].dir = 2 
            else:
                # Face up
                self.agents[i].dir = 3

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            # print(actions[0], self.agents[i].pos, fwd_pos, 'free' if fwd_cell is None else fwd_cell.type)

            if fwd_cell is not None:
                if fwd_cell.type == 'goal':
                    # done = True
                    goals[i] = True
                elif fwd_cell.type == 'switch':
                    self._handle_switch(i, rewards, fwd_pos, fwd_cell)
            elif fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*fwd_pos, self.agents[i])
                self.grid.set(*self.agents[i].pos, None)
                self.agents[i].pos = fwd_pos
            self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

        # Timeout
        if self.step_count >= self.max_steps:
            done = True

        # Victory
        if all(goals):
            # done = True
            rewards = np.ones(len(self.agents))
        else:
            rewards = np.zeros(len(self.agents))

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            positions = [ag.pos for ag in self.agents]
            s_t = self.state.get(positions)
        return s_t, rewards, done, {}

    
    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        if not self.random_starts: self.seed(seed=self.random_seed)
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None
            a.terminated = False

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            positions = [ag.pos for ag in self.agents]
        return self.state.get(positions)

class DuoNavigationGameEnv(DuoNavigationEnv):

    def __init__(self, **kwargs):
        flags = kwargs['flags']

        # Gather enviroment variables
        size = flags.size + 2
        n_agents = flags.n_agents
        random_starts = flags.random_starts 
        seed = flags.seed
        agents_index = [i for i in range(1, n_agents + 1)]
        max_steps = flags.max_steps
        

        super(DuoNavigationGameEnv, self).__init__(
            size=size,
            agents_index=agents_index,
            random_starts=random_starts,
            seed=seed,
            max_steps=flags.max_steps
        )

def main():
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
    )
    env = gym.make('duo-navigation-v0')

    _ = env.reset()

    nb_agents = len(env.agents)
    
    while True:
       env.render(mode='human', highlight=True)
       time.sleep(0.1)

       actions = [env.action_space.sample() for _ in range(nb_agents)]

       next_state, next_reward, done, _ = env.step(actions)

       if done:
           break

if __name__ == "__main__":
    main()
