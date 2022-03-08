'''DuoNavigationGame: Team navigation game for Reinforcement Learning.'''
import time
from itertools import product
from operator import itemgetter

import numpy as np
from numpy.random import uniform
from cached_property import cached_property

import gym
from gym.envs.registration import register

from env.duogrid import Agent, Grid, Goal, MultiGridEnv, World
from env.duogrid import NavigationActions

from utils import action_set, pos2state

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
        random_starts=True,
        max_steps=10000,
        seed=47,
        view_size=1,
        episodic=False,
    ):
        self.world = World
        self.random_starts = random_starts
        self.random_seed = seed
        self.episodic = episodic

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
        self.goal_pos = None

        

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
                  max_tries=np.inf
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
                if self.grid.get(*pos) is None:
                    break

                if isinstance(self.grid.get(*pos), Goal) or \
                        not self.grid.get(*pos).can_overlap():
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

        # Never overwrite goal on reset.
        self.goal_pos = self.place_obj(Goal(self.world, 0, 1), top=self.goal_pos)

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, rewards, goal_reward=1):
        """
        Compute the reward to be given upon success
        """
        for j, ag in enumerate(self.agents):
            rewards[j] = goal_reward if self.goal_reached else -1e-1

    @property
    def goal_reached(self):
        # evaluates if all agents have reached the goal.
        return all(tuple(ag.pos.tolist()) == self.goal_pos for ag in self.agents)

    def step(self, actions):
        self.step_count += 1

        # stop conditions: Timeout or goal
        timeout = (self.step_count >= self.max_steps)

        # Agent earns the reward by reaching the goal
        # not by making the action that leads to goal.
        done = (self.episodic and self.goal_reached)

        for i, ag in enumerate(self.agents):

            # uses terminated as indicator that it has reached the goal.
            if ag.terminated  or ag.paused or not ag.started:
                continue

            # Align according to current orientation
            if actions[i] == self.actions.right:
                # Face right
                ag.dir = 0 
            elif actions[i] == self.actions.down:
                # Face down
                ag.dir = 1
            elif actions[i] == self.actions.left:
                # Face left
                ag.dir = 2 
            else:
                # Face up
                ag.dir = 3

            # Get the position in front of the agent
            # def circular(x):
            #     if x[0] == 0: x[0] = self.width - 2
            #     x[0] =  max(x[0] % (self.width - 1), 1)
            #     if x[1] == 0: x[1] = self.height - 2
            #     x[1] =  max(x[1] % (self.height - 1), 1)
            #     return x
            # fwd_pos = circular(ag.front_pos)
            fwd_pos = ag.front_pos
            # print(ag.pos, ag.front_pos, fwd_pos)

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*fwd_pos, ag)
                self.grid.rm(*ag.pos, ag)
                ag.pos = fwd_pos


        rewards = np.ones(len(actions)) * -1e-1
        if self.goal_reached: rewards = -rewards
        return self.state, rewards, done, timeout

    
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
        return self.state

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                found = False
                for ag in self.agents:
                    if ([i, j] == ag.pos.tolist()):
                        str += 2 * AGENT_DIR_TO_STR[ag.dir]
                        found = True
                if found: continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    # Provides a generator to pagine every state
    def next_states(self):
        goal_pos = self.goal_pos
        rows, cols = [*range(1, self.height - 1)], [*range(1, self.width - 1)]
        agents_positions = [np.array([c, r]) for r in rows for c in cols]

        # Neat way to combine by number of agents.
        agents_positions = product(agents_positions, repeat=len(self.agents))

        # Map to positions to states.
        agents_positions = [(pos2state(pos), pos) for pos in agents_positions]
        # Order by states asc.
        agents_positions = sorted(agents_positions, key=itemgetter(0))

        for x, p in agents_positions: 
            yield x, p
        return 0

    @property
    def state(self):
        return pos2state(self.position)

    @cached_property
    def n_states(self):
        return ((self.width - 2) * (self.height - 2)) ** len(self.agents)

    @property
    def position(self):
        return [ag.pos for ag in self.agents]

    @cached_property
    def action_set(self):
        return action_set(len(self.agents))
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
        episodic = flags.episodic
        

        super(DuoNavigationGameEnv, self).__init__(
            size=size,
            agents_index=agents_index,
            random_starts=random_starts,
            seed=seed,
            max_steps=max_steps,
            episodic=episodic,
        )


def main():
    register(
        id='duo-navigation-v0',
        entry_point='env:DuoNavigationGameEnv',
    )
    env = gym.make('duo-navigation-v0').unwrapped

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
