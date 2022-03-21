"""
    DuoNavigationGame: Team navigation game for Reinforcement Learning.

    TODO:
        - Exclude partial observability.
        - Verify if the other agent is already on the other goal,
        before placing an agent over a goal.
"""

import time
from itertools import product
from operator import itemgetter
from operator import attrgetter

import numpy as np
from cached_property import cached_property

import gym
from gym.envs.registration import register

from env.duogrid import Agent, Grid, Goal, MultiGridEnv, World
from env.duogrid import NavigationActions

from utils import action_set, pos2state, state2pos

N_PLAYERS = 2
N_GOALS = 2
REWARD = 0.1

# Converts a list of numpy.ndarrays into a list of tuples.
def tuplefy(x, attr=None):
    if attr is not None:
        x = map(lambda k: attrgetter(attr)(k), x)
    return map(tuple, map(lambda k: k.tolist(), x))


def tuples2set(x, attr=None):
    return {*tuplefy(x, attr=attr)}


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
        n_goals=N_GOALS,
    ):
        self.world = World
        self.random_starts = random_starts
        self.random_seed = seed
        self.episodic = episodic
        self.n_goals = n_goals
        self.episodes = 0

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
            partial_obs=False,
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=np.inf):
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

        if self.grid.get(*top) is None or (
            N_GOALS == 2 and self.grid.get(*top).can_overlap()
        ):
            pos = top
        else:
            num_tries = 0

            while True:
                # This is to handle with rare cases where rejection sampling
                # gets stuck in an infinite loop
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")

                num_tries += 1

                pos = np.array(
                    (
                        self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                        self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                    )
                )

                # Cell is free -- its okay
                if self.grid.get(*pos) is None:
                    break

                # Don't place the object on top of a wall
                if not self.grid.get(*pos).can_overlap():
                    continue

                # Don't place an agent over goal and other agent
                stacked_classes = [
                    type(stacked) for stacked in self.grid.get_stack(*pos)
                ]
                if (self.n_goals == 1) and (
                    Goal in stacked_classes and Agent in stacked_classes
                ):
                    continue

                # Prevents two goals from landing on the same spot
                if (self.n_goals == 2) and (
                    Goal in stacked_classes and isinstance(obj, Goal)
                ):
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
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        # Never overwrite goal on reset.
        if self.goal_pos is None:
            self.goal_pos = []
            for i in range(self.n_goals):
                goal = Goal(self.world, i, reward=REWARD, color=i + 1)
                self.goal_pos.append(self.place_obj(goal, top=None))
        else:
            for i, goal_pos in enumerate(self.goal_pos):
                goal = Goal(self.world, i, reward=REWARD, color=i + 1)
                self.place_obj(goal, top=goal_pos)

        # Randomize the player start position and orientation
        # for a in self.agents:
        #     self.place_agent(a)
        n_states = ((self.width - 2) * (self.height - 2)) ** N_PLAYERS

        def gn(x):
            return state2pos(
                x, n_players=N_PLAYERS, width=self.width - 2, height=self.height - 2
            )

        positions = map(gn, range(n_states))

        def fn(x):
            return not (
                np.array_equal(x, self.goal_pos)
                or np.array_equal(x, self.goal_pos[-1::-1])
            )

        initial_positions = [*filter(fn, positions)]
        initial_index = self.episodes % len(initial_positions)
        initial_position = initial_positions[initial_index]

        for pos, ag in zip(initial_position, self.agents):
            ag.pos = pos
            ag.init_pos = pos
            ag.dir = np.random.randint(0, 4)
            ag.init_dir = ag.dir
        np.testing.assert_array_equal(initial_position, [ag.pos for ag in self.agents])

    def _reward(self, rewards, goal_reward=1):
        """
        Compute the reward to be given upon success
        """
        for j, ag in enumerate(self.agents):
            rewards[j] = goal_reward if self.goal_reached else -1e-1

    @property
    def goal_reached(self):
        # evaluates if all agents have reached the goal.
        return tuples2set(self.goal_pos) == tuples2set(self.agents, "pos")

    def step(self, actions):
        self.step_count += 1

        for i, ag in enumerate(self.agents):

            # uses terminated as indicator that it has reached the goal.
            if ag.terminated or ag.paused or not ag.started:
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

            fwd_pos = ag.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*fwd_pos, ag)
                self.grid.rm(*ag.pos, ag)
                ag.pos = fwd_pos

        # stop conditions: Timeout or goal
        timeout = self.step_count >= self.max_steps

        # Agent earns the reward by reaching the goal
        # not by making the action that leads to goal.
        done = self.episodic and self.goal_reached

        rewards = -np.ones(len(self.agents)) * REWARD
        if self.goal_reached:
            rewards = -rewards
        return self.state, rewards, done, timeout

    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        if not self.random_starts:
            self.seed(seed=self.random_seed)
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
        self.episodes += 1

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
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = "_"

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        str = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                found = False
                for ag in self.agents:
                    if [i, j] == ag.pos.tolist():
                        str += 2 * AGENT_DIR_TO_STR[ag.dir]
                        found = True
                if found:
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += "  "
                    continue

                if c.type == "door":
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"

        return str

    # Provides a generator to pagine every state
    def next_states(self):
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
    def __init__(
        self,
        size=2,
        random_starts=True,
        seed=0,
        max_steps=20,
        episodic=True,
        n_goals=N_GOALS,
        **kwargs
    ):

        size += 2  # include border size
        agents_index = [i for i in range(1, N_PLAYERS + 1)]
        super(DuoNavigationGameEnv, self).__init__(
            size=size,
            agents_index=agents_index,
            random_starts=random_starts,
            seed=seed,
            max_steps=max_steps,
            episodic=episodic,
            n_goals=n_goals,
        )


def main():
    register(
        id="duo-navigation-v0",
        entry_point="env:DuoNavigationGameEnv",
    )
    env = gym.make("duo-navigation-v0").unwrapped

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode="human", highlight=True)
        time.sleep(0.1)

        actions = [env.action_space.sample() for _ in range(nb_agents)]

        next_state, next_reward, done, _ = env.step(actions)

        if done:
            break


if __name__ == "__main__":
    main()
