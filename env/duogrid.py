'''DuoNavigationGame: Team navigation game for Reinforcement Learning.'''
import time
from itertools import product
from operator import itemgetter

import numpy as np
from numpy.random import uniform

import gym_multigrid.multigrid as mult
from gym_multigrid.multigrid import Goal, MultiGridEnv, World

# TODO: Deprecate State, TeamActions and Rewards.
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

        self.n_phi = (state.n_states, n_joint_actions, n_critic)
        self.n_critic = n_critic

        self.phi = uniform(size=self.n_phi) 
        self.phi_test = self.phi.copy().reshape((state.n_states * n_joint_actions, n_critic))
        # self.phi = self.phi / np.abs(self.phi).sum(keepdims=True, axis=-1)

        self.n_varphi = (state.n_states, team_actions.n_actions, state.n_agents, n_actor)
        self.n_actor = n_actor
        self.varphi = uniform(size=self.n_varphi) 
        # self.varphi = self.varphi / np.abs(self.varphi).sum(keepdims=True, axis=-1)

    def get_phi(self, state, actions): 
        k = self.team_actions.n_team_actions
        u = self.team_actions.get(actions)
        test = self.phi_test[state * k + u, :]
        res = self.phi[state][u][:]
        np.testing.assert_almost_equal(res, test)
        return res

    def get_varphi(self, state):
        val = self.varphi[state, ...]
        return val

class NavigationActions:
    available=['right', 'down', 'left', 'up']
    right = 0
    down = 1
    left = 2
    up = 3

# Agents may overlap.
class Agent(mult.Agent):
    def can_overlap(self):
        return True

# Extends original grid
class Grid(mult.Grid):
    """
    Represent a grid and operations on it
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.grid = [None] * width * height
        self.stack = [[] for _ in range(width * height)] 

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        # backwards compatibility
        u = self.grid[j * self.width + i]
        if v is None and u is not None:
            self.stack[j * self.width + i].remove(u)
        self.grid[j * self.width + i] = v
        if v is not None:
            self.stack[j * self.width + i].append(v)

    def rm(self, i, j, v, not_exist_ok=True):
        # Removes v from tile (j, i) leaves everything else
        # the same.
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        try:
            self.stack[j * self.width + i].remove(v)
        except ValueError as exc:
            if not not_exist_ok: raise exc

        if self.grid[j * self.width + i] is v: 
            self.grid[j * self.width + i] = None

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = StackableGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:

            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(agent_pos=(a.view_size // 2, a.view_size - 1))
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = StackableGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid


if __name__ == "__main__":
    main()
