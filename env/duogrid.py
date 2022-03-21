'''DuoNavigationGame: Team navigation game for Reinforcement Learning.'''
import time
from itertools import product
from operator import itemgetter

import numpy as np
from numpy.random import uniform

import gym_multigrid.multigrid as mult
from gym_multigrid.multigrid import Goal, MultiGridEnv, World

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


    def get_stack(self, i, j):
        # checks if v is in (i, j).
        assert j >= 0 and j < self.height
        assert i >= 0 and i < self.width
        return self.stack[j * self.width + i]

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

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

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid


if __name__ == "__main__":
    main()
