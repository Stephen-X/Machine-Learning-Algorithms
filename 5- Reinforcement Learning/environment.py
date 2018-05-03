#!/usr/bin/env python3
"""
Project 8: Maze Solver with Reinforcement Learning
Author: Stephen Xie <***@andrew.cmu.edu>

Maze environment simulation for an agent, used in Q-learning. It keeps track of
the current state of the agent, and it supports two methods that allow users to
give action order to the agent or reset agent's state.

Data Assumptions:
1. Maze data are rectangular (i.e. all rows have the same number of characters)
   and valid (only contains 'S', 'Q', '.', '*'; etc.).

Notes:
1. When the agent reaches the goal/terminal state, it'll stay there.
2. Whenever the agent performs an action (except when it stays in the
   goal/terminal state), it receives an immediate reward of -1.
3. When the agent takes an action that'll hit the border or an obstacle, it'll
   return to the original state, and it'll still need to pay -1 immediate reward.
"""

import sys
import numpy as np


class Environment:
    def __init__(self, maze_input, output_file=None, action_seq_file=None):
        # Read in maze data
        self.maze_data, self.start, self.goal = self._load_maze(maze_input)
        # print('maze_data:\n', self.maze_data)

        # Initialize agent's current state to the start state
        self.current = self.start

        results = None
        if action_seq_file is not None:
            # The agent needs to take a few pre-specified actions first
            actions = np.genfromtxt(action_seq_file, dtype=np.int,
                                    delimiter=' ', autostrip=True)
            # print('actions:', actions)
            results = [self.step(a) for a in actions]
        # print('results:', results)

        if output_file is not None and results is not None:
            # Output feedback
            with open(output_file, mode='w') as f:
                for next_state, reward, is_terminal in results:
                    f.write('%d %d %d %d\n' % (next_state[0], next_state[1],
                                               reward, is_terminal))

    def _load_maze(self, maze_input):
        """
        Load maze in memory.

        Returns maze data as numpy matrix, and the coordinate tuples of the
        start/initial state and the goal/terminal state.
        """
        maze_data = []
        with open(maze_input, mode='r') as f:
            for line in f:
                line = line.strip()
                # Shatter row string into list of chars
                maze_data.append(list(line))
        maze_data = np.asarray(maze_data)

        # Get coordinates to start state and goal state as (x, y) tuples
        start = tuple(np.argwhere(maze_data == 'S')[0])
        goal = tuple(np.argwhere(maze_data == 'G')[0])

        return maze_data, start, goal

    def step(self, action):
        """
        Simulate a step according to the action input and set the current state
        to the next state.

        Returns next_state (coordinates tuple of the agent after taking action),
        reward, is_terminal (1 if reached goal state, 0 otherwise)
        """
        if self.current == self.goal:
            # Agent already reached goal state
            return self.current, 0, 1
        else:
            x, y = self.current  # current coordinate
            if action == 0:  # attempt to move west
                if y > 0 and self.maze_data[x, y-1] != '*':
                    self.current = (x, y - 1)
            elif action == 1:  # attempt to move north
                if x > 0 and self.maze_data[x-1, y] != '*':
                    self.current = (x - 1, y)
            elif action == 2:  # attempt to move east
                if y < self.maze_data.shape[1] - 1 and self.maze_data[x, y+1] != '*':
                    self.current = (x, y + 1)
            elif action == 3:  # attempt to move south
                if x < self.maze_data.shape[0] - 1 and self.maze_data[x+1, y] != '*':
                    self.current = (x + 1, y)
            else:
                raise ValueError('Parameter "action" must be integer 0, 1, 2, or 3.')
            reward = -1
            is_terminal = 1 if self.current == self.goal else 0

            return self.current, reward, is_terminal

    def reset(self):
        """
        Reset the agent state to the initial state.

        Returns the initial state.
        """
        self.current = self.start

        return self.current

    def get_total_states(self):
        """
        Get total number of states in the maze.
        """
        return self.maze_data.size

    def get_maze_dimension(self):
        """
        Get the dimension of the maze as a tuple.
        """
        return self.maze_data.shape

    def get_obstacles(self):
        """
        Get a set of coordinate tuples for all obstacles in the maze.
        Note: This is used only for result output. Don't use it during
              q-learning process.
        """
        return set([tuple(coord) for coord in np.argwhere(self.maze_data == '*')])


if __name__ == '__main__':
    maze_input = sys.argv[1]
    output_file = sys.argv[2]
    action_seq_file = sys.argv[3]

    model = Environment(maze_input, output_file, action_seq_file)

    # model = Environment('medium_maze.txt', 'output.feedback',
    #                     'medium_maze_action_seq.txt')
