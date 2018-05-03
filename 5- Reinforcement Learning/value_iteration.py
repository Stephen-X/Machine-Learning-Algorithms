#!/usr/bin/env python3
"""
Project 8: Maze Solver with Reinforcement Learning
Author: Stephen Xie <***@andrew.cmu.edu>

Learn policies to walk through a maze by reinforcement learning. This program
implements value iteration with synchronous updates.

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
import warnings
from itertools import product

# for debugging: print the full numpy array
np.set_printoptions(threshold=np.inf, linewidth=100)


class ValueIteration:

    def __init__(self, maze_input, value_file, q_value_file, policy_file,
                 num_epoch, discount_factor):
        values, next_states, goal_state_ind, self.maze_dimension = self.load(maze_input)
        learned_values, learned_q_values, learned_policies = \
            self.learn(values, next_states, goal_state_ind, discount_factor,
                       num_epoch)

        # Output results
        self.output_values(learned_values, value_file)
        self.output_q_values(learned_q_values, q_value_file)
        self.output_policies(learned_policies, policy_file)

    def load(self, maze_input):
        """
        Read in and parse maze information.

        Returns the value function V(s), value function indices of next states
        that correspond to transition probabilities P(s'|s,a), goal state index
        in V(s), and maze dimension as (length, width) tuple.
        """
        # Load maze map into memory
        maze_data = []
        with open(maze_input, mode='r') as f:
            for line in f:
                line = line.strip()
                # Shatter row string into list of chars
                maze_data.append(list(line))
        maze_data = np.asarray(maze_data)
        # print('maze:\n', maze_data)

        # Initialize parameters according to maze map

        # Value function V(s)
        values = np.zeros(maze_data.size + 1)
        values[0] = np.nan  # Here we adds a NaN value for any invalid states

        # Index of the next state in values if the agent takes action a when in
        # state a. Invalid next states (i.e. out of bond or obstacles) are
        # pointed to values[0] which is NaN. Matrix indices correspond to
        # positions in transition probabilities, which is a num_of_states x num_of_actions matrix.
        # Notes:
        # 1. Because transition probabilities are either 1 or 0 (actions are
        # deterministic), here we skip transition probabilities modeling
        # 2. To line up with values (for convenience in the learn function),
        # here we add a nan row at front
        next_states = np.zeros((maze_data.size + 1, 4), dtype=np.int)
        state_ind = 0  # current state's index in the value function
        goal_state_ind = -1  # index of goal state
        for x, row in enumerate(maze_data):
            for y, state in enumerate(row):
                state_ind += 1
                if maze_data[x, y] == '*':
                    # skip obstacles
                    continue
                if maze_data[x, y] == 'G':
                    # agent stays in goal/terminal state
                    next_states[state_ind] = state_ind
                    goal_state_ind = state_ind
                    continue
                if y > 0 and maze_data[x, y-1] != '*':
                    # can move west
                    next_states[state_ind, 0] = state_ind - 1
                else:
                    # agent stays
                    next_states[state_ind, 0] = state_ind
                if x > 0 and maze_data[x-1, y] != '*':
                    # can move north
                    next_states[state_ind, 1] = state_ind - maze_data.shape[1]
                else:
                    # agent stays
                    next_states[state_ind, 1] = state_ind
                if y < maze_data.shape[1] - 1 and maze_data[x, y+1] != '*':
                    # can move east
                    next_states[state_ind, 2] = state_ind + 1
                else:
                    # agent stays
                    next_states[state_ind, 2] = state_ind
                if x < maze_data.shape[0] - 1 and maze_data[x+1, y] != '*':
                    # can move south
                    next_states[state_ind, 3] = state_ind + maze_data.shape[1]
                else:
                    # agent stays
                    next_states[state_ind, 3] = state_ind

        return values, next_states, goal_state_ind, maze_data.shape

    def learn(self, values, next_states, goal_state_ind, lam, num_epoch):
        """
        Learn by value iteration. Uses synchronous update.

        lam: the discount factor \\lambda

        Returns the learned values, q-values and optimal policies.
        """
        # Update values V(s)
        with warnings.catch_warnings():
            # Suppress the "RuntimeWarning: All-NaN slice encountered" message
            # in np.nanmax and np.nanargmax
            # Ref: https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
            warnings.simplefilter('ignore')

            if num_epoch > 0:
                for e in range(num_epoch):
                    # Immediate reward will always be -1 for each action...
                    values = -1 + lam * np.nanmax(values[next_states], axis=1)
                    # ...except for goal/terminal state: it incurs no cost staying there
                    # Note: according to assignment's reference output, other than
                    # the goal state, all other "stay" actions will still receive
                    # a reward of -1.
                    values[goal_state_ind] += 1
                    # print('values in epoch %d:\n' % (e+1), values[1:].reshape(self.maze_dimension))
                # print('values:', values)
            else:
                # Keep running until convergence
                iteration_count = 0
                while True:
                    iteration_count += 1

                    new_values = -1 + lam * np.nanmax(values[next_states], axis=1)
                    new_values[goal_state_ind] += 1

                    max_diff = np.nanmax(np.absolute(new_values - values))
                    # print('iteration %d: max_diff %f' % (iteration_count, max_diff))
                    values = new_values
                    # print('values in epoch %d:\n' % (iteration_count), values[1:].reshape(self.maze_dimension))
                    if max_diff < 0.001:
                        # values converged
                        break
                print('total iterations:', iteration_count)

            # Compute Q values Q(s, a): expected discounted reward for taking
            # action a in state s
            q_values = -1 + lam * values[next_states]
            q_values[goal_state_ind] += 1
            # print('q_values:\n', q_values)

            # Compute optimal policies \\pi(s)
            # Note: np.nanargmax raises ValueError if there're rows of all NaNs;
            # hence the mask method below
            # Create mask that highlights rows of all NaNs
            mask = np.all(np.isnan(q_values), axis=1)
            policies = np.empty(values.shape, dtype=np.float)
            policies[mask] = np.nan  # rows that are all NaNs don't have a policy
            policies[~mask] = np.nanargmax(q_values[~mask], axis=1)
            # print('policies:', policies)

        # Remove the padded first (row of) NaN values
        return values[1:], q_values[1:], policies[1:]

    def output_values(self, learned_values, value_file):
        """
        Output learned values V(s) to file.
        """
        with open(value_file, mode='w') as f:
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if not np.isnan(learned_values[i]):
                    f.write('%d %d %f\n' % (x, y, learned_values[i]))

    def output_q_values(self, learned_q_values, q_value_file):
        """
        Output learned q values Q(s, a) to file.
        """
        with open(q_value_file, mode='w') as f:
            # Create mask that highlights rows of all NaNs
            mask = np.all(np.isnan(learned_q_values), axis=1)
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if not mask[i]:  # not an obstacle
                    for d in range(learned_q_values.shape[1]):
                        f.write('%d %d %d %f\n' % (x, y, d, learned_q_values[i, d]))

    def output_policies(self, learned_policies, policy_file):
        """
        Output learned policies \\pi(s) to file.
        """
        with open(policy_file, mode='w') as f:
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if not np.isnan(learned_policies[i]):
                    f.write('%d %d %d\n' % (x, y, learned_policies[i]))


if __name__ == '__main__':
    maze_input = sys.argv[1]
    value_file = sys.argv[2]
    q_value_file = sys.argv[3]
    policy_file = sys.argv[4]
    num_epoch = int(sys.argv[5])
    discount_factor = float(sys.argv[6])

    model = ValueIteration(maze_input, value_file, q_value_file, policy_file,
                           num_epoch, discount_factor)

    # model = ValueIteration('tiny_maze.txt', 'value_output.txt',
    #                        'q_value_output.txt', 'policy_output.txt', 5, 0.9)

    # model = ValueIteration('medium_maze.txt', 'value_output.txt',
    #                        'q_value_output.txt', 'policy_output.txt', 5, 0.9)

    # model = ValueIteration('maze1.txt', 'value_output.txt',
    #                        'q_value_output.txt', 'policy_output.txt', -1, 0.9)
    # model = ValueIteration('maze2.txt', 'value_output.txt',
    #                        'q_value_output.txt', 'policy_output.txt', -1, 0.9)
