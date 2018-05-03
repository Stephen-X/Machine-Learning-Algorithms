#!/usr/bin/env python3
"""
Project 8: Maze Solver with Reinforcement Learning
Author: Stephen Xie <***@andrew.cmu.edu>

Learn policies to walk through a maze by reinforcement learning. This program
implements Q-learning algorithm, a model-free version of reinforcement learning
algorithm.

Data Assumptions:
1. Maze data are rectangular (i.e. all rows have the same number of characters)
   and valid (only contains 'S', 'Q', '.', '*'; etc.).

Notes:
1. As per assignment requirement, the learning rate \\alpha in q-learning is
   set to a constant defined by users in the program. In practice, it's better
   to define it as 1 / number_of_times_you_visited_Q(s, a), so that \\alpha
   approaches zero as number of Q visits grows.
"""

import sys
import numpy as np
import random
from itertools import product
from environment import Environment


class QLearning:
    def __init__(self, maze_input, value_file, q_value_file, policy_file,
                 num_episodes, max_episode_length, learning_rate,
                 discount_factor, epsilon):
        agent = Environment(maze_input)
        # print(agent.maze_data)

        # Initialize q values (Q^*(s, a); expected discounted reward for taking
        # action a in state s) to all zeros
        # Note: for computation and output convenience, here the environment
        # class will also report on total number of states and maze dimension.
        # These may not be necessary in strict definition of q-learning
        # algorithm.
        q_values = np.zeros((agent.get_total_states(), 4))
        self.maze_dimension = agent.get_maze_dimension()
        # Note: The locations of obstacles are used only for result output (as
        # per assignment requirement).
        obstacles = agent.get_obstacles()

        total_steps = 0
        for episode in range(num_episodes):
            total_steps += self.learn(agent, q_values, max_episode_length,
                                      epsilon, learning_rate, discount_factor)
        print('average # steps per episode:', total_steps / num_episodes)

        # Compute final values V(s)
        values = np.max(q_values, axis=1)
        # Compute optimal policies \\pi(s)
        policies = np.argmax(q_values, axis=1)

        # Output results
        self.output_values(values, obstacles, value_file)
        self.output_q_values(q_values, obstacles, q_value_file)
        self.output_policies(policies, obstacles, policy_file)

    def learn(self, agent, q_values, max_episode_length, epsilon, alpha, lam):
        """
        In one episode, learn by moving the agent around, and update q-values
        accordingly.

        max_episode_length: maximum number of steps the agent can take before
            the current episode is forced to terminate (besides reaching
            terminal state)
        epsilon: parameter for epsilon-greedy action selection
        alpha: the learning rate for the q-learning algorithm
        lam: the discount factor \\lambda

        Returns: total number of steps taken in the current episode
        """
        curr_state = agent.reset()
        step_count = 0
        if max_episode_length > 0:
            for step in range(max_episode_length):
                step_count += 1
                # row index in q_values that corresponds to the current state
                curr_state_ind = self._get_state_index(curr_state)
                # Pick action and move the agent
                action = self.select_action(epsilon, q_values[curr_state_ind])
                next_state, reward, is_terminal = agent.step(action)
                next_state_ind = self._get_state_index(next_state)
                # Update q-values
                q_values[curr_state_ind, action] = \
                    (1 - alpha) * q_values[curr_state_ind, action] + \
                    alpha * (reward + lam * np.max(q_values[next_state_ind]))
                # Move on if terminal state is not reached
                if is_terminal:
                    break
                else:
                    curr_state = next_state
        else:
            # Keep running until the agent reaches terminal state
            while True:
                step_count += 1

                curr_state_ind = self._get_state_index(curr_state)
                action = self.select_action(epsilon, q_values[curr_state_ind])
                next_state, reward, is_terminal = agent.step(action)
                next_state_ind = self._get_state_index(next_state)
                q_values[curr_state_ind, action] = \
                    (1 - alpha) * q_values[curr_state_ind, action] + \
                    alpha * (reward + lam * np.max(q_values[next_state_ind]))
                if is_terminal:
                    break
                else:
                    curr_state = next_state

        return step_count

    def select_action(self, epsilon, q_values):
        """
        Select agent's next action with epsilon-greedy strategy, i.e. select
        the optimal action with probability 1 - \\epsilon and select uniformly
        at random from one of the 4 actions (0, 1, 2, 3) with probability
        \\epsilon.

        q_values: Q(s, *), an array of q values of all possible actions for the
                  current state
        """
        if random.uniform(0, 1) > epsilon:
            # Gives optimal action (exploitation)
            return np.argmax(q_values)
        else:
            # Gives random action (exploration)
            return random.choice((0, 1, 2, 3))

    def output_values(self, learned_values, obstacles, value_file):
        """
        Output learned values V(s) to file.
        """
        with open(value_file, mode='w') as f:
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if (x, y) not in obstacles:
                    f.write('%d %d %f\n' % (x, y, learned_values[i]))

    def output_q_values(self, learned_q_values, obstacles, q_value_file):
        """
        Output learned q values Q(s, a) to file.
        """
        with open(q_value_file, mode='w') as f:
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if (x, y) not in obstacles:
                    for d in range(learned_q_values.shape[1]):
                        f.write('%d %d %d %f\n' % (x, y, d, learned_q_values[i, d]))

    def output_policies(self, learned_policies, obstacles, policy_file):
        """
        Output learned policies \\pi(s) to file.
        """
        with open(policy_file, mode='w') as f:
            for i, (x, y) in enumerate(product(range(self.maze_dimension[0]),
                                               range(self.maze_dimension[1]))):
                if (x, y) not in obstacles:
                    f.write('%d %d %d\n' % (x, y, learned_policies[i]))

    def _get_state_index(self, coord):
        """
        Get the row index in the q_values matrix that corresponds to the given
        state.

        coord: coordinate tuple of the current state.
        """
        return coord[0] * self.maze_dimension[1] + coord[1]


if __name__ == '__main__':
    maze_input = sys.argv[1]
    value_file = sys.argv[2]
    q_value_file = sys.argv[3]
    policy_file = sys.argv[4]
    num_episodes = int(sys.argv[5])
    max_episode_length = int(sys.argv[6])
    learning_rate = float(sys.argv[7])
    discount_factor = float(sys.argv[8])
    epsilon = float(sys.argv[9])

    model = QLearning(maze_input, value_file, q_value_file, policy_file,
                      num_episodes, max_episode_length, learning_rate,
                      discount_factor, epsilon)

    # model = QLearning('tiny_maze.txt', 'value_output.txt', 'q_value_output.txt',
    #                   'policy_output.txt', 1000, 20, 0.8, 0.9, 0.05)

    # model = QLearning('maze1.txt', 'value_output.txt', 'q_value_output.txt',
    #                   'policy_output.txt', 2000, -1, 0.1, 0.9, 0.2)
    # model = QLearning('maze2.txt', 'value_output.txt', 'q_value_output.txt',
    #                   'policy_output.txt', 2000, -1, 0.1, 0.9, 0.2)

    # model = QLearning('maze2.txt', 'value_output.txt', 'q_value_output.txt',
    #                   'policy_output.txt', 2000, -1, 0.1, 0.9, 0.01)
    # model = QLearning('maze2.txt', 'value_output.txt', 'q_value_output.txt',
    #                   'policy_output.txt', 2000, -1, 0.1, 0.9, 0.8)
