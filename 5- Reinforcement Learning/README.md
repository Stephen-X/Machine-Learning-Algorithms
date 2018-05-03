# Maze Solver with Reinforcement Learning
**Author:** Stephen Tse \<***@cmu.edu\>

This project implements a maze solver with reinforcement learning. It supports both
(model-based) value iteration learning and (model-less) Q-Learning.


## Assumptions & RL Setups

1. A maze is rectangular with one and only one start state and one and only one goal state. We use “S” to indicate the initial state (start state), “G” to indicate the terminal state (goal state), ”*” to indicate an obstacle, and “.” to indicate a state an agent can go through. One can assume that there are no obstacles at “S” and “G”, and there are walls around the border of the maze.

    Example:
    ```
    .*..*..G
    S......*
    ```

2. When the agent reaches the goal/terminal state, it'll stay there.

3. Whenever the agent performs an action (except when it stays in the goal/terminal state), it receives an immediate reward of -1.

4. In each state, the agent has 4 possible actions: moving west, north, east or south. However, when the agent takes an action that'll hit the border or an obstacle, it'll return to the original state, and it'll still need to pay -1 immediate reward.

## Implementation Notes

1. As per assignment requirement, the learning rate \\alpha in q-learning is set to a constant defined by users in the program. In practice, it's better to define it as 1 / number_of_times_you_visited_Q(s, a), so that \\alpha approaches zero as number of Q visits grows.

## Usage

You may use `value_iteration.py` for model-based learning, or use `q_learning.py` for model-less learning (in this case, the environment model is hidden from the learning algorithm: it will learn by interacting with the `environment` class, which only reports the next state and the reward the agent receives after making an action and whether it has arrived at the goal state).

```bash
python value_iteration.py <maze input> <value file> <q value file> <policy file> <num epoch> <discount factor>
```
1. `<maze input>`: path to the environment input .txt
2. `<value file>`: path to output the values V(s)
3. `<q value file>`: path to output the q-values Q(s; a)
4. `<policy file>`: path to output the optimal actions \\pi(s)
5. `<num epoch>`: the number of epochs your program should train the agent for. Setting it to numbers <= 0 will allow the program to keep looping until convergence.
6. `<discount factor>`: the discount factor \\lambda

```bash
python q_learning.py <maze input> <value file> <q value file> <policy file> <num episodes> <max episode length> <learning rate> <discount factor> <epsilon>
```
1. `<maze input>`: path to the environment input .txt
2. `<value file>`: path to output the values V(s)
3. `<q value file>`: path to output the q-values Q(s; a)
4. `<policy file>`: path to output the optimal actions \\pi(s)
5. `<num episodes>`: the number of episodes your program should train the agent for.
6. `<max episode length>`: the maximum of the length of an episode. Setting it to numbers <= 0 will allow the agent to keep exploring until it reaches the goal state for the current episode.
7. `<learning rate>`: the learning rate \\alpha of the q-learning algorithm
8. `<discount factor>`: the discount factor \\lambda
9. `<epsilon>`: the value \\epsilon for the epsilon-greedy strategy

Find the optimal policy for each state in `<policy file>` output. It's formatted as “x y optimal action” for each state s = (x, y) and the corresponding optimal policy \\pi(s). The 4 possible actions: west, north, east and south is denoted by integer 0, 1, 2, 3 respectively.


### Example

```bash
python value_iteration.py data/maze2.txt value_iteration_output/value_output.txt value_iteration_output/q_value_output.txt value_iteration_output/policy_output.txt -1 0.9
```

```bash
python q_learning.py data/maze2.txt q_learning_output/value_output.txt q_learning_output/q_value_output.txt q_learning_output/policy_output.txt 2000 -1 0.1 0.9 0.8
```


## Language & Dependencies

**Language:** Python 3.6

**Dependency Requirements:** `numpy` (tested with 1.14.1)
