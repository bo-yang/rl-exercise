"""
Example 4.3 of Sutton's Reinforcement Learning(2nd ed).
"""
import numpy as np
import matplotlib.pyplot as plt

goal = 100
states = np.arange(0, goal+1) # capitals, including state 0 and goal
values = np.zeros(len(states))
values[goal] = 1.0
policy = np.zeros(len(states))
ph = 0.4 # probability of the coin coming up heads
gamma = 1

theta = 1e-9
steps = 0
MAX_ITERATION = 100000
# value iteration
while steps < MAX_ITERATION:
    delta = 0
    for s in states[1 : goal]:
        actions = np.arange(0, min(s, goal-s)+1)
        expected_values = []
        v = values[s]
        for a in actions:
            win_reward = ph * gamma * values[s+a]
            lose_reward = (1-ph) * gamma * values[s-a]
            expected_values.append(win_reward + lose_reward)
        values[s] = max(expected_values)
        delta = max(delta, np.abs(v-values[s]))
    print("Step %d, values" %steps) # FIXME: TEST ONLY
    print(values) # FIXME: TEST ONLY
    steps += 1
    if delta < theta:
        break

# policy improvement
for s in states[1 : goal]:
    actions = np.arange(0, min(s, goal-s)+1)
    expected_values = []
    for a in actions:
        win_reward = ph * gamma * values[s+a]
        lose_reward = (1-ph) * gamma * values[s-a]
        expected_values.append(win_reward + lose_reward)
    policy[s] = actions[np.argmax(expected_values)]

# print value estimations
plt.figure(1)
plt.plot(states, values)
plt.axis([1, goal, 0, 1]) #[xmin, xmax, ymin, ymax]
plt.ylabel('Value Estimates')
plt.xlabel('Capital')
plt.grid(True)

# print optimal policy
plt.figure(2)
plt.plot(states, policy)
plt.axis([1, goal, 0, 50]) #[xmin, xmax, ymin, ymax]
plt.ylabel('Policy(Stake)')
plt.xlabel('Capital')
plt.grid(True)

plt.show()
