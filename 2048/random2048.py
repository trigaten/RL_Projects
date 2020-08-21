# average reward:  951.932
# average steps:  113.045
import gym
import gym_2048

import random

EPISODES = 1000
env = gym.make("2048-v0")
total_reward = 0
total_steps = 0
for _ in range(EPISODES):
    env.reset()
    while True:
        state = env.step(random.randint(0, 3))
        total_reward+= state[1]
        total_steps+= 1
        if state[2] is True:
            break

print("average reward: ", total_reward/EPISODES)
print("average steps: ", total_steps/EPISODES)
