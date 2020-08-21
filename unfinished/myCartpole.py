import torch
import torch.nn as nn
import numpy as np 
from tensorboardX import SummaryWriter
import gym
import ptan

HIDDEN_SIZE = 32
epsilon = 1
EPSILON_DECAY = 0.99
REPLAY_SIZE = 10000
GAMMA = 0.95

solved = False
writer = SummaryWriter('saves')


env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
actions = env.action_space.n 

# Neural Net class
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())

net = Net(obs_size, HIDDEN_SIZE, actions)

tgt_net = ptan.agent.TargetNet(net)

selector = ptan.actions.ArgmaxActionSelector()
# if not epsilon, argmax
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1, selector=selector)
agent = ptan.agent.DQNAgent(net, selector)

exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=GAMMA)

buffer = ptan.experience.PrioReplayBufferNaive(exp_source, REPLAY_SIZE)

step = 0
episode = 0
solved = False

while not solved:
    step+=1
    buffer.populate(1)
    env.render()
    print(exp_source.pop_rewards_steps())