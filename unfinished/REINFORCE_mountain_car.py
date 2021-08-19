import torch
import torch.nn as nn
import gym
import numpy as np
import ptan
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir runs

writer = SummaryWriter()
# __file__[:-3]
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class P_net(nn.Module):
    def __init__(self, in_features, out_features):
        super(P_net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, out_features),
        )

    def forward(self, x):
        return self.linear(x)


env = gym.make('LunarLander-v2')

net = P_net(env.observation_space.shape[0], env.action_space.n) 

optimizer = optim.Adam(net.parameters(), lr=0.001)

agent = ptan.agent.PolicyAgent(lambda x: net(x), apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor, device='cpu')
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.0, steps_count=1)

env.reset()


episode_count = 0
while True:
    states = []
    actions = [] 
    rewards = []
    for idx, exp in enumerate(exp_source):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)

        if exp.last_state is None:
            episode_count+=1
            break

        # env.render()

    q_vals = []
    GAMMA = 0.9
    total = 0
    for idx, reward in enumerate(reversed(rewards)):
        total *= GAMMA
        total+= reward
        q_vals.append(total)

    q_vals.reverse()

    q_vals = torch.FloatTensor(q_vals)

    actions = torch.LongTensor(actions)
    states = torch.FloatTensor(states)
    net_vals = net(states)
    net_probs = F.log_softmax(net_vals, dim=1)

    # an array which contains the corresponding log probability of each action in actions
    log_prob_actions_v = net_probs[range(len(states)), actions]
    # multiply by q vals
    log_prob_actions_v = q_vals * log_prob_actions_v

    loss_v = -log_prob_actions_v.mean()

    loss_v.backward()
    optimizer.step()
    optimizer.zero_grad()

    writer.add_scalar("ep reward", sum(rewards), episode_count)
    writer.add_scalar("ep loss", loss_v.item(), episode_count)
    writer.flush()




            
