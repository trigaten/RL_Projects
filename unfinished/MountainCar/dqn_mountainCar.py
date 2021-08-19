# adapted from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter07/06_cartpole.py

import gym
import ptan 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter('saves')
HIDDEN_SIZE = 32
GAMMA = 0.95
REPLAY_SIZE = 10000
LR = 1e-3
EPS_DECAY=0.99999
TGT_NET_SYNC = 100
BATCH_SIZE = 16
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

env = gym.make("MountainCar-v0")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
# the net
net = Net(obs_size, HIDDEN_SIZE, n_actions)
tgt_net = ptan.agent.TargetNet(net)
# arg max
selector = ptan.actions.ArgmaxActionSelector()
# if not epsilon, argmax
selector = ptan.actions.EpsilonGreedyActionSelector(
    epsilon=1, selector=selector)
agent = ptan.agent.DQNAgent(net, selector)
# experience source
exp_source = ptan.experience.ExperienceSourceFirstLast(
    env, agent, gamma=GAMMA)
# buffer uses experience source
buffer = ptan.experience.ExperienceReplayBuffer(
    exp_source, buffer_size=REPLAY_SIZE)
# nn optimizer
optimizer = optim.Adam(net.parameters(), LR)

step = 0
episode = 0
solved = False

while True:
    # increment step counter
    step += 1
    # put data in buffer
    buffer.populate(1)
    # populated every 200 steps (at end of episode)
    for reward, steps in exp_source.pop_rewards_steps():
        
        episode += 1
        print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
            step, episode, reward, selector.epsilon))
        writer.add_scalar("reward", reward, episode)
        solved = reward > -110
    if solved:
        print("DONE")
        break
    
    # make sure we have emough samples in the buffer for training else go to top of loop
    if len(buffer) < 2*BATCH_SIZE:
        continue

    batch = buffer.sample(BATCH_SIZE)
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []

    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    
    states = torch.tensor(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    last_states = torch.tensor(last_states)

    last_state_q = net(last_states)
    # UPDATE Q(s, a)  = r + y*Q(s', a')
    best_last_q = torch.max(last_state_q, dim=1)[0]
    tgt_q = best_last_q * GAMMA + rewards

    optimizer.zero_grad()

    q_v = net(states)
    q_v = q_v.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    loss_v = F.mse_loss(q_v, tgt_q)
    loss_v.backward()
    writer.add_scalar("loss", loss_v.item(), episode)
    
    optimizer.step()
    selector.epsilon *= EPS_DECAY

    if step % TGT_NET_SYNC == 0:
        tgt_net.sync()
writer.close()