# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter12/02_pong_a2c.py
import gym_2048
import gym

import ptan
import numpy as np
import math
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.95
LEARNING_RATE = 0.003
ENTROPY_BETA = 0.002
BATCH_SIZE = 64
NUM_ENVS = 1

REWARD_STEPS = 4
CLIP_GRAD = 0.25
EPSILON = 1e-3

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)
    
    @staticmethod
    def process(frame):
        # convert to frame of floats instead of ints
        frame = list(np.float_(frame))
        # normalize? values in dataframe by taking their log
        for i in range(len(frame)):
            for j in range(len(frame[i])):
                if frame[i][j] != 0:
                    frame[i][j] = math.log(frame[i][j])
        # add channel dimension to frame
        frame = [frame]
        return frame

class A2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2C, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=1),
        nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.actor(conv_out), self.critic(conv_out)

if __name__ == "__main__":
    device = 'cpu'
    make_env = lambda : ProcessFrame(gym.make("2048-v0"))   
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter("2048A2Csaves")
    net = A2C((1, 4, 4), envs[0].action_space.n).to(device)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=EPSILON)

    batch = []
    rewards = []
    ep_steps = 0
    episodes = 0
    for step_idx, exp in enumerate(exp_source):
        batch.append(exp)
        rewards.append(exp.reward)
        print(exp.state)
        if exp.last_state is None:
            rew = sum(rewards)
            writer.add_scalar("episode reward", rew, episodes)
            episodes+=1
            rewards.clear()

        # time to train
        if len(batch) >= BATCH_SIZE:
            states = torch.FloatTensor([x.state for x in batch]).to(device)
            actions = torch.LongTensor([x.action for x in batch]).to(device)
            rewards = [x.reward for x in batch]
            last_states = [x.last_state for x in batch]

            last_state_indexes = []
            last_actual_states = []

            for index, state in enumerate(last_states):
                if state is not None:
                    last_state_indexes.append(index)
                    last_actual_states.append(state)
            
            rewards_np = np.array(rewards, dtype=np.float32)
            if last_state_indexes:
                last_states_v = torch.FloatTensor(np.array(last_actual_states, copy=False)).to(device)
                last_vals = net(last_states_v)[1]
                last_vals = last_vals.data.cpu().numpy()[:, 0]
                last_vals *= GAMMA ** REWARD_STEPS
                rewards_np[last_state_indexes] += last_vals
            
            vals = torch.FloatTensor(rewards).to(device)

            optimizer.zero_grad()
            logits_v, value_v = net(states)

            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            adv_v = vals - value_v.detach()
            log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

            loss_policy_v.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                    for p in net.parameters()
                                    if p.grad is not None])

            # apply entropy and value gradients
            loss_v = entropy_loss_v + loss_value_v
            loss_v.backward()
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
                    
            batch.clear()

# 2.5 days 1000 score
# GAMMA = 0.95
# LEARNING_RATE = 0.001
# ENTROPY_BETA = 0.01
# BATCH_SIZE = 64
# NUM_ENVS = 1

# REWARD_STEPS = 4
# CLIP_GRAD = 0.1
# EPSILON = 1e-3

