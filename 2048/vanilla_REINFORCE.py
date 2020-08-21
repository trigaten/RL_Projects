# AUTHOR


# in about 24 hours got to 1100 mean score, 150 better than random agent. Also achieved 256 tile where random agent could not
# ALSO acheived 512 tile after 1 day 5 hrs
# got 1490 mean score after 4 days
import gym_2048
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import math
import ptan

from tensorboardX import SummaryWriter


GAMMA = 0.95
# for nn backprop
LR = 1e-3
EPISODES_TO_TRAIN = 6
# # https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/lib/dqn_model.py
# policy grad net
class PGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PGN, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=2, stride=1),
        nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        # print(conv_out_size)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

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
        # print(frame)
        return frame

# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/02_cartpole_reinforce.py
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == '__main__':
    writer = SummaryWriter('2048saves')
    env = gym.make("2048-v0")
    env = ProcessFrame(env)
    env.seed(42)
    # make a net that takes a single channel 4x4 frame and outputs probs for actions
    net = PGN((1, 4, 4), env.action_space.n)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    # nn optimizer
    optimizer = optim.Adam(net.parameters(), LR)

    stepCount = 0
    episodeCount = 0
    batch_episodes = 0
    total_episodes = 0
    steps = []
    rewards = []

    batch_rewards = []

    state = env.reset()

    for index, exp in enumerate(exp_source):
        steps.append(exp)
        writer.add_scalar("reward", exp.reward, index)
        if exp.reward > 1024:
            print(exp.reward)
        print(exp)
        rewards.append(exp.reward)
        if exp.last_state is None:
            writer.add_scalar("reward sum", sum(rewards), total_episodes)
            q_vals = calc_qvals(rewards)
            batch_rewards.extend(q_vals)
            rewards.clear()
            total_episodes+=1
            batch_episodes+= 1

        # train
        if batch_episodes >= EPISODES_TO_TRAIN:
            optimizer.zero_grad()

            # grabs states and adds channel dimension
            states = [x.state for x in steps]
            # states = [[state] for state in states]

            states_v = torch.FloatTensor(states)

            actions = [x.action for x in steps]
            actions_t = torch.LongTensor(actions)
            
            q_vals = torch.FloatTensor(batch_rewards)
            # print(len(states))
            # print("a", len(actions))
            # print("q", len(q_vals))

            # print(states_v)
            # raw net vals
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = q_vals * log_prob_v[range(len(states)), actions_t]
            loss_v = -log_prob_actions_v.mean()

            batch_rewards.clear()
            steps.clear()
            batch_episodes = 0
            
            loss_v.backward()
            optimizer.step()

writer.close()
        

        
        
#     print('Next Action: "{}"\n\nReward: {}'.format(
#       gym_2048.Base2048Env.ACTION_STRING[action], reward))
#     env.render()

#   print('\nTotal Moves: {}'.format(moves))


# # d = nn.Conv2d(2, 2, kernel_size=2, stride=1)

# # print(d.weight)

