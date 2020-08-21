import torch
import torch.nn as nn 
import gym
import ptan 
import random
from tensorboardX import SummaryWriter

writer = SummaryWriter('saves')
# writer.add_scalar('data/max_position', max_position, episode)
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

HIDDEN_SIZE = 64
BUFFER_SIZE = 5000
epsilon = 0.3
EPS_DECAY = 0.99
env = gym.make("MountainCar-v0")
EPISODES = 1000000
MIN_REWARD = -200
# instantiate sourceless buffer
buffer = ptan.experience.ExperienceReplayBuffer(None, BUFFER_SIZE)
numActions = env.action_space.n
net = Net(env.observation_space.shape[0], HIDDEN_SIZE, numActions)
tgt_net = ptan.agent.TargetNet(net)
for episode in range(EPISODES):
    state = torch.FloatTensor(env.reset())
    done = False
    epHistory = []
    rewardSum = 0
    steps = 0
    while done == False:
        # print(steps)
        # if episode % 200 == 0:
        #     env.render()
        # act randomly
        # if random.random() < epsilon:
        action = random.randint(0, numActions-1)
        # env.render()
        step = env.step(action)
        # else:
        #     prios = net(state)
        #     # returns highest value and its index
        #     value, index = prios.max(0)
        #     # extract value from tensor
        #     action = index.item()
        #     step = env.step(action)
        # episode done
        if step[2] == True:
            exp = ptan.experience.ExperienceFirstLast(state, action, step[1], True)
            done = True
        else:
            exp = ptan.experience.ExperienceFirstLast(state, action, step[1], step[0])
        if step[1] == 0:
            print("DONEDONE")
        epHistory.append(exp)
        # setting current S to next S
        state = torch.FloatTensor(step[0])
        # increment step counter
        steps+=1
        # adding to reward counter
        rewardSum += step[1]
        
        # if done:
        #     # print("step", steps)
        #     # print("reward", rewardSum)
        #     print("episode", episode)
        if done and rewardSum > MIN_REWARD:
            for i in epHistory:
                buffer.add(i)
            print(buffer)

            

        


# env = gym.make("MountainCar-v0")
# env.reset()
# for i in range(9999999):
#     x = env.step(random.randint(0, 2))
#     # print(x[1])
#     if x[1] > -1:
#         print("solved")
