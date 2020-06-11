# use DRL algorithm cross-entropy to learn whether to walk forwards or backwards
# tensorboard --logdir fbSaves --port=6006
from __future__ import print_function
from collections import namedtuple
from builtins import range
import MalmoPython
import os
import sys
import time
import json
import math
import numpy as np

import tensorboardX
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# compiler version stuff
if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# agent actions
actions = ["move -0.7", "move 0", "move 0.7"]
batchesDone = 0
# NN info - inputs will be block directly below agent, the block in front, and in back of that
LayerHeights = [3, 4, 3]
# will the episode be deterministic according our NN optimal policy? (0 exploration)
optimalEpisode = False
# appended to at training time
meanRewards = []
losses = []
# helper object types for storage
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# tensorboard initial setup
writer = SummaryWriter(logdir='fbSaves/tBoard')

class Net(nn.Module):
    def __init__(self, LayerHeights):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(LayerHeights[0], LayerHeights[1]),
            nn.ReLU(),
            nn.Linear(LayerHeights[1], LayerHeights[2]),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# returns observations from environment as 1d list of 0s, 1s
def getFeatures(world_state):
    # get text of last of recent observations
    msg = world_state.observations[-1].text               
    observations = json.loads(msg)       
    # a 1x1x3 grid
    grid = observations.get(u'frontBack', 0)  
    try:
        inputVector = [0] * len(grid)
        # creating feature vector --> emerald is 1, any other block is 0
        for index, blockType in enumerate(grid):
            if blockType == 'emerald_block':
                inputVector[index] = 1
            else:
                inputVector[index] = 0
    # the grid had no items or something went wrong
    except:
        inputVector = []

    return inputVector

# takes a batch and returns the best episode, which will be trained on
def filter_batch(batch):
    print("filtering")
    # a min value beyond anything which could be reached in an episode
    highestReward = -100000
    bestEpisode = []
    mean_r = 0
    train_obs = []
    train_act = []
    for episode in batch:
        mean_r += episode.reward
        if episode.reward > highestReward:
            highestReward = episode.reward
            bestEpisode = episode
    train_obs.extend(map(lambda step: step.observation, bestEpisode.steps))
    train_act.extend(map(lambda step: step.action, bestEpisode.steps))
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    # making sure its an average not an accumulation :)
    mean_r /= len(batch)
    return train_obs_v, train_act_v, mean_r
    
# executes actions and returns next observations, reward, and if is done
def act(agent_host, action, world_state):
    # get agent location from world 
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    # execute action
    agent_host.sendCommand(actions[action])
    # reevaluate world state after movement
    world_state = agent_host.getWorldState()
    # waiting until an observation actually comes in
    while world_state.number_of_observations_since_last_state <= 0 and world_state.is_mission_running:
        world_state = agent_host.getWorldState()
    # make sure the episode isnt over
    if world_state.is_mission_running:
        reward = -1
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'frontBack', 0)
        try:
            if grid[1] == "emerald_block":
                reward = 100
        except:
            reward = -1
            print("error", grid)
        return reward
    # if the episode/mission somehow stops
    return 0

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# set up the mission
mission_file = './forwardsBackwards.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

my_mission_record = MalmoPython.MissionRecordSpec()

# constructing NN
# neuralNet = torch.load("/Users/sander/Desktop/RL/Minecraft/Malmo-0.37.0-Mac-64bit_withBoost_Python3.7/Trigaten_Examples/fbSaves/NNSaves/forwardsBackwardsNet")
neuralNet = Net(LayerHeights)
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=neuralNet.parameters(), lr=0.01)
# how many times has the agent executed an action
actCount = 0
batch = []
means = []
episode_reward = 0.0
episode_steps = []
# Attempt to start a mission:
max_retries = 15

if agent_host.receivedArgument("test"):
    current_episode = 1
else:
    current_episode = 20000

for i in range(current_episode):
    # training time - train on the best episode out of every ten episodes
    if len(batch) == 10 and not optimalEpisode:
        print("training")
        # get info from filter function
        train_obs, train_actions, mean_r = filter_batch(batch)
        if len(train_obs) > 0 and len(train_actions) > 0:
            # clear grads
            optimizer.zero_grad()
            averageLoss = 0
            # forward pass
            action_scores_v = neuralNet(train_obs)
            # compute loss
            loss_v = objective(action_scores_v, train_actions)
            # backprop
            loss_v.backward()
            optimizer.step()   
            # append to list of losses
            losses.append(loss_v.item())
            # append to list of mean scores
            meanRewards.append(mean_r)
            print(i)
            torch.save(neuralNet, "fbSaves/NNSaves/forwardsBackwardsNet")
            batch = []
            inc = (i)/10
            # writer.add_scalar(f'feature_vectors/000/back', 1, 0)
            writer.add_scalar("Loss", loss_v.item(), inc)
            writer.add_scalar("Mean Batch Reward", mean_r, inc)

            # for [0, 0, 0], move should be backwards (0)
            output = neuralNet(torch.FloatTensor([0, 0, 0]))
            writer.add_scalar(f'feature_vectors/000/back', output[0], inc)
            writer.add_scalar(f'feature_vectors/000/stay', output[1], inc)  
            writer.add_scalar(f'feature_vectors/000/forwards', output[2], inc)

            # for [0, 0, 1], move should be forwards (2) - emerald in FRONT
            output = neuralNet(torch.FloatTensor([0, 0, 1]))
            writer.add_scalar(f'feature_vectors/001/back', output[0], inc)
            writer.add_scalar(f'feature_vectors/001/stay', output[1], inc)  
            writer.add_scalar(f'feature_vectors/001/forwards', output[2], inc)

            # for [0, 1, 0], move should be stay (1)
            output = neuralNet(torch.FloatTensor([0, 1, 0]))
            writer.add_scalar(f'feature_vectors/010/back', output[0], inc)
            writer.add_scalar(f'feature_vectors/010/stay', output[1], inc)  
            writer.add_scalar(f'feature_vectors/010/forwards', output[2], inc)  

            # for [1, 0, 0], move should be backwards (0) - emerald BEHIND
            output = neuralNet(torch.FloatTensor([1, 0, 0]))
            writer.add_scalar(f'feature_vectors/100/back', output[0], inc)
            writer.add_scalar(f'feature_vectors/100/stay', output[1], inc)  
            writer.add_scalar(f'feature_vectors/100/forwards', output[2], inc)

            batchesDone+=1
            if batchesDone % 100 == 0:
                torch.save(neuralNet, "fbSaves/NNSaves/forwardsBackwardsNet" + str(batchesDone/100))
    print()
    print('Repeat %d of %d' % (i+1, current_episode))
    my_mission_record = MalmoPython.MissionRecordSpec()

    # trying to start next episode (also reset some info)
    for retry in range(max_retries):
        if i > 3:
            print("problem")
        try:
            actCount = 0
            agent_host.startMission(my_mission, my_mission_record)
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            actCount = 0
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    
    while not world_state.has_mission_begun:
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ")

    # the soft max function
    sm = nn.Softmax(dim=1)
    inputVector = []
    # Loop until mission ends:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        if world_state.number_of_observations_since_last_state > 0: # Have any observations come in?
            features = getFeatures(world_state)
            # converting list inputVector to torch vector
            obs_v = torch.FloatTensor([features])
            # if agent is still in training mode (exploration)
            if optimalEpisode == False:
                # occasionally can fail possibly due to env error
                try:
                    # compute softmax prob dist
                    act_probs_v = sm(neuralNet(obs_v))
                    act_probs = act_probs_v.data.numpy()[0]
                    # selects an action to take according to the softmax prob dist
                    action = np.random.choice(len(act_probs), p=act_probs)
                except:
                    # no movement
                    action = 1
            # if agent is acting optimally (deterministic policy) 
            else:
                action = np.argmax(neuralNet(obs_v).data.numpy())
            actCount += 1
            reward = act(agent_host, action, world_state)
            if optimalEpisode:
                agent_host.sendCommand("pitch 0.2")
            episode_reward += reward
            episode_steps.append(EpisodeStep(observation=features, action=action))

writer.close()
print("Mission ended")
# Mission has ended.