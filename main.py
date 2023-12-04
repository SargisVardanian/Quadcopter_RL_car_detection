import airsim
import numpy as np
import cv2
import math
import time
import pandas
from torchvision import transforms
import torch
from Model_DQN import *
from Script import *

state_size = 3+3+1+4
action_size = 4
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
discount_factor = 0.95
eps = 0.95
eps_decay_factor = 0.999
num_episodes = 500
num_samples = 4


def record_data_and_targets(client, num_samples, state, target):
    data_list = []
    target_list = []
    if np.random.random() < eps:
        for i in range(num_samples):
            action = np.random.uniform(0.4, 1.6, num_samples)
            state, target = perform_action(client, action)
            data_list.append(state)
            target_list.append(target)
    else:
        for i in range(num_samples):
            action = torch.argmax(model(np.vstack(state[i], target[i]))).item()
            print('action', action)
            state, target = perform_action(client, action)
            data_list.append(state)
            target_list.append(target)
    return np.array(data_list), np.array(target_list), state, target

def display_data(data_array):
    print(data_array)


state = [0] * 11
target = [0] * 4

for i in range(num_episodes):
    eps *= eps_decay_factor
    done = False
    while not done:
        data_list, target_list, state, target = record_data_and_targets(client, num_samples, state, target)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        client.hoverAsync().join()
        client.enableApiControl(False)
        client.armDisarm(False)
