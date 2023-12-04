import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import math
from Script import *
import torch.nn.functional as F
import os

# class DQN(nn.Module):
#     def __init__(self, state_size, action_size, load_checkpoint=True):
#         super(DQN, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#
#         self.memory = deque(maxlen=5000)
#         self.gamma = 0.95
#         self.epsilon = 0.35
#         self.epsilon_decay = 0.999
#         self.epsilon_min = 0.001
#         self.max_allowed_pitch_roll = math.radians(15)
#         self.min_desired_speed = 2.0
#         self.learning_rate = 0.001
#         self.model = self.build_model()
#         # if load_checkpoint and os.path.isfile('model_checkpoint'):
#         #     self.model = torch.load('model_checkpoint')
#         #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         #     print('self.model', self.model)
#         # else:
#         #     self.model = self.build_model()
#
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.loss_fn = nn.MSELoss()
#
#     def build_model(self):
#         Police_head = nn.Sequential(
#             nn.Linear(self.state_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.Linear(128, 128),
#             # nn.Dropout(0.4),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, self.action_size),  # Action size is 4 for controlling the quadcopter
#             nn.Tanh()
#         )
#
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state, explore=True):
#         if explore and np.random.rand() <= self.epsilon:
#             print('explore', explore)
#             # if np.random.rand() <= self.epsilon:
#             #     return np.array([0.4, 1.9, 0.6, 1.5, 0.05]).tolist()
#             return np.hstack((np.random.rand(self.action_size-1), np.array([0.05])))
#             # else:
#             #     return np.array([0.6, 1.9, 0.6, 1.9, 0.025]).tolist()
#         else:
#             state_tensor = torch.tensor([state], dtype=torch.float32)
#             q_values = self.model(state_tensor).clone().detach().numpy()[0].tolist()
#             # probabilities = F.softmax(q_values, dim=-1)
#             # action = probabilities.squeeze().tolist()
#             return  q_values # Returning a vector of 4 values for quadcopter control
#
#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
#                 target = (reward + self.gamma * torch.max(self.model(next_state_tensor)).item())
#
#             state_tensor = torch.tensor([state], dtype=torch.float32)
#             target_f = self.model(state_tensor).clone().detach().numpy()
#
#             action_np = np.array(action)
#             target_f[0] = action_np  # Assuming action is a vector
#             random_index = np.random.randint(len(action_np))
#             target_f[0, random_index] = target  # Update the specific action value
#             target_f_tensor = torch.tensor(target_f, dtype=torch.float32)
#
#             self.optimizer.zero_grad()
#             output = self.model(state_tensor)
#             loss = self.loss_fn(output, target_f_tensor)
#             loss.backward()
#             self.optimizer.step()
#         # print('target_f', target_f)
#         # print('target', target)
#         print('loss ', loss)
#         # if self.epsilon > self.epsilon_min:
#         #     self.epsilon *= self.epsilon_decay
#
#     def compute_reward(self, state, bounding_box):
#         z, pitch, roll, yaw, vx, vy, vz = state
#         x, y, w, h = bounding_box
#         # Штраф, если баундинг бокс теряется с виду
#         if w == 1000 and h == 1000:
#             no_box_rew = -2.0  # Присвойте подходящее отрицательное значение
#         else:
#             no_box_rew = 0
#         # Желаемые значения для центра баундинг бокса
#         target_x, target_y = 72, 128
#
#         # Штраф за отклонение размеров баундинг бокса от заданных
#         size_penalty = max(0, abs(w * h - 800) / 800)
#
#         # Увеличь важность нахождения баундинг бокса (увеличь вес center_penalty)
#         center_penalty = abs(x + w / 2 - target_x) / 72 + abs(y + h / 2 - target_y) / 128
#
#         # Простое вознаграждение за сохранение высоты
#         # height_reward = max(0, 1 - abs(z - desired_height) / desired_height)
#
#         # Пример: штраф за сильные наклоны
#         # pitch_roll_penalty = max(0, abs(pitch) + abs(roll) - self.max_allowed_pitch_roll)
#
#         # Увеличь надобность в движении (увеличь вес speed_reward)
#         speed = math.sqrt(vz ** 2 + vy ** 2 + vx ** 2)
#         if speed < self.min_desired_speed:
#             speed_reward = -1
#         else:
#             speed_reward = 1
#
#         # Общее вознаграждение
#         total_reward = no_box_rew + speed_reward - size_penalty - center_penalty
#
#         return total_reward
#
#     def act_and_compute_reward(self, state):
#         action = self.act(state)
#         print('action', action)
#         new_state, bounding_box = perform_action(client, action)
#         # print(f'new_state{new_state}, bounding_box{bounding_box}')
#
#         reward = self.compute_reward(new_state, bounding_box)
#         self.remember(state, action, reward, new_state, False)
#         return action, reward

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).long(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int()
        )

    def __len__(self):
        return len(self.buffer)


class DDQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDQNAgent:
    def __init__(self, state_size, action_size, seed, learning_rate=1e-3, capacity=1000000,
                 discount_factor=0.99, tau=1e-3, update_every=4, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0
        self.memory = deque(maxlen=5000)

        self.qnetwork_local = self.build_DDQM() #Qetwork(state_size, action_size)
        self.qnetwork_target = self.build_DDQM() #QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity)
        self.min_desired_speed = 2.0
        self.update_target_network()


    def build_DDQM(self):
        return nn.Sequential(
                nn.Linear(self.state_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.Linear(128, 128),
                # nn.Dropout(0.4),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_size),
                nn.Tanh()
            )

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.discount_factor * (Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1, 1))
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def update_target_network(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def compute_reward(self, state, bounding_box):
        z, pitch, roll, yaw, vx, vy, vz = state
        x, y, w, h = bounding_box
        if w == 1000 and h == 1000:
            no_box_rew = -2.0
        else:
            no_box_rew = 0
        target_x, target_y = 72, 128
        size_penalty = max(0, abs(w * h - 800) / 800)
        center_penalty = abs(x + w / 2 - target_x) / 72 + abs(y + h / 2 - target_y) / 128

        # height_reward = max(0, 1 - abs(z - desired_height) / desired_height)
        # pitch_roll_penalty = max(0, abs(pitch) + abs(roll) - self.max_allowed_pitch_roll)
        speed = math.sqrt(vz ** 2 + vy ** 2 + vx ** 2)
        if speed < self.min_desired_speed:
            speed_reward = -1
        else:
            speed_reward = 1
        total_reward = no_box_rew + speed_reward - size_penalty - center_penalty
        return total_reward

    def act_and_compute_reward(self, state):
        action = self.act(state)
        print('action', action)
        new_state, bounding_box = perform_action(client, action)
        reward = self.compute_reward(new_state, bounding_box)
        self.remember(state, action, reward, new_state, False)
        return action, reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, eps=0.0):
        self.qnetwork_local.eval()
        print('state', state)
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

# Define QNetwork and ReplayBuffer classes
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


state_size = 1+3+3
action_size = 5
seed = 42

dqn_agent = DDQNAgent(state_size, action_size, seed)
num_episodes = 1000
batch_size = 64

max_steps_per_episode = 32
save_interval = 32
step_count = 0

for episode in range(num_episodes):
    state = get_initial_state()
    while True:
        # action, reward = dqn_agent.act_and_compute_reward(state)
        if len(dqn_agent.replay_buffer) > batch_size:
            dqn_agent.learn(dqn_agent.replay_buffer.sample(batch_size))
        new_state = get_initial_state()
        state = new_state
        done = check_if_episode_is_done(step_count, max_steps_per_episode)
        if done:
            break
        step_count += 1

    step_count = 0
    torch.save(dqn_agent.qnetwork_local.state_dict(), f'model_checkpoint_{episode}.pt')
