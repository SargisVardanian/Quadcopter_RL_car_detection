import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import math
# from Script import *
import torch.nn.functional as F
import os
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from DDPG_utils import *
from Environment import *

class DDPGAlgorithm:
    def __init__(self, actor, critic, target_actor, target_critic, gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.gamma = gamma

    def update(self, obs_tensor, action_tensor, reward_tensor, next_obs_tensor):
        # Get Q-values for actions from trajectory
        current_q = self.critic(obs_tensor, action_tensor)

        # Get target Q-values
        target_q = reward_tensor + self.gamma * self.target_critic(next_obs_tensor, self.target_actor(next_obs_tensor))

        # L2 loss for the difference
        critic_loss = F.mse_loss(current_q, target_q)

        critic_loss.backward()

        # Actor loss based on the deterministic action policy
        actor_loss = -self.critic(obs_tensor, self.actor(obs_tensor)).mean()

        actor_loss.backward()

        # (Optionally) Perform optimization step for both actor and critic

    def collect_trajectories(self, env, num_trajectories):
        # Implement trajectory collection logic using the provided environment
        pass

actor = Actor(action_dim=10, patch_dim=(3, 12, 16), num_heads=16,  memory_size=8, token_size=576)
critic = Critic

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
gamma = 0.99


# Обновление Critic (Q-network)
def update_critic(state, action, next_state, reward):
    target_Q_value = reward + gamma * critic(next_state, actor(next_state).detach())
    current_Q_value = critic(state, action)
    critic_loss = F.mse_loss(current_Q_value, target_Q_value)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()


# Обновление Actor (P-network)
def update_actor(state):
    action = actor(torch.FloatTensor(state))
    actor_loss = -critic(torch.FloatTensor(state), action).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item()


# Обучение
num_episodes = 1000
max_timesteps_per_episode = 1000  # Параметр для ограничения числа шагов в эпизоде

for episode in range(num_episodes):
    state = np.random.rand(*state_dim)  # Начальное состояние, замените на реальные данные
    total_reward = 0

    for t in range(max_timesteps_per_episode):
        # Выбор действия от Actor
        action = actor(torch.FloatTensor(state)).detach().numpy()

        # Выполнение действия в среде и получение нового состояния и награды
        next_state, reward = env.step(action)

        # Обновление Critic и Actor
        critic_loss = update_critic(state, action, next_state, reward)
        actor_loss = update_actor(state)

        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Важно: Этот код предоставляет общий каркас. Вы должны настроить его под конкретные требования вашей задачи и учесть особенности вашей среды.
