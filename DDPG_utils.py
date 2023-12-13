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

class Memory:
    def __init__(self, max_size, token_size):
        self.max_size = max_size
        self.token_size = token_size
        self.memory = [torch.zeros(1, token_size) for _ in range(max_size)]
        self.mask_token = torch.zeros(1, token_size)  # Placeholder for <MASK>

    def push(self, state):
        for _ in range(state.size(0)):
            self.memory.pop(0)
        self.memory.append(state)

    def get_memory(self):
        return torch.cat(self.memory, dim=0)

    def reset(self):
        self.memory = [self.mask_token.clone() for _ in range(self.max_size)]


class AttentionLayer(nn.Module):
    def __init__(self, patch_dim, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.patch_dim = patch_dim
        self.num_patches = (144 // patch_dim[1]) * (256 // patch_dim[2])
        self.patch_embedding = nn.Linear(patch_dim[1] * patch_dim[2] * 3, patch_dim[1] * patch_dim[2] * 3)
        self.positional_embedding = nn.Embedding(patch_dim[1] * patch_dim[2], patch_dim[1] * patch_dim[2] * 3)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=patch_dim[1] * patch_dim[2] * 3, num_heads=num_heads) for _ in range(3)
        ])
        # self.cls_token = nn.Parameter(torch.ones(1, 1, patch_dim[1] * patch_dim[2] * 3))  # Используем константу 1
        self.cls_token = nn.Parameter(torch.rand(1, 1, patch_dim[1] * patch_dim[2] * 3))

    def forward(self, state):
        patches = self.make_patches(state)
        positions = torch.arange(patches.size(0)).unsqueeze(1).repeat(1, 12*16)
        positional_embeddings = self.positional_embedding(positions)
        patch_embeddings = patches + positional_embeddings

        # Добавляем [CLS] токен
        cls_tokens = self.cls_token.repeat(patch_embeddings.size(0), 1, 1)
        patch_embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        attn_output = patch_embeddings
        for attn_layer in self.attention_layers:
            attn_output, _ = attn_layer(attn_output, attn_output, attn_output)
        cls_output = attn_output[:, 0]
        return cls_output


    def make_patches(self, state):
        kc, kh, kw = self.patch_dim
        dc, dh, dw = self.patch_dim
        x = state
        x = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
                      x.size(1) % kh // 2, x.size(1) % kh // 2,
                      x.size(0) % kc // 2, x.size(0) % kc // 2))
        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(-1, 192, kc * kh * kw)
        return patches

    def Reshape_back(self, patches):
        unfold_shape = patches.size()
        patches_orig = patches.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        img = patches_orig.view(1, output_c, output_h, output_w)
        return img

# attention_layer = AttentionLayer(patch_dim=(3, 12, 16), num_heads=16)
# images = torch.randn(2, 3, 144, 256)
# attention_output = attention_layer(images)
# print('attention_output', attention_output.shape)
# print(summary(attention_layer, input_size=(3, 144, 256)))


# Определение Actor сети с аттеншен-слоем
class Actor(nn.Module):
    def __init__(self, action_dim, patch_dim, num_heads=8,  memory_size=128, token_size=576):
        super(Actor, self).__init__()
        self.positional_embedding = nn.Embedding(patch_dim[1] * patch_dim[2],
                                                 patch_dim[1] * patch_dim[2] * 3)
        self.positional_embedding.weight.requires_grad = False

        self.memory = Memory(max_size=memory_size, token_size=token_size)
        self.attention_layer = AttentionLayer(patch_dim, num_heads)
        self.fc1 = nn.Sequential(
            nn.Linear(patch_dim[1] * patch_dim[2] * 3, patch_dim[1] * patch_dim[2]),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(patch_dim[1] * patch_dim[2], action_dim)

        self.cls_token_1 = nn.Parameter(torch.full((1, patch_dim[1] * patch_dim[2] * 3), 0.1), requires_grad=False)
        self.cls_token_2 = nn.Parameter(torch.full((1, 1, patch_dim[1] * patch_dim[2] * 3), 0.2), requires_grad=False)
        self.cls_token_3 = nn.Parameter(torch.full((1, 1, patch_dim[1] * patch_dim[2] * 3), 0.3), requires_grad=False)
        self.cls_token_4 = nn.Parameter(torch.full((1, 1, patch_dim[1] * patch_dim[2] * 3), 0.4), requires_grad=False)

    def forward(self, state):
        state = self.attention_layer(state)
        # state = state.view(state.size(0), 1, 576)
        self.memory.push(state)
        memory_state = self.memory.get_memory()

        positions = torch.arange(memory_state.size(0))
        positional_embeddings = self.positional_embedding(positions)
        memory_state = memory_state + positional_embeddings
        print('memory_state', memory_state.shape)
        # Добавляем размерность для батча
        cls_tokens = self.cls_token_1.repeat(state.size(0), 1)
        x = torch.cat([cls_tokens, memory_state], dim=0)

        print('x', x.shape)
        # Аттеншен-слои
        for attn_layer in self.attention_layer.attention_layers:
            x, _ = attn_layer(x, x, x)
        print('x_attention', x.shape)
        x = x[0].unsqueeze(0)
        x = self.fc1(x)
        return state, torch.softmax(self.fc2(x), dim=-1)

actor = Actor(action_dim=10, patch_dim=(3, 12, 16), num_heads=16,  memory_size=8, token_size=576)
images = torch.randn(1, 3, 144, 256)
state, prob_vec = actor(images)
print('prob_vec', prob_vec, torch.sum(prob_vec))
# print(summary(actor, input_size=(3, 144, 256)))

# Определение Critic сети
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.input_state = nn.Linear(576, 128)
        self.input_prob_vec = nn.Linear(10, 128)
        self.fc1 = nn.Linear(2*128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=1)
        self.output = nn.Linear(128, 1)

    def forward(self, state, prob_vec):
        state = self.input_state(state)
        action = self.input_prob_vec(prob_vec)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        # Аттеншен-слои
        x, _ = self.attention(x, x, x)
        x, _ = self.attention(x, x, x)
        x, _ = self.attention(x, x, x)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.unsqueeze(0)  # Добавляем размерность для батча

        return self.fc2(x)


# Инициализация сетей
state_dim = (3, 144, 256)  # RGB изображение
action_dim = 6  # У нас 6 действий
# actor = Actor(state_dim, action_dim)
# critic = Critic(state_dim, action_dim)




