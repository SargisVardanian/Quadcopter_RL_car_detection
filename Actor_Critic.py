import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch.distributions import Categorical

# Определяем размерность входа и выхода сети Actor
input_dim = 10
action_dim = 16
output_dim = 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("x + self.pe: ", x.shape, self.pe[:x.size(1), :].unsqueeze(0).squeeze(2).shape)
        x = x + self.pe[:x.size(1), :].unsqueeze(0).squeeze(2)
        # print("x + self.pe: ", x.shape)
        return x

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0):
        super(FFN, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        # self.gru = nn.GRU(hidden_size, 256, 1, batch_first=True)

    def forward(self, x):
        # Apply Layer Normalization
        x_norm = self.norm(x)

        # Linear transformation and ReLU activation
        x_hidden = self.dropout(self.elu(self.linear1(x_norm)))

        # Second linear transformation
        output = self.norm(self.elu(self.linear2(x_hidden)))
        # output, hidden_state = self.gru(output, hidden_state)
        return output

n_ffns = 2
state_len = 17
n_intervals = 40

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, state_hot=170, seq_len=1, hidden_dim=128, dropout=0, embed_dim=10, bbox=256):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(state_len * n_intervals, embed_dim)
        self.strides = torch.arange(0, state_len * n_intervals, n_intervals).reshape(1, 1, state_len)

        self.bbll = nn.Linear(bbox, bbox//4)
        self.attn = nn.MultiheadAttention(hidden_dim, 1, dropout=dropout)

        self.elu = nn.ReLU()

        self.ll_from_emb = nn.Linear(state_hot + bbox//4, hidden_dim)
        self.in_lin = nn.Linear(hidden_dim*seq_len, hidden_dim*2)
        self.gru = nn.GRU(self.seq_len * self.hidden_dim, self.hidden_dim*2, num_layers=1)

        self.ffns = nn.Sequential(*[
            FFN(hidden_dim * seq_len, hidden_dim*2, dropout)
            for _ in range(n_ffns)
        ])

        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.fin_ln = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.norm = nn.LayerNorm(hidden_dim*4)

    def forward(self, states, b_vec, hidden_state=None):
        """
        :param states: (batch_size, seq_len, state_len)
        # :param bboxes: (batch_size, seq_len, 256+1)
        :return:
        """
        batch_size,  d_model = states.size()
        x = self.embed(states.long() + self.strides)
        x = x.reshape([batch_size, -1])  # (batch_size, seq_len, state_len*embed_dim)

        position = torch.arange(0, 128, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(b_vec.device)
        position_embedding = torch.sin(position / 50)  # Повторяем для создания вектора 256
        position_embedding = torch.cat((position_embedding, position_embedding), dim=-1)

        b_vec = nn.LayerNorm(64)(self.bbll(b_vec.float() + position_embedding.squeeze(0)))
        x = torch.cat((x, b_vec), dim=-1)  # (batch_size, seq_len, state_len*embed_dim + 128)

        x = self.ll_from_emb(x)  # (batch_size, seq_len, hidden_dim)

        x = x.reshape([-1, self.seq_len * self.hidden_dim])
        x = self.ffns(x)  # (batch_size, d_model)
        x = self.in_lin(x)

        x = self.elu(self.fin_ln(x))
        return x, hidden_state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.shared_layers = Net(state_dim, action_dim)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, action_dim)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.00001)
        self.optimizer0 = optim.AdamW(self.parameters(), lr=0.0003)
        self.dropout = nn.Dropout(0)
        self.norm = nn.LayerNorm(hidden_dim*2)

    def forward(self, state, b_vec, hidden_state):
        x_net, hidden_state = self.shared_layers(state, b_vec, hidden_state)
        x = self.dropout(self.norm(self.fc2(x_net)))
        prob = self.fc3(x)
        # prob = F.softmax(prob, dim=-1)
        return prob, hidden_state


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.shared_layers = Net(state_dim, action_dim, hidden_dim=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        self.ffn = FFN(hidden_dim*4, int(3.5 * hidden_dim))
        self.x_bb = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001)
        self.optimizer0 = optim.AdamW(self.parameters(), lr=0.00003)
        self.dropout = nn.Dropout(0)
        self.norm = nn.LayerNorm(hidden_dim//2)

    def forward(self, state, b_vec):
        x_net, x = self.shared_layers(state, b_vec)
        x = self.dropout(self.fc2(x_net))
        value = self.fc3(x)  # Использование tanh для ограничения выхода
        return value


# class Actor_Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super(Actor_Critic, self).__init__()
#         self.shared_layers = Net(state_dim, action_dim, hidden_dim=hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
#         self.fcv = nn.Linear(hidden_dim*2, 1)
#         self.fca = nn.Linear(hidden_dim*2, action_dim)
#         self.ffn = FFN(hidden_dim*4, int(3.5 * hidden_dim))
#         self.x_bb = nn.Linear(hidden_dim*4, hidden_dim*2)
#         self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
#         self.dropout = nn.Dropout(0)
#         self.norm = nn.LayerNorm(hidden_dim*2)
#
#     def forward(self, state, b_vec, action=None):
#         x_net, b_attn_weights = self.shared_layers(state, b_vec)
#         x = self.dropout(self.norm(self.fc2(x_net)))
#         value = self.fcv(x)
#         # value_b = self.fcv(x)
#         x_final = self.fca(x)
#         prob = F.softmax(x_final, dim=-1)
#         return value, prob, b_attn_weights

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DDQN, self).__init__()
        self.shared_layers = Net(state_dim, action_dim, hidden_dim=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, action_dim)
        self.ffn = FFN(hidden_dim*4, int(3.5 * hidden_dim))
        self.x_bb = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.05)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_dim*2)

    def forward(self, state, b_vec):
        x_net, b_attn_weights = self.shared_layers(state, b_vec)
        x = self.dropout(F.relu(self.norm(self.fc2(x_net))))
        q_values = self.fc3(x)

        x_bbox = self.ffn(x_net)
        x_bbox = F.softmax(self.x_bb(x_bbox), dim=-1)

        return q_values, b_vec[:, -1, :]

