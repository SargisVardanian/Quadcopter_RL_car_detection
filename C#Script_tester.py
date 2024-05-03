import os
import torch
import random
import pandas as pd
import numpy as np
from Actor_Critic import *
import sys
from reward import reward_function
import torch.optim as optim
import torch.nn.functional as F
import pickle
from collections import deque, namedtuple
from torch.distributions import Categorical


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        data = deque(maxlen=1000)
    return data

def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def action_prob_save(actions):
    with open('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/saved_actions.pkl', 'wb') as f:
        pickle.dump(actions, f)

def action_prob_load():
    with open('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/saved_actions.pkl', 'rb') as f:
        loaded_actions = pickle.load(f)
    return loaded_actions

def init_hidden(batch_size, hidden_dim=256, num_layers=1):
    h0 = torch.zeros(num_layers, batch_size, hidden_dim)
    c0 = torch.zeros(num_layers, batch_size, hidden_dim)
    return (h0, c0)


try:
    recent_states = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_states.pkl')
    recent_probs = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_probs.pkl')
    recent_log_probs = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_log_probs.pkl')
    recent_actions = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_actions.pkl')
    recent_values = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_values.pkl')
    done = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/done.pkl')
    entropys = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/entropys.pkl')
    b_boxes = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/b_boxes.pkl')
    recent_q_values = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_q_values.pkl')
    rec_boxx = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/rec_boxx.pkl')
    recent_rewards = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_rewards.pkl')
    recent_hid = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_hid.pkl')
    loss_act = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_act.pkl')
    loss_ = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_.pkl')
    loss_crit = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_crit.pkl')
    sum_rewards = load_data('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/sum_rewards.pkl')
    # recent_hid.append(torch.zeros(1, 1, 256))

except:
    recent_actions = deque(maxlen=20)
    recent_states = deque(maxlen=20)
    recent_rewards = deque(maxlen=20)
    recent_probs = deque(maxlen=20)
    recent_values = deque(maxlen=20)
    recent_log_probs = deque(maxlen=20)
    entropys = deque(maxlen=20)
    recent_q_values = deque(maxlen=20)
    rec_boxx = deque(maxlen=20)
    recent_hid = deque(maxlen=20)
    loss = deque(maxlen=1000)
    loss_act = deque(maxlen=1000)
    loss_ = deque(maxlen=1000)
    loss_crit = deque(maxlen=1000)
    sum_rewards = deque(maxlen=1000)

    recent_hid.append(torch.zeros(1, 1, 256))

input_dim = 17
action_dim = 7
epsilon = 0.1

try:
    actor = Actor(input_dim, action_dim)
    critic = Critic(input_dim, action_dim)
    # ddqn = DDQN(input_dim, action_dim)
    # ddqn.load_state_dict(torch.load('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/ddqn7.pt'))
    actor.load_state_dict(torch.load('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/Actor7.pt'))
    critic.load_state_dict(torch.load('/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/Critic7.pt'))
    tf = True
except:
    # actor = Actor(input_dim, action_dim)
    # critic = Critic(input_dim, action_dim)
    print("new")
    tf = False
    # tf = True


actor_loss_fn = nn.L1Loss(reduction='none')
actor_loss_action_fn = torch.nn.CrossEntropyLoss(reduction='none')  # Используйте 'none' для получения вектора потерь
critic_loss_fn = torch.nn.MSELoss(reduction='none')  # Используйте 'none' для получения вектора потерь

def generate_action_number(states, state_dict, b_vec, hid=None):
    bbox_w = np.array(state_dict['Bounding_box_W'])
    bbox_h = np.array(state_dict['Bounding_box_H'])
    bbox_x = np.array(state_dict['Bounding_box_X'])
    bbox_y = np.array(state_dict['Bounding_box_Y'])  # Получение координаты Y ограничивающего прямоугольника
    altitude = np.array(state_dict['DisTerr'])
    angularVelocity_y = np.array(state_dict['angularVelocityY'])
    object_lost = (bbox_w == -1) | (bbox_h == -1)
    velocity_x = np.array(state_dict['RotX'])
    velocity_z = np.array(state_dict['RotZ'])

    old_act = np.array(states[:, -1])
    # object_lost_tensor = torch.from_numpy(object_lost).type(torch.bool)

    # Определение действий на основе различных условий
    actions = np.where(altitude < 2,  # Если дрон летит слишком низко
        5,  # Поддержание высоты
        np.where(altitude > 40,  # Если высота больше 40
            0,  # Выбор действия 0
            np.where(angularVelocity_y > 0.65,  # Если угловая скорость больше 0.5
                6,  # Поворот влево
                np.where(angularVelocity_y < -0.65,  # Если угловая скорость меньше -0.5
                    6,  # Поворот вправо
                    np.where(object_lost,  # Если объект потерян
                        np.where(old_act == 1,
                            1,
                            np.where(old_act == 2,
                                2,
                                np.where(np.abs(velocity_x)+np.abs(velocity_z) > 7,
                                10,
                                np.random.choice([1, 2], size=bbox_w.shape, p=[0.5, 0.5])))),  # Рандомный выбор поворота
                        np.where((bbox_x < 38) & (bbox_x > 0),
                            8,  # Если bbox_x меньше 48 и больше 0, выбираем действие 2
                            np.where((bbox_x > 218) & (bbox_x > 0),
                                7,  # Если bbox_x больше 208 и больше 0, выбираем действие 1
                                np.where(angularVelocity_y > 0.4,  # Если угловая скорость больше 0.5
                                    9,  # Поворот влево
                                    np.where(angularVelocity_y < -0.4,  # Если угловая скорость меньше -0.5
                                        9,  # Поворот вправо
                                        np.where((bbox_w * bbox_h < 3000),  # Если площадь объекта мала
                                            3,  # Движение вперед
                                            np.where((altitude > 2) & (altitude < 12),  # Если площадь объекта мала
                                                9,     # Иначе движение назад
                                                4
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    actions = torch.from_numpy(actions).type(torch.long)
    # target_vector = torch.nn.functional.one_hot(actions, 7).float()
    target_vector = torch.tensor([
        [0.9, -0.5, -0.5, 0.0, 0.0, 0.0, 0.],  #0
        [0.1, 2, -0.5, 0, 0.1, 0.2, 0.2],  #1
        [0.1, -0.5, 2, 0, 0.1, 0.2, 0.2],  #2
        [0., -1, -1, 2, -0.2, 0.1, -0.5],  #3
        [-0.1, -0, -0, 0.1, 1.5, 0.5, 0.],  #4
        [-0.0, -0.1, -0.1, 0.2, 0.0, 0.9, 0.],  #5
        [0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 1],  #6
        [0, 1.2, -0.5, .5, -0.1, 0, 0.9],  #1_7
        [0, -0.5, 1.2, .5, -0.1, 0, 0.9],  #2_8
        [-0.2, 0., 0, 0.7, 0.1, -0.2, 1.5],  #6_9
        [0.2, 0.8, 0.8, -0.1, 0.8, 0.2, 0.2]  #4_10
    ])
    target_vector = target_vector[actions]
    actions_targ = actions
    probs1, hid = actor(states, b_vec, hid)
    probs = F.softmax(probs1, dim=-1)
    m = Categorical(probs)
    # criterion = torch.nn.HuberLoss(reduction='none')  # Используем reduction='none' для получения вектора потерь
    criterion = torch.nn.MSELoss(reduction='none')  # Используем reduction='none' для получения вектора потерь

    losses = criterion(probs1, target_vector * 3)
    # optimizer = torch.optim.AdamW(actor.parameters(), lr=0.00005)
    losses = losses.mean(dim=1)
    # optimizer.zero_grad()
    # for loss in losses:
    #     loss.backward(retain_graph=True)
    # optimizer.step()
    # loss = 0
    actions = m.sample()
    #
    # actions = probs.argmax(1)
    return actions, probs, m.log_prob(actions), m.entropy(), hid, losses.mean()

def select_action_ddqn(states, b_vec, policy_net, rewards, device='cpu'):
    batch_size = states.size(0)  # Размер батча
    q_values, _boxx = policy_net(states, b_vec)  # Получение Q-значений для всего батча
    best_actions = q_values.argmax(dim=-1)  # Индексы действий с наивысшими Q-значениями
    epsilon_values = torch.linspace(0, 0.9, steps=batch_size, device=device)
    random_actions = torch.randint(0, action_dim, (batch_size,), device=device, dtype=torch.long)
    choose_random = torch.rand(batch_size, device=device) > epsilon_values
    actions = torch.where(choose_random, random_actions, best_actions)
    return best_actions, q_values, _boxx

def process_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
        lines = file_content.split('\n')
        matrix_lines = lines[:-1]
        new_states = np.array([list(map(float, line.split(','))) for line in matrix_lines])
        return new_states.transpose()

def bb_vectors(state, original_size=256, new_size=128):
    state_without_bb = torch.cat((state[:, :10], state[:, 15:21]), dim=1)  # Concatenate to create a tuple
    index = torch.linspace(-2, 2, 40)  # Create an index for values in range -2 to 2 with 40 steps

    # Adjust the first column (assuming it's in range 0 to 32) to be within 0 to 40
    state_without_bb[:, 0] = torch.clamp((state_without_bb[:, 0] / 0.8).round(), 0, 40).int()

    # Adjust other columns (assuming they're in range -2 to 2) to be within 0 to 40
    # Mapping -2 to 0, 0 to 20, and 2 to 40
    state_without_bb[:, 1:] = torch.clamp(((state_without_bb[:, 1:] - (-2)) / (2 - (-2)) * 40).round(), 0, 40).int()

    # Processing bounding boxes
    bb = state[:, 10:14]
    bounding_boxes = bb.round().int()
    invalid_mask = torch.any(bounding_boxes == -1, dim=1)
    scale_factor = new_size / original_size
    bounding_boxes = (bounding_boxes.to(torch.float32) * scale_factor).int()
    y_start, x_start, h, w = bounding_boxes.t()

    y_indices = torch.arange(0, new_size).unsqueeze(0).expand(bounding_boxes.size(0), -1)
    x_indices = torch.arange(0, new_size).unsqueeze(0).expand(bounding_boxes.size(0), -1)
    height_vectors = ((y_indices >= y_start.unsqueeze(1)) & (y_indices < (y_start + h).unsqueeze(1))).int()
    width_vectors = ((x_indices >= x_start.unsqueeze(1)) & (x_indices < (x_start + w).unsqueeze(1))).int()
    height_vectors[invalid_mask] = 0
    width_vectors[invalid_mask] = 0

    bb_vector = torch.cat((height_vectors, width_vectors), dim=1)
    return state_without_bb, bb_vector


def send_data(new_states, keys, recent_actions, b_boxes, recent_rewards, recent_hid, alpha_bb=0.25):
    state = torch.from_numpy(new_states).float()
    state_dict = dict(zip(keys, new_states))
    c = ['DisTerr', 'Roll', 'Pitch', 'Yaw', 'RotX', 'RotY', 'RotZ', 'angularVelocityX',  'angularVelocityY', 'angularVelocityZ',  'Bounding_box_X', 'Bounding_box_Y', 'Bounding_box_W', 'Bounding_box_H', 'time', 'Acceleration_x', 'Acceleration_y', 'Acceleration_z', 'AngularAcceleratio_x', 'AngularAcceleratio_y', 'AngularAcceleratio_z']

    state[1] /= 9
    state[2] /= 9
    state[3] /= 90
    state[4] /= 4
    state[5] /= 4
    state[6] /= 4
    state[7] *= 4
    state[8] *= 4
    state[9] *= 4
    state[15] /= 4
    state[16] /= 4
    state[17] /= 4
    state[20] *= 4

    do = (state[14] < 0.7).int()
    state_t = torch.transpose(state, 0, 1)
    state_only, bb_vec = bb_vectors(state_t)

    rewards = reward_function(state_dict, bb_vec.tolist())

    new_rewards_np = np.round(np.array(rewards), 4)
    new_rewards = torch.from_numpy(new_rewards_np).float()
    recent_rewards.append(new_rewards)

    recent_states.append(state_only)


    b_boxes.append(bb_vec)
    b_box = list(b_boxes)
    b = b_box[-1] - alpha_bb * b_box[-2] - alpha_bb**2 * b_box[-3] - alpha_bb**3 * b_box[-4]

    rec_act = torch.stack(list(recent_actions), dim=-1)[:, -1].unsqueeze(-1)
    # rec_b_b = torch.stack(b, dim=-1)[:, :, -1]
    rec_b_b = torch.from_numpy(np.array(b))
    rec_states = torch.stack(list(recent_states), dim=-1)[:, :, -1]
    rec_states = torch.cat([rec_states, rec_act], dim=-1)


    actions_new, new_probs, log_prob, entropy, hidden_state_w, loss = generate_action_number(rec_states, state_dict, rec_b_b)
    # actions_new = recent_actions[-1]
    # log_prob = recent_log_probs[-1]
    # entropy = entropys[-1]
    # new_probs = recent_probs[-1]
    print(actions_new.tolist(), '\n')
    # print('actions_targ: ', actions_targ)
    # print('vel_reward: ', vel_reward[5])
    # print('angular_velocity_reward: ', angular_velocity_reward[5])
    time = (state[14]*2.5).int()
    print('time: ', time)
    print('bb_vec: ', torch.sum(bb_vec, dim=-1).long())
    print('state_only: ', state_only[5])
    print('loss: ', loss)
    df = pd.DataFrame(state_dict)
    df = df[keys]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print('DisTerr and Bounding_box\n', df[['DisTerr', 'Bounding_box_X', 'Bounding_box_Y', 'Bounding_box_W', 'Bounding_box_H']])
    # print('DisTerr and Bounding_box\n', df[['Acceleration_x', 'Acceleration_y', 'Acceleration_z', 'AngularAcceleratio_x', 'AngularAcceleratio_y', 'AngularAcceleratio_z']])
    print('rewards: ',  new_rewards/3)
    return recent_states, b_boxes, recent_rewards, actions_new, do,  new_probs, log_prob, entropy, recent_hid, time, loss


gamma = 0.9
def discount_rewards(reward_tensors, gamma=0.9):
    rewards = torch.stack(reward_tensors)
    discount = torch.tensor([gamma**i for i in range(rewards.size(0))]).view(-1, 1)
    discount = torch.flip(discount, [0])
    discounted_rewards = rewards * discount
    discounted_rewards = discounted_rewards.sum(0)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
    return discounted_rewards

def train(keys, b_boxes, recent_rewards, recent_hid):
    file_path = sys.argv[1]
    if file_path:
        new_states = process_file(file_path)
    else:
        return print([0]*12)

    recent_states, b_boxes, recent_rewards, next_actions, do, new_probs, log_prob, entropy, recent_hid, time, loss = send_data(new_states, keys, recent_actions, b_boxes, recent_rewards, recent_hid)

    recent_probs.append(new_probs)
    recent_actions.append(next_actions)
    done.append(do)
    entropys.append(entropy)
    recent_log_probs.append(log_prob)

    rec_state = torch.stack(list(recent_states), dim=-1)
    rec_b_boxes = torch.stack(list(b_boxes), dim=-1)
    rec_act = torch.stack(list(recent_actions), dim=-1)

    actor_loss_mean = []
    critic_loss_mean = []
    actor.optimizer.zero_grad()  # Обнуляем градиенты актора перед аккумулированием
    critic.optimizer.zero_grad()  # Обнуляем градиенты критика перед аккумулированием
    alpha_bb = 0.25
    b_box = list(b_boxes)
    for i in range(13, 14):
        rewards = recent_rewards[i]
        # rewards_next = list(recent_rewards)[i+1:i+3]
        # print(discount_rewards(torch.tensor(np.array(rewards_next), dtype=torch.float32)))
        rewards_next = recent_rewards[i+1]
        rewards_next_ = recent_rewards[i+2]
        rewards_next__ = recent_rewards[i+3]

        done_i = done[i]
        log_prob = recent_log_probs[i-1]
        entropy = entropys[i-1]
        # probs = recent_probs[i-1]
        actions =recent_actions[i-1].unsqueeze(-1)

        act = rec_act[:, i-1].unsqueeze(-1)
        b = b_box[i] - alpha_bb * b_box[i-1] - alpha_bb**2 * b_box[i-2] - alpha_bb**3 * b_box[i-3]
        boxes = torch.from_numpy(np.array(b))
        # boxes = rec_b_boxes[:, :, i]
        states = rec_state[:, :, i]
        states = torch.cat([states, act], dim=-1)


        new_act = rec_act[:, i].unsqueeze(-1)
        b = b_box[i+1] - alpha_bb * b_box[i] - alpha_bb**2 * b_box[i-1] - alpha_bb**3 * b_box[i-2]
        boxes_new = torch.from_numpy(np.array(b))
        # boxes_new = rec_b_boxes[:, :, i+1]
        states_new = rec_state[:, :, i+1]
        states_new = torch.cat([states_new, new_act], dim=-1)

        values = critic(states, boxes)
        values = values.squeeze()
        with torch.no_grad():
            next_values = critic(states_new, boxes_new)
            next_values = next_values.squeeze()
        advantages = (rewards + (1 - done_i) * (gamma * rewards_next + gamma**2*rewards_next_ + gamma**3*rewards_next__) - values).detach()
        actor_loss = (-log_prob * advantages) - 0.001 * entropy

        critic_loss = F.mse_loss(values, rewards + (1 - done_i) * (gamma * rewards_next + gamma**2*rewards_next_ + gamma**3*rewards_next__).detach(), reduction='none')

        actor_loss_mean.append(actor_loss)
        critic_loss_mean.append(critic_loss)
    actor_loss_mean = torch.stack(actor_loss_mean, dim=-1).mean(dim=-1)
    critic_loss_mean = torch.stack(critic_loss_mean, dim=-1).mean(dim=-1)
    print('actor_loss: ', actor_loss_mean.sum())
    print('critic_loss: ', critic_loss_mean.sum())
    # total_params_actor = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    # total_params_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    #
    # print("Total trainable parameters in Actor:", total_params_actor)
    # print("Total trainable parameters in Critic:", total_params_critic)

    actor.optimizer.zero_grad()
    critic.optimizer.zero_grad()
    for i in range(12):
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1)
        actor_loss_mean[i].backward(retain_graph=True)  # Обратное распространение для каждого дрона
        critic_loss_mean[i].backward(retain_graph=True)  # Обратное распространение для каждого дрона
        if torch.isnan(loss) or torch.isinf(loss):
            print("Skipping backpropagation due to non-finite loss")
            continue
    actor.optimizer.step()
    critic.optimizer.step()
    new_probs = torch.round(new_probs*1000)/1000
    print('advantages: ', advantages)
    print('\n', next_values, '\n', next_actions[5].tolist(), new_probs[5])
    print('prob: ', new_probs, int(time[0]) % 80)
    # total_params_actor = count_trainable_params(actor)
    #
    # ffn_params = set(p for p in critic.ffn.parameters() or critic.x_bb.parameters())
    # total_params_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad and p not in ffn_params)
    #
    # print("Total trainable parameters in Actor:", total_params_actor)
    # print("Total trainable parameters in Critic:", total_params_critic)

    if int(time[0]) % 80 == 0:
        loss_act.append(actor_loss_mean.sum())
        loss_crit.append(critic_loss_mean.sum())
        sum_rewards.append(torch.mean(torch.stack(list(recent_rewards), dim=-1)[-1]))
        loss_.append(loss)
        print('100: save')
    if tf:
        torch.save(actor.state_dict(), '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/Actor7.pt')
        torch.save(critic.state_dict(), '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/Critic7.pt')
        # torch.save(ddqn.state_dict(), '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/ddqn7.pt')
        print('save')
    save_data(recent_actions, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_actions.pkl')
    save_data(recent_states, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_states.pkl')
    save_data(recent_rewards, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_rewards.pkl')
    save_data(recent_probs, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_probs.pkl')
    save_data(recent_values, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_values.pkl')
    save_data(recent_log_probs, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_log_probs.pkl')
    save_data(done, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/done.pkl')
    save_data(entropys, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/entropys.pkl')
    save_data(b_boxes, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/b_boxes.pkl')
    save_data(recent_q_values, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_q_values.pkl')
    save_data(rec_boxx, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/rec_boxx.pkl')
    save_data(recent_hid, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/recent_hid.pkl')
    save_data(loss_act, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_act.pkl')
    save_data(loss_crit, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_crit.pkl')
    save_data(sum_rewards, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/sum_rewards.pkl')
    save_data(loss_, '/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/data/loss_.pkl')

if __name__ == "__main__":
    keys = ['DisTerr',
            'Roll', 'Pitch', 'Yaw',
            'RotX', 'RotY', 'RotZ',
            'angularVelocityX',  'angularVelocityY', 'angularVelocityZ',
            'Bounding_box_X', 'Bounding_box_Y', 'Bounding_box_W', 'Bounding_box_H',
            'time',
            'Acceleration_x', 'Acceleration_y', 'Acceleration_z',
            'AngularAcceleratio_x', 'AngularAcceleratio_y', 'AngularAcceleratio_z']

    train(keys, b_boxes, recent_rewards, recent_hid)
