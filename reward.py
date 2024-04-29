# import torch
# model_file = "yolov5s.pt"  # Changed the file extension to '.pt'
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import pandas as pd
import numpy as np


def reward_function(state_dict, bb_vec):
    # Извлечение данных
    bbox_x = np.array(state_dict['Bounding_box_X'])
    bbox_y = np.array(state_dict['Bounding_box_Y'])
    bbox_w = np.array(state_dict['Bounding_box_W'])
    bbox_h = np.array(state_dict['Bounding_box_H'])
    altitude = np.array(state_dict['DisTerr'])
    angular_velocity_z = np.array(state_dict['angularVelocityZ'])
    v_x = np.array(state_dict['RotX'])
    v_z = np.array(state_dict['RotZ'])
    object_lost = (bbox_w == -1) | (bbox_h == -1)

    target_altitude = 15 # Целевая высота полёта

    # Размеры изображения (предположим)
    img_width, img_height = 256, 256
    # Центрирование объекта
    center_x = bbox_x + bbox_w / 2
    center_y = bbox_y + bbox_h / 2
    center_target_x = img_width / 2
    center_target_y = img_height / 2

    # Расчет смещения центра от центра кадра
    center_reward = np.exp(-np.abs(center_x - center_target_x) * 0.02) * np.exp(-np.abs(center_y - center_target_y) * 0.02)
    center_reward = np.round(center_reward, 4)
    # Вознаграждение за центрирование объекта
    # center_reward = np.exp(-center_offset / 100)  # Уменьшенная чувствительность смещения

    # Вознаграждение за размер объекта
    size_reward = np.round(np.where((bbox_w > -1) & (bbox_h > -1), np.exp(-np.abs(bbox_w * bbox_h - 1220) / 3660), -2), 4)  # Удвоенный вес
    # так же возможен такой вариант z\ =\ \exp\left(-\left(\frac{yx}{1000}-1\right)^{4}\cdot4\right)

    vel_reward = np.round(np.where((size_reward < 15*15) & (bbox_w > -1) & (bbox_h > -1), 0, np.exp(-((v_z-5)**2)*0.5)*np.exp(-((v_z-5)**2)*0.5)), 4)
    # Вознаграждение за высоту

    altitude_reward = np.exp(-np.abs(altitude - target_altitude) / 8)-0.2

    # Вознаграждение за угловую скорость при потере объекта
    angular_velocity_reward = np.where(object_lost, np.exp(-np.abs(angular_velocity_z-0.5) * 7), 0)

    # Штраф за потерю объекта
    loss_penalty = np.where((bbox_w == -1) | (bbox_h == -1), -2, 0)
    # Штраф за слишком низкий полёт
    low_altitude_penalty = np.where(altitude < 1, -0.2, 0.2)  # Сильный штраф за полёт ниже 1 метра

    # Комбинирование вознаграждений
    total_reward = (center_reward + size_reward+ altitude_reward + loss_penalty +
                    angular_velocity_reward + low_altitude_penalty + vel_reward)

    return total_reward.tolist()




