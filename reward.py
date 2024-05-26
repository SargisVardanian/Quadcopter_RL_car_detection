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

    target_altitude = 25 # Целевая высота полёта

    # Размеры изображения (предположим)
    img_width, img_height = 256, 256
    # Центрирование объекта
    center_x = bbox_x + bbox_w / 2
    center_y = bbox_y + bbox_h / 2
    center_target_x = img_width / 2
    center_target_y = img_height / 2

    # Расчет смещения центра от центра кадра
    center_reward = np.exp(-((center_x - center_target_x)**2 + (center_y - center_target_y)**2) * 0.0002) - 0.2
    center_reward = np.round(center_reward, 4)

    # Вознаграждение за размер объекта
    size_reward = np.round(np.where((bbox_w > -1) & (bbox_h > -1), np.exp(-np.abs(bbox_w * bbox_h - 2505) / 3050), -1), 4)  # Удвоенный вес

    v_r = np.exp(-((-np.abs(v_x)-7)**2)*0.02) + np.exp(-((-np.abs(v_z)-7)**2)*0.02)
    vel_reward = np.round(np.where((bbox_w > -1) & (bbox_h > -1),
                                   np.where((bbox_w * bbox_h > 70*70), 0, v_r), -v_r), 4)

    # altitude_reward = np.exp(-(altitude - target_altitude)**4 / 25000)-0.6

    # Вознаграждение за угловую скорость при потере объекта
    angular_velocity_reward = np.where(object_lost, np.exp(-np.abs(np.abs(angular_velocity_z) -0.5) * 6), 0)

    # Штраф за потерю объекта
    loss_penalty = np.where((bbox_w == -1) | (bbox_h == -1), -2, 0)

    # Штраф за слишком низкий полёт
    low_altitude_penalty = np.where(altitude < 3,
                                    -0.7,
                                    np.where(
                                        altitude > 50,
                                        -0.7,
                                        0.0
                                    ))  # Сильный штраф за полёт ниже 1 метра

    # Комбинирование вознаграждений
    total_reward = (center_reward*2 + size_reward * 8 + loss_penalty +
                    angular_velocity_reward*2 + low_altitude_penalty + vel_reward)

    return total_reward.tolist(), vel_reward




