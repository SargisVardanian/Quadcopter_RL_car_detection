import airsim
import numpy as np
import cv2
import torch







def perform_action(client, action, start=True):

    if start:
        act_val = action_dict[action]
    else:
        act_val = [0.25, 0.25, 0.25, 0.25]
    client.moveByMotorPWMsAsync(act_val[0], act_val[1], act_val[2], act_val[3], 0.7).join()

    # client.moveByMotorPWMsAsync(0.1, 0.2, 0.1, 0.2, 0.7).join()
    initial_state = get_initial_state()
    response = client.simGetImages([image_request])[0]

    image_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    try:
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model([image_rgb])
        boxes = results.xyxy[0].cpu().numpy()
        print(image.shape)
        try:
            if len(boxes) > 0:
                for box in boxes:
                    x_min, y_min, x_max, y_max, confidence, class_id = box
                    x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print('image', x, y, w, h)
            else:
                print('No car detected')
                x, y, w, h = 1000, 1000, 1000, 1000
        except Exception as e:
            print(f'Error drawing bounding boxes: {e}')
            x, y, w, h = 1000, 1000, 1000, 1000
        # cv2.imshow('YOLOv5 Detection', image)
        return initial_state, [x, y, w, h]
    except:
        print("Error")


def check_if_episode_is_done(step_count, max_steps):
    return step_count >= max_steps


class Environment:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.client = airsim.MultirotorClient()
        print('client', self.client)
        self.client.confirmConnection()
        state = self.client.getMultirotorState()
        print("Состояние мультиротора:", state)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        camera_name = "0"
        self.image_request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene)
        self.response = self.client.simGetImages([self.image_request])
        self.initial_ground_altitude = self.client.getMultirotorState().kinematics_estimated.position.z_val
        self.act_val = np.array([0.25, 0.25, 0.25, 0.25])
        self.action_dict = {
            0: [0.05, 0.05, 0.05, 0.05],
            1: [-0.05, -0.05, -0.05, -0.05],
            2: [-0.05, -0.05, 0.05, 0.05],
            3: [0.05, 0.05, -0.05, -0.05],
            4: [-0.05, 0.05, -0.05, 0.05],
            5: [0.05, -0.05, -0.05, 0.05],
            6: [0.4, 0.4, 0.4, 0.4],  # const
            7: [0.3, 0.45, 0.3, 0.45],  # const
            8: [0.45, 0.3, 0.45, 0.3],  # const
            9: [0.55, 0.45, 0.3, 0.3],  # const
            10: [0.3, 0.3, 0.45, 0.45],  # const
            11: [0.1, -0.033, -0.033, -0.033],
            12: [-0.033, 0.1, -0.033, -0.033],
            13: [-0.033, -0.033, 0.1, -0.033],
            14: [-0.033, -0.033, -0.033, 0.1],
            15: [0.4, 0.0, 0.0, 0.0],  # partly const
            16: [0.0, 0.4, 0.0, 0.0],  # partly const
            17: [0.0, 0.0, 0.4, 0.0],  # partly const
            18: [0.0, 0.0, 0.0, 0.4]  # partly const
        }

        self.box_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

    def step(self, action):
        if action <= 10 and action >= 6:
            self.act_val = self.action_dict[action]
        elif action == 15:
            self.act_val[0] = self.action_dict[action][0]
        elif action == 16:
            self.act_val[1] = self.action_dict[action][1]
        elif action == 17:
            self.act_val[2] = self.action_dict[action][2]
        elif action == 18:
            self.act_val[3] = self.action_dict[action][3]
        else:
            act_val = self.action_dict[action]
            self.act_val += np.array(act_val)
        if np.sum(self.act_val) <= 0.4:
            self.act_val = np.array([0.25, 0.25, 0.25, 0.25])
        self.client.moveByMotorPWMsAsync(self.act_val[0], self.act_val[1], self.act_val[2], self.act_val[3], 0.7).join()
        image_data = np.frombuffer(self.response.image_data_uint8, dtype=np.uint8)
        initial_state = self.get_initial_state()
        try:
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.box_model([image_rgb])
            boxes = results.xyxy[0].cpu().numpy()
            try:
                if len(boxes) > 0:
                    for box in boxes:
                        x_min, y_min, x_max, y_max, confidence, class_id = box
                        x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print('image', x, y, w, h)
                else:
                    print('No car detected')
                    x, y, w, h = 1000, 1000, 1000, 1000
        except:
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            x, y, w, h = 1000, 1000, 1000, 1000

    def get_initial_state(self):
        state = self.client.getMultirotorState().kinematics_estimated
        kinematics = self.client.simGetGroundTruthKinematics()
        z = kinematics.position.z_val
        orientation = state.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation)
        vx, vy, vz = state.linear_velocity
        return [z, pitch, roll, yaw, vx, vy, vz]

    def reward(self):
        state = self.get_initial_state()
        z, pitch, roll, yaw, vx, vy, vz = state
        x, y, w, h = bounding_box
        if w == 1000 and h == 1000:
            no_box_rew = -2.0  # Присвойте подходящее отрицательное значение
        else:
            no_box_rew = 0
        target_x, target_y = 72, 128

        size_penalty = max(0, abs(w * h - 800) / 800)

        center_penalty = abs(x + w / 2 - target_x) / 72 + abs(y + h / 2 - target_y) / 128

        # Простое вознаграждение за сохранение высоты
        # height_reward = max(0, 1 - abs(z - desired_height) / desired_height)

        # Пример: штраф за сильные наклоны
        # pitch_roll_penalty = max(0, abs(pitch) + abs(roll) - self.max_allowed_pitch_roll)

        # Увеличь надобность в движении (увеличь вес speed_reward)
        speed = math.sqrt(vz ** 2 + vy ** 2 + vx ** 2)
        if speed < 2:
            speed_reward = -1
        else:
            speed_reward = 1

        total_reward = no_box_rew + speed_reward - size_penalty - center_penalty

        return total_reward

    def step(self, state):
        action = self.act(state)
        print('action', action)
        new_state, bounding_box = perform_action(client, action)
        # print(f'new_state{new_state}, bounding_box{bounding_box}')

        reward = self.reward(new_state, bounding_box)
        self.remember(state, action, reward, new_state, False)
        return action, reward


env = Environment(state_dim, action_dim)
action, reward = env.step(state)
