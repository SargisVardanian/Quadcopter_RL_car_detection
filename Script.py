import airsim
import numpy as np
import cv2
import torch


client = airsim.MultirotorClient()
print('client', client)
client.confirmConnection()

state = client.getMultirotorState()
print("Состояние мультиротора:", state)

client.enableApiControl(True)
client.armDisarm(True)

camera_name = "0"
image_request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene)
response = client.simGetImages([image_request])

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

initial_ground_altitude = client.getMultirotorState().kinematics_estimated.position.z_val
def perform_action(client, action):
    client.moveByMotorPWMsAsync(action[0], action[1], action[2], action[3], action[4]).join()
    initial_state = get_initial_state()
    response = client.simGetImages([image_request])[0]

    image_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    try:
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model([image_rgb])
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
        except Exception as e:
            print(f'Error drawing bounding boxes: {e}')
            x, y, w, h = 1000, 1000, 1000, 1000
        # cv2.imshow('YOLOv5 Detection', image)
        return initial_state, [x, y, w, h]
    except:
        print("Error")


def get_initial_state():
    state = client.getMultirotorState().kinematics_estimated
    kinematics = client.simGetGroundTruthKinematics()
    z = kinematics.position.z_val
    orientation = state.orientation
    pitch, roll, yaw = airsim.to_eularian_angles(orientation)
    vx, vy, vz = state.linear_velocity
    return [z, pitch, roll, yaw, vx, vy, vz]

def check_if_episode_is_done(step_count, max_steps):
    return step_count >= max_steps
