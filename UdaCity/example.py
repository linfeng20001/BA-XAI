import time

from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction, UdacityObservation
from udacity_gym import SupervisedAgent
from udacity_gym.agent import PIDUdacityAgent
from PIL import Image
import numpy as np
import os
import udacity_gym.utils as ut


def load_image(file_path: str) -> np.ndarray:
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array
'''
if __name__ == '__main__':
    def predict_images_in_folder(folder_path: str, agent):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                IMG = load_image(img_path)
                observation = UdacityObservation(IMG, IMG, (0, 0, 0), 1, 0, 12, 0, 1)
                #action = agent.action(observation)
                img = ut.resize(observation.input_image).astype("float32")
                steering, throttle = agent2.predict(img)
                print("Image:", filename)
                print("Steering Angle:", steering)
                print("Throttle:", throttle)
                print("------------")
    img_path = "C:/Unet/track1/normal/IMG/"
    model_path = "C:/Unet/track1-track1-udacity-dave2-001-final.h5"
    agent = PIDUdacityAgent(kp=0.055, kd=0.75, ki=0.000001)
    agent2 = SupervisedAgent.SupervisedAgent("", model_path, 12, 0)
    predict_images_in_folder(img_path, agent2)
    exit()

if __name__ == '__main__':  #single img
    img_path = "C:/Unet/track1/normal/IMG/center_2024_04_11_11_46_49_070.png"
    model_path = "C:/Unet/track1-track1-udacity-dave2-001-final.h5"
    IMG = load_image(img_path)
    observation = UdacityObservation(IMG, IMG, (0,0,0), 1, 0, 12, 0, 1)
    agent = PIDUdacityAgent(kp=0.055, kd=0.75, ki=0.000001)
    agent2 = SupervisedAgent.SupervisedAgent("", model_path, 12, 0)
    #action = agent.action(observation)
    img = ut.resize(observation.input_image).astype("float32")
    action1, action2 = agent2.predict(img)
    print(action1)
    print(action2)
    exit()
'''
if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "C:/Users/Linfe/OneDrive/Desktop/simulation/self_driving_car_nanodegree_program.exe"
    model_path = "C:/Unet/track1-track1-udacity-dave2-001-final.h5"
    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=simulator_exe_path,
        host=host,
        port=port,
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
        track="lake",
    )

    simulator.start()
    observation, _ = env.reset(track="lake")

    # Wait for environment to set up
    while observation.input_image is None or observation.input_image.sum() == 0:
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)
        break
    agent2 = SupervisedAgent.SupervisedAgent(env, model_path, 12, 0)
    agent = PIDUdacityAgent(kp=0.055, kd=0.75, ki=0.000001)

    # Interacting with the gym environment


    for _ in range(20000000):
        time.sleep(4)
        '''
        print(observation.input_image.shape)
        img = ut.resize(observation.input_image).astype("float32")
        steering, throttle = agent2.predict(img)
        action = UdacityAction(steering,throttle)
        '''
        action = agent.action(observation)

        last_observation = observation

        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    simulator.close()
    env.close()
    print("Experiment concluded.")

