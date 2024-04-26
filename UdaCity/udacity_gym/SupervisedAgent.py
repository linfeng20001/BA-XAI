import os
import tensorflow as tf

from udacity_gym.gym import gym
import numpy as np
from tensorflow.keras.models import load_model
from udacity_gym.action import UdacityAction

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#from agents.agent import Agent
from udacity_gym.utils import preprocess


class SupervisedAgent():

    def __init__(
            self,
            env: gym.Env,
            model_path: str,
            max_speed: int,
            min_speed: int,
            predict_throttle: bool = False,
    ):

        assert os.path.exists(model_path), 'Model path {} not found'.format(model_path)
        with tf.device('cpu:0'):
            self.model = load_model(filepath=model_path)

        self.predict_throttle = predict_throttle
        self.model_path = model_path

        self.max_speed = max_speed
        self.min_speed = min_speed

    def predict(self, obs: np.ndarray, speed: float = 0.0) -> np.ndarray:
        #obs = preprocess(image=obs, if_yuv=True)
        #obs = preprocess(image=obs)
        # the model expects 4D array
        obs = np.array([obs])

        if self.predict_throttle:
            action = self.model.predict(obs, batch_size=1, verbose=0)
            steering, throttle = action[0], action[1]
        else:
            steering = float(self.model.predict(obs, batch_size=1, verbose=0))

            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            # steering = self.change_steering(steering=steering)
            throttle = np.clip(a=1.0 - steering ** 2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

        #return UdacityAction(steering_angle=steering, throttle=throttle)
        return np.asarray([steering, throttle])

    '''
    def action(self, observation: UdacityObservation, *args, **kwargs):
        error = observation.cte
        diff_err = error - self.prev_error

        # Calculate steering angle
        steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)
        steering_angle = max(-1, min(steering_angle, 1))

        # Calculate throttle
        throttle = 0.25 - (np.abs(error) * 0.1) * 0.55 if observation.speed < 100 else 0.0

        # Save error for next prediction
        self.total_error += error
        self.total_error = self.total_error * 0.99
        self.prev_error = error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)
        '''