import datetime
import os
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from ThirdEye.ase22.config import Config

from ThirdEye.ase22.utils import get_driving_styles
from ThirdEye.ase22.utils_models import *

import os
import tensorflow as tf

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Adjust channels as needed (RGB or grayscale)
    image = tf.image.resize(image, (80, 160))  # Adjust size as needed
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image

if __name__ == '__main__':
    model_PATH = "/mnt/c/Unet/track1-track1-udacity-dave2-001-final.h5"
    IMG_PATH = "/mnt/c/Unet/track1/normal/IMG/center_2024_04_11_11_46_55_603.png"
    model = tf.keras.models.load_model(model_PATH)
    model.summary()

    processed_image = preprocess_image(IMG_PATH)

    processed_image = tf.expand_dims(processed_image, axis=0)

    print(processed_image.shape)
    predictions = model.predict(processed_image)
    print(predictions.shape)
    predicted_class = tf.argmax(predictions, axis=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(processed_image[0])  # Assuming processed_image is a batch, so we take the first element
    plt.title(f'Predicted class: {predicted_class.numpy()}')
    plt.axis('off')
    plt.show()


