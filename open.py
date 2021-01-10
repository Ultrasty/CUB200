#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.models import *
import numpy as np
import glob

print('tensorflow_version:{}'.format(tf.__version__))

imgs_path = glob.glob('CTtest/*/*.jpg')

model = load_model('model.h5')

index_to_label = dict({0: 'Cap', 1: 'Covid-19', 2: 'Normal'})


def load_and_process_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image


for n in range(90):
    test_tensor = load_and_process_image(imgs_path[n])
    test_tensor = tf.expand_dims(test_tensor, axis=0)
    pred = model.predict(test_tensor)
    print(index_to_label.get(np.argmax(pred)))
    print(imgs_path[n])
