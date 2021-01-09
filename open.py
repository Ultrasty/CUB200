#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import numpy as np
import glob

print('tensorflow_version:{}'.format(tf.__version__))

imgs_path = glob.glob('birds/*/*.jpg')

all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]

label_names = np.unique(all_labels_name)

label_to_index = dict((name, i) for i, name in enumerate(label_names))

index_to_label = dict((v, k) for k, v in label_to_index.items())

i = int(len(imgs_path) * 0.8)

model = load_model('model.h5')


def load_and_process_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image


for n in range(90, 150):
    test_tensor = load_and_process_image(imgs_path[n])
    test_tensor = tf.expand_dims(test_tensor, axis=0)
    pred = model.predict(test_tensor)
    print(index_to_label.get(np.argmax(pred)))
    print(imgs_path[n])
