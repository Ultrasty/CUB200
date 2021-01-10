#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import glob

print('tensorflow_version:{}'.format(tf.__version__))

imgs_path = glob.glob('CT/*/*/*.jpg')
all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]
label_names = np.unique(all_labels_name)
label_to_index = dict((name, i) for i, name in enumerate(label_names))
index_to_label = dict((v, k) for k, v in label_to_index.items())
all_labels = [label_to_index.get(name) for name in all_labels_name]
np.random.seed(2021)
random_index = np.random.permutation(len(imgs_path))

imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]

i = int(len(imgs_path) * 0.8)

train_path = imgs_path[:i]
train_labels = all_labels[:i]
test_path = imgs_path[i:]
test_labels = all_labels[i:]

train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))


def load_img(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32

train_ds = train_ds.repeat().shuffle(300).batch(BATCH_SIZE)

test_ds = test_ds.batch(BATCH_SIZE)

model = load_model('model.h5')

train_count = len(train_path)
test_count = len(test_path)

steps_per_epoch = train_count // BATCH_SIZE
validation_steps = test_count // BATCH_SIZE

try:
    history = model.fit(train_ds, epochs=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds,
                        validation_steps=validation_steps)
except Exception:
    model.save('model1.h5')


model.save('model1.h5')