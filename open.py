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


resultList = {}
result = {
    "pred_count_Cap": 0,
    "pred_count_Covid-19": 0,
    "pred_count_Normal": 0
}
for n in range(2000):
    try:
        path = imgs_path[n].split('\\')[1]
        if path not in resultList:
            resultList[path] = {
                "pred_count_Cap": 0,
                "pred_count_Covid-19": 0,
                "pred_count_Normal": 0
            }
        test_tensor = load_and_process_image(imgs_path[n])
        test_tensor = tf.expand_dims(test_tensor, axis=0)
        pred = model.predict(test_tensor)
        if index_to_label.get(np.argmax(pred)) == 'Cap':
            resultList[path]["pred_count_Cap"] = resultList[path]["pred_count_Cap"] + 1
        if index_to_label.get(np.argmax(pred)) == 'Covid-19':
            resultList[path]["pred_count_Covid-19"] = resultList[path]["pred_count_Covid-19"] + 1
        if index_to_label.get(np.argmax(pred)) == 'Normal':
            resultList[path]["pred_count_Normal"] = resultList[path]["pred_count_Normal"] + 1
    except Exception:
        print("一共有", n, "张图片")
        break
for key in resultList:
    print(key, ":", resultList[key])
