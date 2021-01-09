from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
import keras
import tensorflow as tf
from keras.utils import plot_model
tf.compat.v1.disable_eager_execution()
import numpy as np
import glob
import os

def modeling():
    FC_nums = 12089
    freeze_layers = 17
    image_size = 256
    num_classes = 3
    base_model = VGG19(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet'
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_nums, activation='relu')(x)
    prediction = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    model.summary()
    print("layer nums: ", len(model.layers))

    for layer in model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in model.layers[freeze_layers:]:
        layer.trainable = True
    return model

def ConvModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), input_shape=(256, 256, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3)
        # 将这种不激活的输出叫做logits， k分类就输出一个长度为k的张量，预测一个长度为3的张量，预测值最大的对应就是哪一类
    ])
    return model

def load_image(path, lable):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image,lable

def load_image2(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image

def training():
    print("[INFO] modeling")
    model = ConvModel()
    print("[INFO] training images...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # 对于多分类问题 最后一步输出没有激活 因此from_logits为True
        metrics=['acc']
    )
    train_count = len(train_path)
    test_count = len(test_path)
    steps_per_epoch = train_count//BATCH_SIZE
    validation_steps = test_count//BATCH_SIZE

    history = model.fit(
        train_ds, epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds,
        validation_steps=validation_steps
    )
    model.save(filepath='F:\\study\\DIP\\project', include_optimizer=False)

if __name__=='__main__':
    image_path = glob.glob('F:\\study\\DIP\\trainData\\slice-level\\training2\\*\\*.png')
    str_label = [image_p.split('\\')[-2] for image_p in image_path]
    label = []
    for item in str_label:
        label.append(float(item))
    random_index = np.random.permutation(len(image_path))
    # 进行数据和标签同时乱序
    image_path = np.array(image_path)[random_index]
    label = np.array(label)[random_index]
    # 总长度为8进行切片，前面为训练集，后面为测试集
    i = int(len(image_path) * 0.8)
    train_path = image_path[:i]
    train_label = label[:i]
    test_path = image_path[i:]
    test_label = label[i:]
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))

    autotune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(load_image, num_parallel_calls=autotune)
    test_ds = test_ds.map(load_image, num_parallel_calls=autotune)
    # 多线程读入,自己判断几个线程 这一系列操作还是由CPU进行

    BATCH_SIZE = 16
    train_ds = train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
    # 不停地输出，在内存中产生1个区域  ，随机生成其中的图片，shuffle不能太大
    test_ds = test_ds.batch(BATCH_SIZE)

    # training()
    history = keras.models.load_model('', compile = True)
    history.summary()
    plot_model(history, to_file='model.png')
    # loss, acc = history.evaluate(test_ds)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    ver_path = glob.glob('F:\\study\\DIP\\trainData\\testing\\*\\*\\*.png')
    ver_image = []
    img = load_image2(ver_path[0])
    img = tf.expand_dims(img, axis=0)

    # for path in ver_path:
    #     tmp_image = load_image2(path)
    #     tmp_image = tf.expand_dims(tmp_image, axis=0)
    #     tmp_image = preprocess_input(tmp_image)
    #     ver_image.append(tmp_image)
    # print(np.argmax(pred))
