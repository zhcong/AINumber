# -*- coding:utf8 -*-
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

source_image_dir = 'image'
source_image_size = 16
target_image_size = 28
# 读取图片至image
# 卷积自编码器
inputs = tf.placeholder(tf.float32, (1, source_image_size, source_image_size, 1), name='image_input')

# 扩张成28*28
unpool = tf.image.resize_images(inputs, size=(target_image_size, target_image_size),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 反卷积，得到５个反卷积图像
conv = tf.layers.conv2d_transpose(inputs=unpool, filters=10, kernel_size=(3, 3), strides=1, padding='same',
                                  activation=tf.nn.sigmoid)
conv = conv[0]
# conv1 1x28x28x5
W_image = tf.Variable(initial_value=tf.random_normal(conv.shape), dtype=tf.float32)
W_merge = tf.ones(shape=[conv.shape[0], conv.shape[2], 1])
V_merge = tf.random_normal(shape=[int(conv.shape[0]), int(conv.shape[1]), ])
conv = tf.multiply(W_image, conv)
conv = tf.matmul(conv, W_merge)
conv = tf.reshape(conv, [conv.shape[0], conv.shape[1]])

out = tf.layers.dense(inputs=conv, units=conv.shape[0])
out = tf.sigmoid(out) * 255

source_image = []
source_file_list = os.listdir(source_image_dir)
source_file_list.sort()
for file in source_file_list:
    if not file.find('png') == -1:
        im = np.array(Image.open(os.path.join(source_image_dir, file)).convert('L'), dtype=np.int16)
        source_image.append(im)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 11 step
    tf.train.Saver().restore(sess, "model/model_save")

    for i, im in enumerate(source_image):
        feed = {inputs: np.reshape(im, [1, source_image_size, source_image_size, 1])}
        out_img = sess.run(out, feed_dict=feed)
        plt.subplot(5,4,i*2+1)
        plt.imshow(im, cmap='gray')
        plt.xlabel('from font')
        plt.subplot(5,4,i*2+2)
        plt.imshow(out_img, cmap='gray')
        plt.xlabel('from AI')
    plt.show()
