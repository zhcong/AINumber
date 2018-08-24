# -*- coding:utf8 -*-
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

source_image_dir = 'image'
source_image_size = 16
target_image_size = 28

source_image = []
source_file_list = os.listdir(source_image_dir)
source_file_list.sort()

for file in source_file_list:
    if not file.find('png') == -1:
        im = np.array(Image.open(os.path.join(source_image_dir, file)).convert('L'), dtype=np.int16)
        source_image.append(im)
mnist = input_data.read_data_sets("image")
target_image = mnist.train.images[:2000]
target_image = np.reshape(target_image, [target_image.shape[0], 28, 28])
target_image_label = mnist.train.labels[:2000]

# 卷积自编码器
learn_rate = 5
loop_num = 100

inputs = tf.placeholder(tf.float32, (1, source_image_size, source_image_size, 1), name='image_input')
targets = tf.placeholder(tf.float32, (target_image_size, target_image_size), name='image_output')
# 解码

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

loss = tf.losses.mean_squared_error(predictions=out, labels=targets)
tran_op = tf.train.AdadeltaOptimizer(learn_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # tf.train.Saver().restore(sess, "./model_save")

    print('working...')
    for i in range(loop_num):
        for t_i, target_im in zip(target_image_label, target_image):
            feed = {inputs: np.reshape(source_image[t_i], [1, source_image_size, source_image_size, 1]),
                    targets: target_im}
            sess.run(tran_op, feed_dict=feed)
        loss_ = sess.run(loss, feed_dict=feed)
        tf.train.Saver().save(sess, "./model_save")
        print('step %d, loss=%f' % (i, loss_))
