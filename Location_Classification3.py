# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from main_pre import get_test_data_another,get_train_data_another, accurances_train,accurances_test
from tensorflow.examples.tutorials.mnist import input_data


x_train_input, x_train_label = get_train_data_another()
x_test_input, x_test_label = get_test_data_another()

#载入数据集
# mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 88
#计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([2, 8], stddev=0.1))
b1 = tf.Variable(tf.zeros([8]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([8, 1], stddev=0.1))
b2 = tf.Variable(tf.zeros([1]) + 0.1)
prediction = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
# prediction = tf.matmul(L1, W2) + b2

# W3 = tf.Variable(tf.truncated_normal([6, 1], stddev=0.1))
# b3 = tf.Variable(tf.zeros([1]) + 0.1)
# prediction = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
# prediction = tf.matmul(L2, W3) + b3



# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# # 结果存放在一个布尔型列表中
correct_prediction = tf.equal(y, prediction)  # argmax返回一维张量中最大的值所在的位置
#
# # 求准确率
accuracy = tf.cast(correct_prediction, tf.float32)
# accuracy_test = tf.count_nonzero(accuracy)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(2176):
            # print x_train_input.shape, x_train_label.shape
            sess.run(train_step, feed_dict={x: x_train_input[batch*88:(batch + 1)*88,:], y: x_train_label[batch*88:(batch + 1)*88,:]})
        # print (correct_prediction.shape)
        test_acc = sess.run(prediction,feed_dict={x:x_test_input,y:x_test_label})
        train_acc = sess.run(prediction,feed_dict={x:x_train_input,y:x_train_label})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))
        # print test_acc, test_acc.shape
        for i in range(107712):
            if test_acc[i][0] >= 0.5:
                test_acc[i][0] = 1
            else:
                test_acc[i][0] = 0
        for i in range(191488):
            if train_acc[i][0] >= 0.5:
                train_acc[i][0] = 1
            else:
                train_acc[i][0] = 0


        b = 0
        for i in range(107712):
            if test_acc[i][0] == 1:
                b = b + 1
        b = b/107712
        c = 0
        for i in range(191488):
            if train_acc[i][0] == 1:
                c = c + 1
        c = np.floor_divide(c, 191488)
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(b) +",Training Accuracy " + str(c))
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) +",Training Accuracy " + str(train_acc))
        # print accuracy, accuracy.shape
        # print tf.size(accuracy)



