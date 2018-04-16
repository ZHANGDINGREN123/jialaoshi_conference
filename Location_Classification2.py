# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from main_pre import get_test_data_another,get_train_data_another, accurances_train,accurances_test

x_train_input, x_train_label = get_train_data_another()
x_test_input, x_test_label = get_test_data_another()

# 每个批次的大小
# 计算一共有多少个批次
n_batch = 2

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([2, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]) + 0.1)
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([100, 40], stddev=0.1))
b2 = tf.Variable(tf.zeros([40]) + 0.1)
L2 = tf.nn.softmax(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([40, 1], stddev=0.1))
b3 = tf.Variable(tf.zeros([1]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2, W3) + b3)



# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(y, prediction)  # argmax返回一维张量中最大的值所在的位置

# 求准确率
accuracy = tf.cast(correct_prediction, tf.float32)


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1500):
        for batch in range(n_batch):
            print x_train_input.shape, x_train_label.shape
            sess.run(train_step, feed_dict={x: x_train_input, y: x_train_label})
        # print (correct_prediction.shape)
        test_acc = sess.run(accuracy,feed_dict={x:x_test_input,y:x_test_label})
        train_acc = sess.run(accuracy,feed_dict={x:x_train_input,y:x_train_label})
        print test_acc.shape
        # accuracy_train = accurances_train(test_acc)
        # accuracy_test = accurances_test(train_acc)
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(accuracy_test) +",Training Accuracy " + str(accuracy_train))
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) +",Training Accuracy " + str(train_acc))
        # print accuracy, accuracy.shape
        # print tf.size(accuracy)



