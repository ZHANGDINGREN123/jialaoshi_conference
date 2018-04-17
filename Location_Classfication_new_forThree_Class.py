# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from main_pre_new_forThree_Class import get_test_data,get_train_data

x_train_input, x_train_label = get_train_data()
x_test_input, x_test_label = get_test_data()
# print x_train_input.shape, x_train_label.shape,x_test_input.shape, x_test_label.shape


# 每个批次的大小
# 计算一共有多少个批次
n_batch = 10

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 5])
y = tf.placeholder(tf.float32, [None, 3])
keep_prob=tf.placeholder(tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([5, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([100, 40], stddev=0.1))
b2 = tf.Variable(tf.zeros([40]) + 0.1)
L2 = tf.nn.sigmoid(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([40, 3], stddev=0.1))
b3 = tf.Variable(tf.zeros([3]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)


# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(500):
        for batch in range(n_batch):
            sess.run(train_step, feed_dict={x: x_train_input, y: x_train_label, keep_prob:0.7})

        test_acc = sess.run(accuracy,feed_dict={x:x_test_input,y:x_test_label, keep_prob:1})
        train_acc = sess.run(accuracy,feed_dict={x:x_train_input,y:x_train_label, keep_prob:1})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) +",Training Accuracy " + str(train_acc))



