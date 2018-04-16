#coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf

def get_train_data(scale = 0.8):
    df = pd.read_csv("./3and4.csv", names=['one_node_longitude', 'one_node_latitude', 'kongbai', 'two_node_longitude',
                                        'two_node_latitude'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:95744, 0:2]
    a1 = np.ones(95744)[np.newaxis]
    b1 = np.zeros(95744)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_train_label1 = c1.T
    x_train_input2 = df_matrix_f32[0:95744:, 3:]
    a2 = np.ones(95744)[np.newaxis]
    b2 = np.zeros(95744)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_train_label2 = c2.T
    x_train_input = np.vstack((x_train_input1, x_train_input2))
    x_train_label = np.vstack((x_train_label1, x_train_label2))
    return x_train_input, x_train_label

def get_test_data(scale = 0.8):
    df = pd.read_csv("./3and4.csv", names=['one_node_longitude', 'one_node_latitude', 'kongbai', 'two_node_longitude', 'two_node_latitude'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[95744:, 0:2]
    a1 = np.ones(53856)[np.newaxis]
    b1 = np.zeros(53856)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_test_label1 = c1.T
    x_test_input2 = df_matrix_f32[95744:, 3:]
    a2 = np.ones(53856)[np.newaxis]
    b2 = np.zeros(53856)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_test_label2 = c2.T
    x_test_input = np.vstack((x_test_input1, x_test_input2))
    x_test_label = np.vstack((x_test_label1, x_test_label2))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label

def get_train_data_another(scale = 0.8):
    df = pd.read_csv("./3and4.csv", names=['one_node_longitude', 'one_node_latitude', 'kongbai', 'two_node_longitude',
                                        'two_node_latitude'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:95744, 0:2]
    a1 = np.ones(95744)[np.newaxis]
    b1 = np.zeros(95744)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_train_label1 = a1.T
    x_train_input2 = df_matrix_f32[0:95744:, 3:]
    a2 = np.ones(95744)[np.newaxis]
    b2 = np.zeros(95744)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_train_label2 = b2.T
    x_train_input = np.vstack((x_train_input1, x_train_input2))
    x_train_label = np.vstack((x_train_label1, x_train_label2))
    return x_train_input, x_train_label

def get_test_data_another(scale = 0.8):
    df = pd.read_csv("./3and4.csv", names=['one_node_longitude', 'one_node_latitude', 'kongbai', 'two_node_longitude', 'two_node_latitude'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[95744:, 0:2]
    a1 = np.ones(53856)[np.newaxis]
    b1 = np.zeros(53856)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_test_label1 = a1.T
    x_test_input2 = df_matrix_f32[95744:, 3:]
    a2 = np.ones(53856)[np.newaxis]
    b2 = np.zeros(53856)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_test_label2 = b2.T
    x_test_input = np.vstack((x_test_input1, x_test_input2))
    x_test_label = np.vstack((x_test_label1, x_test_label2))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label

def accurances_train(tensor = np.ones((191488, 1))):
    a = 0
    for i in range(191488):
        if tensor[i][0] == 1:
            a = a + 1
    # print tensor[0 * 88:(0 + 1) * 88, :].shape
    return a / 191488
# print(accurances_train())

def accurances_test(tensor = np.ones((107712, 1))):
    b = 0
    for i in range(107712):
        if tensor[i][0] == 1:
            b = b + 1
    return b / 107712
# print(accurances_test())

