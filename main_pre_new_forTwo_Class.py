#coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf

def get_train_data(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:111450, 0:5]
    a1 = np.ones(111450)[np.newaxis]
    b1 = np.zeros(111450)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_train_label1 = c1.T
    x_train_input2 = df_matrix_f32[0:111450:, 6:]
    a2 = np.ones(111450)[np.newaxis]
    b2 = np.zeros(111450)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_train_label2 = c2.T
    x_train_input = np.vstack((x_train_input1, x_train_input2))
    x_train_label = np.vstack((x_train_label1, x_train_label2))
    # print x_train_input.shape, x_train_label.shape
    return x_train_input, x_train_label
# print get_train_data()
def get_test_data(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[111450:, 0:5]
    a1 = np.ones(37150)[np.newaxis]
    b1 = np.zeros(37150)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_test_label1 = c1.T
    x_test_input2 = df_matrix_f32[111450:, 6:]
    a2 = np.ones(37150)[np.newaxis]
    b2 = np.zeros(37150)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_test_label2 = c2.T
    x_test_input = np.vstack((x_test_input1, x_test_input2))
    x_test_label = np.vstack((x_test_label1, x_test_label2))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label
# print get_test_data()


def get_train_data_only_one_classification(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:111450, 0:5]
    a1 = np.ones(111450)[np.newaxis]
    b1 = np.zeros(111450)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_train_label1 = c1.T
    x_train_input2 = df_matrix_f32[0:111450:, 6:]
    a2 = np.ones(111450)[np.newaxis]
    b2 = np.zeros(111450)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_train_label2 = c2.T
    x_train_input = np.vstack((x_train_input1))
    x_train_label = np.vstack((x_train_label1))
    # print x_train_input.shape, x_train_label.shape
    return x_train_input, x_train_label
# print get_train_data()

def get_train_data_only_two_classification(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:111450, 0:5]
    a1 = np.ones(111450)[np.newaxis]
    b1 = np.zeros(111450)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_train_label1 = c1.T
    x_train_input2 = df_matrix_f32[0:111450:, 6:]
    a2 = np.ones(111450)[np.newaxis]
    b2 = np.zeros(111450)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_train_label2 = c2.T
    x_train_input = np.vstack((x_train_input2))
    x_train_label = np.vstack((x_train_label2))
    # print x_train_input.shape, x_train_label.shape
    return x_train_input, x_train_label
# print get_train_data()



def get_test_data_only_one_classification(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[111450:, 0:5]
    a1 = np.ones(37150)[np.newaxis]
    b1 = np.zeros(37150)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_test_label1 = c1.T
    x_test_input2 = df_matrix_f32[111450:, 6:]
    a2 = np.ones(37150)[np.newaxis]
    b2 = np.zeros(37150)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_test_label2 = c2.T
    x_test_input = np.vstack((x_test_input1))
    x_test_label = np.vstack((x_test_label1))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label
# print get_test_data()

def get_test_data_only_two_classification(scale = 0.8):
    df = pd.read_csv("./GPS_Data/1_maxmin/1_maxmin_2and3.csv", names=['1', '2', '3', '4',
                                        '5', 'kongbai', '7', '8', '9', '10', '11'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[111450:, 0:5]
    a1 = np.ones(37150)[np.newaxis]
    b1 = np.zeros(37150)[np.newaxis]
    c1 = np.vstack((a1, b1))
    x_test_label1 = c1.T
    x_test_input2 = df_matrix_f32[111450:, 6:]
    a2 = np.ones(37150)[np.newaxis]
    b2 = np.zeros(37150)[np.newaxis]
    c2 = np.vstack((b2, a2))
    x_test_label2 = c2.T
    x_test_input = np.vstack((x_test_input2))
    x_test_label = np.vstack((x_test_label2))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label
# print get_test_data()

