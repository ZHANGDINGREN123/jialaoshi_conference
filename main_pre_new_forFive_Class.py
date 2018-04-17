#coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf

def get_train_data(scale = 0.8):
    df = pd.read_csv("./GPS_Data/multi_maxmin/multi_maxmin_1and2and3and4and5.csv", names=['0', '1', '2', '3',
                                        '4', 'kongbai', '6', '7', '8', '9', '10', 'kongbai1', '12', '13', '14', '15', '16', 'kongbai2',
                                                                                      '18', '19', '20', '21', '22', 'kongbai2',
                                                                                          '24', '25', '26', '27', '28'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_train_input1 = df_matrix_f32[0:95744, 0:5]
    a1 = np.ones(95744)[np.newaxis]
    b1 = np.zeros(95744)[np.newaxis]
    c1 = np.zeros(95744)[np.newaxis]
    d1 = np.zeros(95744)[np.newaxis]
    e1 = np.zeros(95744)[np.newaxis]
    f1 = np.vstack((a1, b1, c1, d1, e1))
    x_train_label1 = f1.T
    x_train_input2 = df_matrix_f32[0:95744:, 6:11]
    f2 = np.vstack((b1, a1, c1, d1, e1))
    x_train_label2 = f2.T
    x_train_input3 = df_matrix_f32[0:95744:, 12:17]
    f3 = np.vstack((b1, c1, a1, d1, e1))
    x_train_label3 = f3.T
    x_train_input4 = df_matrix_f32[0:95744:, 18:23]
    f4 = np.vstack((b1, c1, d1, a1, e1))
    x_train_label4 = f4.T
    x_train_input5 = df_matrix_f32[0:95744:, 24:]
    f5 = np.vstack((b1, c1, d1, e1, a1))
    x_train_label5 = f5.T
    x_train_input = np.vstack((x_train_input1, x_train_input2, x_train_input3, x_train_input4, x_train_input5))
    x_train_label = np.vstack((x_train_label1, x_train_label2, x_train_label3, x_train_label4, x_train_label5))
    # print x_train_input.shape, x_train_label.shape
    return x_train_input, x_train_label
# get_train_data()
def get_test_data(scale = 0.8):
    df = pd.read_csv("./GPS_Data/multi_maxmin/multi_maxmin_1and2and3and4and5.csv", names=['0', '1', '2', '3',
                                        '4', 'kongbai', '6', '7', '8', '9', '10', 'kongbai1', '12', '13', '14', '15', '16', 'kongbai2',
                                                                                      '18', '19', '20', '21', '22', 'kongbai2',
                                                                                          '24', '25', '26', '27', '28'])
    df_matrix = df.as_matrix()
    df_matrix_f32 = np.float32(df_matrix)
    x_test_input1 = df_matrix_f32[95744:, 0:5]
    a1 = np.ones(52856)[np.newaxis]
    b1 = np.zeros(52856)[np.newaxis]
    c1 = np.zeros(52856)[np.newaxis]
    d1 = np.zeros(52856)[np.newaxis]
    e1 = np.zeros(52856)[np.newaxis]
    f1 = np.vstack((a1, b1, c1, d1, e1))
    x_test_label1 = f1.T
    x_test_input2 = df_matrix_f32[95744:, 6:11]
    f2 = np.vstack((b1, a1, c1, d1, e1))
    x_test_label2 = f2.T
    x_test_input3 = df_matrix_f32[95744:, 12:17]
    f3 = np.vstack((b1, c1, a1, d1, e1))
    x_test_label3 = f3.T
    x_test_input4 = df_matrix_f32[95744:, 18:23]
    f4 = np.vstack((b1, c1, d1, a1, e1))
    x_test_label4 = f4.T
    x_test_input5 = df_matrix_f32[95744:, 18:23]
    f5 = np.vstack((b1, c1, d1, e1, a1))
    x_test_label5 = f5.T
    x_test_input = np.vstack((x_test_input1, x_test_input2, x_test_input3, x_test_input4, x_test_input5))
    x_test_label = np.vstack((x_test_label1, x_test_label2, x_test_label3, x_test_label4, x_test_label5))
    # print x_test_input.shape, x_test_label.shape
    return x_test_input, x_test_label
# print get_test_data()