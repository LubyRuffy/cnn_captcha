'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#from tflearn.helpers.evaluator.Evaluator import predict
from tflearn.metrics import Metric

import tensorflow as tf
import keras.backend as K


import os,sys
import cv2
import string
import re
import random
#import cPickle
import _pickle as cPickle
import h5py
import urllib
#import urllib2



tmp_prex = '../../../cnn_captcha/tmp/'
nb_epoch = 50
batch_size = 64
kernel_size = (3, 3)
pool_size = (2, 2)
IMG_W = 64
IMG_H = 16
IMG_C = 1
X_dim = IMG_W*IMG_H*IMG_C
Y_dim_each_classes = 10
Y_dim_nums = 4
Y_dim = Y_dim_each_classes * Y_dim_nums


def save_dataset_hdf5():
    lst = []
    lst_prex = os.getcwd() + '/../../tmp/4_qq/'
    lst_path = '../../tmp/4_qq/4_qq_res-all_label.txt'

    for line in open(lst_path, 'r'):
       lst.append(lst_prex + '/' + line.strip())

    # loading data
    X_data = np.zeros( (0, X_dim), dtype=np.float32 )
    Y_data = np.zeros( (0, Y_dim), dtype=np.float32 )


    for sam_idx in range(len(lst)):
        line = lst[sam_idx]
        print(line)
        v = re.split(line, ' ')
        v_len = len(v)
        img0 = cv2.imread(v[0])
        if img0 != None:
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img00, (IMG_W, IMG_H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img1 = img.reshape( (-1,) )
            img1 = img1/255.0

            X_data = np.row_stack( (X_data, img1) )

            tmp = np.zeros( (Y_dim) )
            tmp[0*Y_dim_each_classes +int(v[1])] = 1.0
            tmp[1*Y_dim_each_classes+int(v[2])] = 1.0
            tmp[2*Y_dim_each_classes+int(v[3])] = 1.0
            tmp[3*Y_dim_each_classes+int(v[4])] = 1.0
            Y_data = np.row_stack( (Y_data, tmp) )
            #Y_data = np.row_stack( (Y_data, tmp[0: 10]) )

            '''
            if sam_idx == 123:
                print int(v[1])
                print int(v[2])
                print int(v[3])
                print int(v[4])
                print tmp
                xx = input('pause:')
            '''


    X_data = X_data.reshape( (-1, IMG_W, IMG_H, IMG_C) )
    Y_data = Y_data

    #save_data as hdf5
    h5_file = h5py.File(tmp_prex + '4nums.h5', 'w')
    h5_file.create_dataset('X_data', data=X_data)
    h5_file.create_dataset('Y_data', data=Y_data)
    h5_file.close()



# loading data
save_dataset_hdf5()
h5_file = h5py.File(tmp_prex + '4nums.h5', 'r')
print(h5_file.keys())

X_data = h5_file['X_data'][:]
Y_data = h5_file['Y_data'][:]

for ii in range(Y_data.shape[0]):
    r = Y_data[ii, :]
    #r[10:40] = np.zeros(shape=(30))
    Y_data[ii, :] = r


print(X_data.shape)
print(Y_data.shape)

samples_cnt = X_data.shape[0]
idx_lst = np.arange(samples_cnt)


train_idxs = idx_lst[0:int(samples_cnt*0.8)]
test_idxs = idx_lst[int(samples_cnt*0.8):]

X_train = X_data[train_idxs, :]
Y_train = Y_data[train_idxs, :]
X_test = X_data[test_idxs, :]
Y_test = Y_data[test_idxs, :]


X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')


#dim_ordering = K.image_dim_ordering()
K_dim_ordering = 'tf'

if K_dim_ordering == 'th':
    X_train = X_train.reshape(X_train.shape[0], IMG_C, IMG_H, IMG_W)
    X_test = X_test.reshape(X_test.shape[0], IMG_C, IMG_H, IMG_W)
    input_shape = (IMG_C, IMG_H, IMG_W)
else:
    X_train = X_train.reshape(X_train.shape[0], IMG_H, IMG_W, IMG_C)
    X_test = X_test.reshape(X_test.shape[0], IMG_H, IMG_W, IMG_C)
    input_shape = (IMG_H, IMG_W, IMG_C)


print('X_train.shape', X_train.shape)
print('Y_train.shape', Y_train.shape)
print('X_test.shape', X_test.shape)
print('Y_test.shape', Y_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#print(X_train[0, :])
#print(Y_train[0, :])

print('finished loading data')
xxx = input('pause')


def get_model():
    # Building convolutional network

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    #network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)

    input_shape_list = [None] + list(input_shape)
    network = input_data(shape=input_shape_list, dtype=tf.float32, data_preprocessing=img_prep, data_augmentation=img_aug, name='input')

    network = conv_2d(network, 8, 3, activation='relu', regularizer="L2")
    #network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 8, 3, activation='relu', regularizer="L2")
    #network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    #network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)

    #network = fully_connected(network, 256, activation='tanh')
    #network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.5)


    fc11 = fully_connected(network, 256, activation='relu', regularizer='L2', weight_decay=0.001)
    fc11 = dropout(fc11, 0.5)
    fc12 = fully_connected(network, 256, activation='relu', regularizer='L2', weight_decay=0.001)
    fc12 = dropout(fc12, 0.5)
    fc13 = fully_connected(network, 256, activation='relu', regularizer='L2', weight_decay=0.001)
    fc13 = dropout(fc13, 0.5)
    fc14 = fully_connected(network, 246, activation='relu', regularizer='L2', weight_decay=0.001)
    fc14 = dropout(fc14, 0.5)

    fc21 = fully_connected(network, 10, activation='softmax')
    fc22 = fully_connected(network, 10, activation='softmax')
    fc23 = fully_connected(network, 10, activation='softmax')
    fc24 = fully_connected(network, 10, activation='softmax')

    fc = merge( [fc21, fc22, fc23, fc24], mode = 'concat', axis = 1)


    # my looo function
    def custom_loss(y_pred, y_true):
        print('myloss')
        print(y_pred.get_shape())
        print(y_true.get_shape())

        '''
        rt_acc = tf.placeholder([None, 1], dtype=tf.float32, name=None)

        for i in range(4):
            y_pred_tmp = y_pred[:, i*10: i*10+10]
            y_true_tmp = y_true[:, i*10: i*10+10]
            tmp_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))
            tf.add(rt_acc[:, ], tmp_acc)

        s = tf.Variable(1, dtype=tf.float32)
        return tf.mul(rt_acc, s)
        '''
        rt1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred[0:10], y_true[0:10]))
        rt2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred[10:20], y_true[10:20]))
        rt3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred[20:30], y_true[20:30]))
        rt4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred[30:40], y_true[30:40]))
        return tf.add( tf.add(rt1, rt2), tf.add(rt3, rt4) )


    # my metrics function
    def custom_metric(y_pred, y_true, x):
        rt1 = tf.equal(tf.argmax(y_pred[:, 0:10], 1), tf.argmax(y_true[:, 0:10], 1))
        rt2 = tf.equal(tf.argmax(y_pred[:, 10:20], 1), tf.argmax(y_true[:, 10:20], 1))
        rt3 = tf.equal(tf.argmax(y_pred[:, 20:30], 1), tf.argmax(y_true[:, 20:30], 1))
        rt4 = tf.equal(tf.argmax(y_pred[:, 30:40], 1), tf.argmax(y_true[:, 30:40], 1))
        rtsum = tf.add( tf.add(rt1, rt2), tf.add(rt3, rt4) )
        acc = tf.reduce_mean(tf.cast(rtsum, tf.float32))
        return acc



    #network = fc
    network = fully_connected(fc, Y_dim, activation='linear')

    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss=custom_loss, name='target')


    return network



def train_model(network):
    print(X_train.dtype)

    # Training
    ckpt_path = tmp_prex + 'weihts'
    model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path=ckpt_path,
                            max_checkpoints=10)

    model.fit({'input': X_train}, {'target': Y_train}, n_epoch=50,
           validation_set=({'input': X_test}, {'target': Y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

    model_path = tmp_prex + 'weihts.tflearn'
    model.save(model_path)



def test_model_web(model):
    url = 'https://www.ed3688.com/sb2/me/generate_validation_code.jsp'

    while(True):
        web = urllib.urlopen(url)
        jpg = web.read()
        tmp_path = tmp_prex + 'tmp.jpg'
        try:
            File = open(tmp_path, "wb")
            File.write( jpg)
            File.close()
        except IOError:
            print('IOError')

        print('prediction')

        img0 = cv2.imread(tmp_path)
        if img0 != None:
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img00, (IMG_W, IMG_H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img1 = img.reshape( (-1,) )
            img1 = img1/255.0

            cv2.imshow('img', img00)
            X_pred0 = img1
            X_pred0 = X_pred0[np.newaxis, :]
            print(X_pred0.shape)

            if K_dim_ordering == 'th':
                X_pred0 = X_pred0.reshape(X_pred0.shape[0], IMG_C, IMG_H, IMG_W)
            else:
                X_pred0 = X_pred0.reshape(X_pred0.shape[0], IMG_H, IMG_W, IMG_C)



            pred_y0 = model.predict(X_pred0, batch_size=1)
            print(pred_y0.shape)
            print(pred_y0)
            idx1 = np.argmax(pred_y0[0, 0*Y_dim_each_classes:1*Y_dim_each_classes])
            idx2 = np.argmax(pred_y0[0, 1*Y_dim_each_classes:2*Y_dim_each_classes])
            idx3 = np.argmax(pred_y0[0, 2*Y_dim_each_classes:3*Y_dim_each_classes])
            idx4 = np.argmax(pred_y0[0, 3*Y_dim_each_classes:4*Y_dim_each_classes])
            print(idx1, idx2, idx3, idx4)
            cv2.waitKey(0)



def test_model_img(model, lst_path):
    mymodel = tflearn.DNN(network)

    lst = []
    for e in open(lst_path, 'r'):
        lst.append(tmp_prex + e.strip())

    print(len(lst))

    for sam_idx in range(len(lst)):
        line = lst[sam_idx]
        v = re.split(line, ' ')
        v_len = len(v)
        img0 = cv2.imread(v[0])
        if img0 != None:
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img00, (IMG_W, IMG_H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img1 = img.reshape( (-1,) )
            img1 = img1/255.0

            cv2.imshow('img', img00)
            X_pred0 = img1
            X_pred0 = X_pred0[np.newaxis, :]
            #print(X_pred0.shape)

            if K_dim_ordering == 'th':
                X_pred0 = X_pred0.reshape(X_pred0.shape[0], IMG_C, IMG_H, IMG_W)
            else:
                X_pred0 = X_pred0.reshape(X_pred0.shape[0], IMG_H, IMG_W, IMG_C)



            pred_y0 = model.predict(X_pred0, batch_size=1)
            #print(pred_y0.shape)
            #print(pred_y0)
            idx1 = np.argmax(pred_y0[0, 0*Y_dim_each_classes:1*Y_dim_each_classes])
            idx2 = np.argmax(pred_y0[0, 1*Y_dim_each_classes:2*Y_dim_each_classes])
            idx3 = np.argmax(pred_y0[0, 2*Y_dim_each_classes:3*Y_dim_each_classes])
            idx4 = np.argmax(pred_y0[0, 3*Y_dim_each_classes:4*Y_dim_each_classes])
            print(idx1, idx2, idx3, idx4)
            cv2.waitKey(0)



model = get_model()

model_path = tmp_prex + 'weihts.h5'
nb_epoch = 10
train_model(model)
model.load_weights(model_path)
#test_web(model)
#test_img(model, tmp_prex+'4nums_test_lst.txt')
