'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
np.random.seed(1337)  # for reproducibility



import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Merge, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


#import tensorflow as tf



import os,sys
import cv2
import string
import random
import pickle
import h5py
import urllib
import time



tmp_prex = '../../../cnn_captcha/tmp/'
dataset_name = '4_qq'
nb_epoch = 50
batch_size = 64
kernel_size = (3, 3)
pool_size = (2, 2)
IMG_W = 64
IMG_H = 64
IMG_C = 3
X_dim = IMG_W*IMG_H*IMG_C
Y_dim_each_classes = 26
Y_dim_nums = 4
Y_dim = Y_dim_each_classes * Y_dim_nums
EACH_HDF5_CNT = 32
SAVE_HDF5_CNT = batch_size*EACH_HDF5_CNT
HDF5_CNT = 12


#dim_ordering = K.image_dim_ordering()
K_dim_ordering = 'tf'

if K_dim_ordering == 'th':
    input_shape = (IMG_C, IMG_H, IMG_W)
else:
    input_shape = (IMG_H, IMG_W, IMG_C)



def read_hdf5(idx):
    # loading data
    hdf5_file_path = tmp_prex + dataset_name + str(idx)+ '.h5'
    h5_file = h5py.File(hdf5_file_path, 'r')
    print(h5_file.keys())



    X_data = h5_file['X_data'][:]
    Y_data = h5_file['Y_data'][:]

    print(X_data.shape)
    print(Y_data.shape)

    samples_cnt = X_data.shape[0]
    idx_lst = np.arange(samples_cnt)

    random.shuffle(idx_lst)

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

    return(X_train, Y_train, X_test, Y_test)



def get_model():

    model = Sequential()

    #64X16X3
    #3X3@16
    model.add(Convolution2D(16, 3, 3,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=pool_size))


    #64X16X16
    #3X3@48
    model.add(Convolution2D(88, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    #32X8X48
    #3X3X64
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    #32X8X64
    #3X3X64
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    #16X4X64
    #3X3X48
    model.add(Convolution2D(88, 3, 3,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    #8X2X48
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(Y_dim))
    model.add(Activation('softmax'))
    #model.add(Activation('sigmoid'))

    def softmax4(x):
        s1 = K.softmax(x[0:10])
        s2 = K.softmax(x[10:20])
        s3 = K.softmax(x[20:30])
        s4 = K.softmax(x[30:40])

        #rt = K.concatenate( [s1, s2, s3, s4] )
        rt = keras.engine.merge( [s1, s2, s3, s4], mode='concat')
        print(K.get_variable_shape(rt))
        return rt

    #model.add(Activation(softmax4))
    return model


def get_model4_small():
    #input data
    inputs = Input(shape=input_shape)

    #64X16X3
    #3X3@16
    conv1 = Convolution2D(8, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(inputs)
    relu1 = Activation('relu')(conv1)
    #model.add(MaxPooling2D(pool_size=pool_size))


    #64X16X16
    #3X3@48
    conv2 = Convolution2D(8, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(relu1)
    relu2 = Activation('relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=pool_size)(relu2)


    #32X8X48
    #3X3X64
    conv3 = Convolution2D(12, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(maxpool2)
    relu3 = Activation('relu')(conv3)
    maxpool3 = MaxPooling2D(pool_size=pool_size)(relu3)

    #32X8X64
    #3X3X64
    conv4 = Convolution2D(8, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool3)
    relu4 = Activation('relu')(conv4)
    maxpool4 = MaxPooling2D(pool_size=pool_size)(relu4)


    #16X4X64
    #3X3X48
    conv5 = Convolution2D(8, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool4)
    relu5 = Activation('relu')(conv5)
    maxpool5 = MaxPooling2D(pool_size=pool_size)(relu5)


    #8X2X48
    flat1 = Flatten()(maxpool5)
    fc1 = Dense(32)(flat1)

    fc21 = Dense(12)(fc1)
    fc22 = Dense(12)(fc1)
    fc23 = Dense(12)(fc1)
    fc24 = Dense(12)(fc1)


    fc31 = Dense(Y_dim/4)(fc21)
    fc32 = Dense(Y_dim/4)(fc22)
    fc33 = Dense(Y_dim/4)(fc23)
    fc34 = Dense(Y_dim/4)(fc24)


    fc41 = Activation('softmax')(fc31)
    fc42 = Activation('softmax')(fc32)
    fc43 = Activation('softmax')(fc33)
    fc44 = Activation('softmax')(fc34)

    #fc5 = Merge([fc41, fc42, fc43, fc44], mode='concat', concat_axis=1)
    fc5 = keras.engine.merge([fc41, fc42, fc43, fc44], mode='concat')
    predictions = fc5


    def my_layer(inputs):
        pass

    def my_layer_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        return tuple(shape)

    #predictions = Lambda()

    model = Model(input=inputs, output=predictions)
    return model



def get_model4_nums():
    #input data
    inputs = Input(shape=input_shape)

    filters_4_nums = 6
    #64X16X3
    #3X3@16
    conv1 = Convolution2D(filters_4_nums, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(inputs)
    relu1 = Activation('relu')(conv1)
    #model.add(MaxPooling2D(pool_size=pool_size))


    #64X16X16
    #3X3@48
    conv2 = Convolution2D(filters_4_nums, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(relu1)
    relu2 = Activation('relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=pool_size)(relu2)


    #32X8X48
    #3X3X64
    conv3 = Convolution2D(filters_4_nums, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(maxpool2)
    relu3 = Activation('relu')(conv3)
    maxpool3 = MaxPooling2D(pool_size=pool_size)(relu3)

    #32X8X64
    #3X3X64
    conv4 = Convolution2D(filters_4_nums, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool3)
    relu4 = Activation('relu')(conv4)
    maxpool4 = MaxPooling2D(pool_size=pool_size)(relu4)


    #16X4X64
    #3X3X48
    conv5 = Convolution2D(filters_4_nums, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool4)
    relu5 = Activation('relu')(conv5)
    maxpool5 = MaxPooling2D(pool_size=pool_size)(relu5)


    #8X2X48
    flat1 = Flatten()(maxpool5)
    fc1 = Dense(6)(flat1)

    fc21 = Dense(12)(fc1)
    fc22 = Dense(12)(fc1)
    fc23 = Dense(12)(fc1)
    fc24 = Dense(12)(fc1)


    fc31 = Dense(Y_dim/4)(fc1)
    fc32 = Dense(Y_dim/4)(fc1)
    fc33 = Dense(Y_dim/4)(fc1)
    fc34 = Dense(Y_dim/4)(fc1)


    fc41 = Activation('softmax')(fc31)
    fc42 = Activation('softmax')(fc32)
    fc43 = Activation('softmax')(fc33)
    fc44 = Activation('softmax')(fc34)

    #fc5 = Merge([fc41, fc42, fc43, fc44], mode='concat', concat_axis=1)
    fc5 = keras.engine.merge([fc41, fc42, fc43, fc44], mode='concat')
    predictions = fc5


    def my_layer(inputs):
        pass

    def my_layer_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        return tuple(shape)

    #predictions = Lambda()

    model = Model(input=inputs, output=predictions)
    return model


n_filters = 8
def get_model4_qq():
    #input data
    inputs = Input(shape=input_shape)

    #64X16X3
    #3X3@16
    conv1 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(inputs)
    relu1 = Activation('relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=pool_size)(relu1)


    #32X8X16
    #3X3@48
    conv2 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(maxpool1)
    relu2 = Activation('relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=pool_size)(relu2)


    #16X4X48
    #3X3X64
    conv3 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(maxpool2)
    relu3 = Activation('relu')(conv3)
    #maxpool3 = MaxPooling2D(pool_size=pool_size)(relu3)
    maxpool3 = relu3


    ###
    #32X8X64
    #3X3X64
    conv4 = Convolution2D(24, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool3)
    relu4 = Activation('relu')(conv4)
    maxpool4 = MaxPooling2D(pool_size=pool_size)(relu4)


    #16X4X64
    #3X3X48
    conv5 = Convolution2D(32, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool4)
    relu5 = Activation('relu')(conv5)
    maxpool5 = MaxPooling2D(pool_size=pool_size)(relu5)


    #8X2X48
    flat1 = Flatten()(maxpool3)
    fc1 = Dense(1024)(flat1)

    '''
    fc21 = Dense(128)(fc1)
    fc22 = Dense(128)(fc1)
    fc23 = Dense(128)(fc1)
    fc24 = Dense(128)(fc1)

    fc31 = Dense(64)(fc21)
    fc32 = Dense(64)(fc22)
    fc33 = Dense(64)(fc23)
    fc34 = Dense(64)(fc24)
    '''

    fc41 = Dense(Y_dim/4)(fc1)
    fc42 = Dense(Y_dim/4)(fc1)
    fc43 = Dense(Y_dim/4)(fc1)
    fc44 = Dense(Y_dim/4)(fc1)


    fc51 = Activation('softmax')(fc41)
    fc52 = Activation('softmax')(fc42)
    fc53 = Activation('softmax')(fc43)
    fc54 = Activation('softmax')(fc44)

    #fc5 = Merge([fc41, fc42, fc43, fc44], mode='concat', concat_axis=1)
    fc6 = keras.engine.merge([fc51, fc52, fc53, fc54], mode='concat')
    predictions = fc6


    def my_layer(inputs):
        pass

    def my_layer_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        return tuple(shape)

    #predictions = Lambda()

    model = Model(input=inputs, output=predictions)
    return model



n_filters = 16
kernel_size = (5, 5)
pool_strider_size = (2, 2)
def get_model4_qq_paper():
    #input data
    inputs = Input(shape=input_shape)

    #130X50X3
    #5X5@n_filters
    conv1 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape)(inputs)
    relu1 = Activation('relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=pool_size, strides=(pool_strider_size[0], pool_strider_size[1]))(relu1)

    #125X45X n_filters
    #5X5@n_filters
    conv2 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape)(maxpool1)
    relu2 = Activation('relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=pool_size, strides=(pool_strider_size[0], pool_strider_size[1]))(relu2)


    #120X40X n_filters
    #5X5X n_filters
    conv3 = Convolution2D(n_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape)(maxpool2)
    relu3 = Activation('relu')(conv3)
    maxpool3 = MaxPooling2D(pool_size=pool_size, strides=(pool_strider_size[0], pool_strider_size[1]))(relu3)


    #120X40X n_filters
    flat1 = Flatten()(maxpool3)
    fc1 = Dropout(0.5)(Activation('relu')(Dense(384)(flat1)))

    fc21 = Dropout(0.5)(Activation('relu')(Dense(128)(fc1)))
    fc22 = Dropout(0.5)(Activation('relu')(Dense(128)(fc1)))
    fc23 = Dropout(0.5)(Activation('relu')(Dense(128)(fc1)))
    fc24 = Dropout(0.5)(Activation('relu')(Dense(128)(fc1)))

    '''
    fc31 = Dense(64)(fc21)
    fc32 = Dense(64)(fc22)
    fc33 = Dense(64)(fc23)
    fc34 = Dense(64)(fc24)
    '''

    fc41 = Activation('softmax')(Dense(Y_dim/4)(fc21))
    fc42 = Activation('softmax')(Dense(Y_dim/4)(fc22))
    fc43 = Activation('softmax')(Dense(Y_dim/4)(fc23))
    fc44 = Activation('softmax')(Dense(Y_dim/4)(fc24))


    #fc5 = Merge([fc41, fc42, fc43, fc44], mode='concat', concat_axis=1)
    fc5 = keras.engine.merge([fc41, fc42, fc43, fc44], mode='concat')
    predictions = fc5


    def my_layer(inputs):
        pass

    def my_layer_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        return tuple(shape)

    #predictions = Lambda()

    model = Model(input=inputs, output=predictions)
    return model



def my_metricK(y_true, y_pred):
    rt1 = K.equal(K.argmax(y_pred[:, 0*Y_dim_each_classes:1*Y_dim_each_classes], 1), K.argmax(y_true[:, 0*Y_dim_each_classes:1*Y_dim_each_classes], 1))
    rt2 = K.equal(K.argmax(y_pred[:, 1*Y_dim_each_classes:2*Y_dim_each_classes], 1), K.argmax(y_true[:, 1*Y_dim_each_classes:2*Y_dim_each_classes], 1))
    rt3 = K.equal(K.argmax(y_pred[:, 2*Y_dim_each_classes:3*Y_dim_each_classes], 1), K.argmax(y_true[:, 2*Y_dim_each_classes:3*Y_dim_each_classes], 1))
    rt4 = K.equal(K.argmax(y_pred[:, 3*Y_dim_each_classes:4*Y_dim_each_classes], 1), K.argmax(y_true[:, 3*Y_dim_each_classes:4*Y_dim_each_classes], 1))


    # error !!!
    #print(K.eval(rt1))

    #print(K.get_value(rt1))
    #print(K.get_value(rt2))
    #print(K.get_value(rt3))
    #print(K.get_value(rt4))


    print('get rt1 shape')
    print(K.get_variable_shape(rt1))
    rt1 = K.expand_dims(rt1)
    rt2 = K.expand_dims(rt2)
    rt3 = K.expand_dims(rt3)
    rt4 = K.expand_dims(rt4)
    print('get rt1 shape')
    print(K.get_variable_shape(rt1))

    #############################################
    #rt = K.concatenate( [rt1, rt2, rt3, rt4] )
    rt = keras.engine.merge( [rt1, rt2, rt3, rt4], mode='concat', concat_axis=-1)
    #############################################
    print('shape of rt')
    print(K.get_variable_shape(rt))

    #rt = K.stack( [rt1, rt2, rt3, rt4]  )
    #rt = K.stack( K.rt1  )

    #print(K.get_value(rt))
    #rtsum = K.sum(K.cast(rt, dtype='float32'), axis=-1)
    #print('shape of rtsum')
    #print(K.get_variable_shape(rtsum))
    #print(K.get_value(rtsum))
    acc1 = K.mean(rt, axis=0)
    acc = K.mean(acc1, axis=0)
    print('shape of acc')
    print(K.get_variable_shape(acc))
    #acc = K.variable(0.5, dtype='float32')
    #acc = K.mean(K.cast(rtsum, dtype='float32'))
    #acc = K.dot(acc, K.variable(0.25, dtype='float32'))
    return acc



def my_metricK0(y_true, y_pred):
    print(K.get_variable_shape(y_true))
    print(K.get_variable_shape(y_pred))

    #if K.get_variable_shape(y_true) != K.get_variable_shape(y_pred):
    #    print('y_true and y_pred shape is not same!')

    rt1 = K.equal(K.argmax(y_pred[:, 0:10], 1), K.argmax(y_true[:, 0:10], 1))
    rt2 = K.equal(K.argmax(y_pred[:, 10:20], 1), K.argmax(y_true[:, 10:20], 1))
    rt3 = K.equal(K.argmax(y_pred[:, 20:30], 1), K.argmax(y_true[:, 20:30], 1))
    rt4 = K.equal(K.argmax(y_pred[:, 30:40], 1), K.argmax(y_true[:, 30:40], 1))

    # error !!!
    #print(K.eval(rt1))

    #print(K.get_value(rt1))
    #print(K.get_value(rt2))
    #print(K.get_value(rt3))
    #print(K.get_value(rt4))


    print('get rt1 shape')
    print(K.get_variable_shape(rt1))
    rt1 = K.expand_dims(rt1)
    rt2 = K.expand_dims(rt2)
    rt3 = K.expand_dims(rt3)
    rt4 = K.expand_dims(rt4)
    print('get rt1 shape')
    print(K.get_variable_shape(rt1))

    #############################################
    #rt = K.concatenate( [rt1, rt2, rt3, rt4] )
    rt = keras.engine.merge( [rt1, rt2, rt3, rt4], mode='concat', concat_axis=-1)
    #############################################
    print('shape of rt')
    print(K.get_variable_shape(rt))

    #rt = K.stack( [rt1, rt2, rt3, rt4]  )
    #rt = K.stack( K.rt1  )

    #print(K.get_value(rt))
    #rtsum = K.sum(K.cast(rt, dtype='float32'), axis=-1)
    #print('shape of rtsum')
    #print(K.get_variable_shape(rtsum))
    #print(K.get_value(rtsum))
    acc1 = K.mean(rt, axis=0)
    acc = K.mean(acc1, axis=0)
    print('shape of acc')
    print(K.get_variable_shape(acc))
    #acc = K.variable(0.5, dtype='float32')
    #acc = K.mean(K.cast(rtsum, dtype='float32'))
    #acc = K.dot(acc, K.variable(0.25, dtype='float32'))
    return acc





def my_metric(y_true, y_pred):
    rt1 = tf.equal(tf.argmax(y_pred[:, 0*Y_dim_each_classes:1*Y_dim_each_classes], 1), tf.argmax(y_true[:, 0*Y_dim_each_classes:1*Y_dim_each_classes], 1))
    rt2 = tf.equal(tf.argmax(y_pred[:, 1*Y_dim_each_classes:2*Y_dim_each_classes], 1), tf.argmax(y_true[:, 1*Y_dim_each_classes:2*Y_dim_each_classes], 1))
    rt3 = tf.equal(tf.argmax(y_pred[:, 2*Y_dim_each_classes:3*Y_dim_each_classes], 1), tf.argmax(y_true[:, 2*Y_dim_each_classes:3*Y_dim_each_classes], 1))
    rt4 = tf.equal(tf.argmax(y_pred[:, 3*Y_dim_each_classes:4*Y_dim_each_classes], 1), tf.argmax(y_true[:, 3*Y_dim_each_classes:4*Y_dim_each_classes], 1))
    print(K.eval(rt1))

    rtsum = tf.add( tf.add(rt1, rt2), tf.add(rt3, rt4) )
    print(K.eval(rtsum))
    acc = tf.reduce_mean(tf.cast(rtsum, tf.float32))
    print(K.eval(acc))
    return acc



def DataAugmentation(X_batch, Y_batch):
    n = X_batch.shape[0]
    for i in range(n):
        x = X_batch[i, ]
        y = Y_batch[i, ]

        #print(x.shape)
        #print(y.shape)

        if K_dim_ordering == 'th':
            xx = x.reshape(IMG_C, IMG_H, IMG_W)
        else:
            xx = x.reshape(IMG_H, IMG_W, IMG_C)

        #print('xx shape')
        #print(xx.shape)

        crop_gap = 50.0
        crop_w = random.random()/crop_gap
        crop_h = random.random()/crop_gap

        (x1,y1) = (y[0],y[1])
        (x2,y2) = (y[2],y[3])

        crop_x1 = x1 - crop_w
        crop_y1 = y1 - crop_h

        crop_x2 = x2 - crop_w
        crop_y2 = y2 - crop_h

        if crop_x1 <=0 or crop_y1 <=0 or crop_x2 <=0 or crop_y2 <= 0:
            continue

        #print(crop_w, crop_h, crop_x1, crop_y1, crop_x2, crop_y2)
        lt_x1 = int(crop_w*IMG_W)
        lt_y1 = int(crop_h*IMG_H)
        crop_img = xx[lt_y1:IMG_H-lt_y1, lt_x1:IMG_W-lt_x1]
        crop_img_y = np.array( [crop_x1, crop_y1, crop_x2, crop_y2] )

        #print(lt_x1, lt_y1)

        #print(crop_img.shape)
        #print(crop_img_y.shape)
        img = cv2.resize(crop_img, (IMG_W, IMG_H), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        img1 = img.reshape( (-1,) )

        if K_dim_ordering == 'th':
            xxx = img1.reshape(IMG_C, IMG_H, IMG_W)
        else:
            xxx = img1.reshape(IMG_H, IMG_W, IMG_C)

        X_batch[i, ] = xxx
        Y_batch[i, ] = np.array( crop_img_y )

        #check
        '''
        cv2.imshow('org', xx)
        ckx1 = int(float(crop_x1)*IMG_W)
        cky1 = int(float(crop_y1)*IMG_H)
        ckx2 = int(float(crop_x2)*IMG_W)
        cky2 = int(float(crop_y2)*IMG_H)
        cv2.circle(img, (ckx1, cky1), 1, (0,255,0), 1)
        cv2.circle(img, (ckx2, cky2), 1, (0,255,0), 1)
        img_test = img*255
        img_test = img_test.astype('uint8')
        cv2.imshow('img_test', img_test)
        #cv2.waitKey(0)

        print('ckeck data')
        print(crop_w, crop_h)
        print(x1, y1, x2, y2)
        print(crop_x1, crop_y1, crop_x2, crop_y2)
        #cv2.waitKey(0)
        '''
    return (X_batch, Y_batch)




def test_model_web(model):
    url = 'https://www.ed3688.com/sb2/me/generate_validation_code.jsp'
    url = 'http://211.139.145.140:9080/eucp/validateAction.do'
    url = 'http://captcha.qq.com/getimage?0.6939826908284301'

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
            #img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img00 = img0
            print('img shape', img00.shape)
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
            print(chr(idx1+97), chr(idx2+97), chr(idx3+97), chr(idx4+97))
            cv2.waitKey(0)



def test_model_img(model, lst_path):
    lst = []
    for e in open(lst_path, 'r'):
        lst.append(tmp_prex + e.strip())

    print(len(lst))

    for sam_idx in range(len(lst)):
        line = lst[sam_idx]
        v = string.split(line, ' ')
        v_len = len(v)
        img0 = cv2.imread(v[0])
        if img0 != None:
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img00, (IMG_W, IMG_H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img1 = img.reshape( (-1,) )
            img1 = img1/255.0

            #cv2.imshow('img', img00)
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




def main_train_batch():
    model = get_model()
    model.summary()

    cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    result_prex = tmp_prex + dataset_name + '/result/' + cur_time

    json_string = model.to_json()
    json_path = result_prex + '.json'
    open(json_path,'w').write(json_string)


    xxx = input('disp network')

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=[my_metricK])


    plot(model, to_file=result_prex+'.png', show_shapes=True)

    ckpt_prex = result_prex + '_ckpt'

    if os.path.exists(ckpt_prex):
        os.redir(ckpt_prex)
        os.mkdir(ckpt_prex)


    nb_epoch_base = 0
    nb_epoch = 12000


    for n_step0 in range(nb_epoch):
        n_step = n_step0 + nb_epoch_base
        #print('n_step:', n_step)


        HDF5_IDX = n_step / (SAVE_HDF5_CNT) % HDF5_CNT
        HDF5_IDX = 0
        (X_train, Y_train, X_test, Y_test) = read_hdf5(HDF5_IDX+1)



        print('HDF5_IDX={0}'.format(HDF5_IDX))
        for n_batch_idx in range(EACH_HDF5_CNT):

            X_train_batch = X_train[n_batch_idx*batch_size:(n_batch_idx+1)*batch_size, ]
            Y_train_batch = Y_train[n_batch_idx*batch_size:(n_batch_idx+1)*batch_size, ]

            (X_train_batch1, Y_train_batch1) = DataAugmentation(X_train_batch, Y_train_batch)
            #(X_train_batch1, Y_train_batch1) = (X_train_batch, Y_train_batch)
            train_on_batch_error = model.train_on_batch(X_train_batch1, Y_train_batch1)


            X_test_batch = X_test[n_batch_idx*batch_size:(n_batch_idx+1)*batch_size, ]
            Y_test_batch = Y_test[n_batch_idx*batch_size:(n_batch_idx+1)*batch_size, ]

            (X_test_batch1, Y_test_batch1) = DataAugmentation(X_test_batch, Y_test_batch)
            #(X_test_batch1, Y_test_batch1) = (X_test_batch, Y_test_batch)
            test_on_batch_error = model.test_on_batch(X_test_batch1, Y_test_batch1)

            print('n_step={0} {1} {2}'.format(n_step, train_on_batch_error[0], test_on_batch_error[0]) )

            if n_step % 100 == 0:
                #checkpoint = ModelCheckpoint(filepath='./ckpt/checkpoint-{epoch:05d}-{val_loss:.4f}.hdf5')

                save_path = ckpt_prex + '/checkpoint-%08d.h5' % (n_step)
                print(save_path)
                #model.save_weights(save_path)

            #n_step = n_step + EACH_HDF5_CNT


        del X_train
        del Y_train
        del X_test
        del Y_test



    model.summary()
    #model.save_weights(result_prex+'weight.h5')

    print(model.metrics_names)
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('loss:', score[0])
    print('accuracy:', score[1])



def main_train():

    datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)


    model = get_model()
    model.summary()

    cur_time = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    result_prex = tmp_prex + dataset_name + '/result/' + cur_time

    json_string = model.to_json()
    json_path = result_prex + '.json'
    open(json_path,'w').write(json_string)


    xxx = input('disp network')

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=[my_metricK])
              #metrics=['accuracy'])



    plot(model, to_file=result_prex+'.png', show_shapes=True)

    ckpt_prex = result_prex + '_ckpt'

    if os.path.exists(ckpt_prex):
        os.redir(ckpt_prex)
        os.mkdir(ckpt_prex)


    nb_epoch_base = 0
    nb_epoch = 12000


    for n_step0 in range(nb_epoch):
        n_step = n_step0 + nb_epoch_base
        #print('n_step:', n_step)


        HDF5_IDX = n_step / (SAVE_HDF5_CNT) % HDF5_CNT
        HDF5_IDX = 0
        (X_train, Y_train, X_test, Y_test) = read_hdf5(HDF5_IDX+1)


        datagen.fit(X_train)

        hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), nb_epoch=nb_epoch,
                   samples_per_epoch=len(X_train), verbose=1, validation_data=(X_test, Y_test))


        #hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        #          verbose=1, validation_data=(X_test, Y_test))

        #hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        #          verbose=1)
        #output = open('hist.pkl', 'wb')
        #pickle.dump(hist, output, -1)

        model.summary()
        model_path = ckpt_prex + '/checkpoint-final.h5'
        model.save_weights(model_path)

        print(model.metrics_names)
        score = model.evaluate(X_test, Y_test, verbose=0)

        print('loss:', score[0])
        print('accuracy:', score[1])



main_train_batch()
#main_train()
