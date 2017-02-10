'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Merge, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot

#import tensorflow as tf


import os,sys
import cv2
import string
import random
import cPickle
import h5py
import urllib
import urllib2



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
    lst_prex = os.getcwd() + '/../../tmp/'
    lst_path = '../../tmp/4nums_lst_res.txt'

    for line in open(lst_path, 'r'):
       lst.append(lst_prex + '/' + line.strip())

    # loading data
    X_data = np.zeros( (0, X_dim), dtype=np.float32 )
    Y_data = np.zeros( (0, Y_dim), dtype=np.float32 )


    for sam_idx in range(len(lst)):
        line = lst[sam_idx]
        print(line)
        v = string.split(line, ' ')
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
                xx = raw_input('pause:')
            '''


    X_data = X_data.reshape( (-1, IMG_W, IMG_H, IMG_C) )
    Y_data = Y_data

    #save_data as hdf5
    h5_file = h5py.File(tmp_prex + '4nums.h5', 'w')
    h5_file.create_dataset('X_data', data=X_data)
    h5_file.create_dataset('Y_data', data=Y_data)
    h5_file.close()



# loading data
#save_dataset_hdf5()
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
xxx = raw_input('pause')


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


    model.add(Dense(64))
    model.add(Dropout(0.8))
    model.add(Dense(64))
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
    fc42 = Activation('softmax')(fc31)
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



def get_model4():
    #input data
    inputs = Input(shape=input_shape)

    #64X16X3
    #3X3@16
    conv1 = Convolution2D(48, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(inputs)
    relu1 = Activation('relu')(conv1)
    #model.add(MaxPooling2D(pool_size=pool_size))


    #64X16X16
    #3X3@48
    conv2 = Convolution2D(48, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(relu1)
    relu2 = Activation('relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=pool_size)(relu2)


    #32X8X48
    #3X3X64
    conv3 = Convolution2D(64, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape)(maxpool2)
    relu3 = Activation('relu')(conv3)
    maxpool3 = MaxPooling2D(pool_size=pool_size)(relu3)

    #32X8X64
    #3X3X64
    conv4 = Convolution2D(64, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool3)
    relu4 = Activation('relu')(conv4)
    maxpool4 = MaxPooling2D(pool_size=pool_size)(relu4)


    #16X4X64
    #3X3X48
    conv5 = Convolution2D(64, 3, 3,
                            border_mode='same',
                            input_shape=input_shape)(maxpool4)
    relu5 = Activation('relu')(conv5)
    maxpool5 = MaxPooling2D(pool_size=pool_size)(relu5)


    #8X2X48
    flat1 = Flatten()(maxpool5)
    fc1 = Dense(256)(flat1)

    fc21 = Dense(128)(fc1)
    fc22 = Dense(128)(fc1)
    fc23 = Dense(128)(fc1)
    fc24 = Dense(128)(fc1)


    fc31 = Dense(Y_dim/4)(fc21)
    fc32 = Dense(Y_dim/4)(fc22)
    fc33 = Dense(Y_dim/4)(fc23)
    fc34 = Dense(Y_dim/4)(fc24)


    fc41 = Activation('softmax')(fc31)
    fc42 = Activation('softmax')(fc31)
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

    #rt = K.concatenate( [rt1, rt2, rt3, rt4] )
    rt = keras.engine.merge( [rt1, rt2, rt3, rt4] )

    #print(K.get_value(rt))
    rtsum = K.sum(K.cast(rt, dtype='float32'))
    #print(K.get_value(rtsum))
    acc = K.mean(K.cast(rtsum, dtype='float32'))
    #print(K.get_value(acc))

    #print(K.get_variable_shape(rt1))
    #acc = K.variable(0.5, dtype='float32')
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



def train_model():
    print(X_train.dtype)
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

    #hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #          verbose=1)
    #output = open('hist.pkl', 'wb')
    #pickle.dump(hist, output, -1)

    model.summary()
    model.save_weights(model_path)

    print(model.metrics_names)
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('loss:', score[0])
    print('accuracy:', score[1])
    print('my_metricK0:', score[2])




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
#model = get_model4()
#model = get_model4_small()
model.summary()

xxx = raw_input('disp network')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              #metrics=['accuracy', my_metricK0])
              metrics=['accuracy', my_metricK0])

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plot(model, to_file=tmp_prex+'model.png', show_shapes=True)

model_path = tmp_prex + 'weihts.h5'
nb_epoch = 20
#batch_size = 1
train_model()
model.load_weights(model_path)
test_model_web(model)
#test_model_img(model, tmp_prex+'4nums_test_lst.txt')
