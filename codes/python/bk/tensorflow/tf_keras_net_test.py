'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot


import cv2
import string
import random
import pickle


nb_epoch = 50
batch_size = 64
kernel_size = (3, 3)
pool_size = (2, 2)
IMG_W = 64
#IMG_W = 224
IMG_H = 16
#IMG_H = 56
IMG_C = 1
X_dim = IMG_W*IMG_H*IMG_C
Y_dim_each_classes = 26
Y_dim_nums = 4
Y_dim = Y_dim_each_classes * Y_dim_nums


# loading data
X_data = np.zeros( (0, X_dim), dtype=np.float32 )
Y_data = np.zeros( (0, Y_dim), dtype=np.float32 )

lst = []
#lst_path = '../../../tmp_dir/code1.txt'
lst_path = '../../../dataset/data1_1_res-all-OK.txt'
for line in open(lst_path, 'r'):
    lst.append(line.strip())

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


'''
print('load mnist data')
IMG_W = 28
IMG_H = 28
IMG_C = 1
X_dim = IMG_W*IMG_H*IMG_C
Y_dim = 40
X_train = None
Y_train = None
X_test = None
Y_test = None
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#Y_train = Y_train[np.newaxis, :]
#Y_test = Y_test[np.newaxis, :]
'''



X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')


if K.image_dim_ordering() == 'th':
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
    model.add(Convolution2D(48, kernel_size[0], kernel_size[1],
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
    model.add(Convolution2D(48, 3, 3,
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))


    #8X2X48
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(Y_dim))
    model.add(Activation('softmax'))

    return model



def get_mnist_model():
    model = Sequential()

    nb_filters = 32
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                                        input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


#model = get_mnist_model()
model = get_model()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


plot(model, to_file='tmp/model.png', show_shapes=True)


#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          verbose=1, validation_data=(X_test, Y_test))

#output = open('hist.pkl', 'wb')
#pickle.dump(hist, output, -1)

model.summary()
w_path = '../../../tmp_dir/keras_models/model'
#model.save_weights(w_path)

model.load_weights(w_path)

print(X_data.shape)




Y_dim_each_classes = 10
def test_web(model):
    url = 'https://www.ed3688.com/sb2/me/generate_validation_code.jsp'

    while(True):
        web = urllib.urlopen(url)
        jpg = web.read()
        tmp_path = './tmp/tmp.jpg'
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

            if K.image_dim_ordering() == 'th':
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






test_web(model)



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
        print(X_pred0.shape)

        if K.image_dim_ordering() == 'th':
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
