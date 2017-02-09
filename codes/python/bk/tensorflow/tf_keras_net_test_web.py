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


if K.image_dim_ordering() == 'th':
    input_shape = (IMG_C, IMG_H, IMG_W)
else:
    input_shape = (IMG_H, IMG_W, IMG_C)



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


model = get_model()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


plot(model, to_file='tmp/model.png', show_shapes=True)

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

