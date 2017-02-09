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
from keras.models import model_from_json, model_from_yaml
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot


import cv2
import string
import random
#import pickle
import cPickle as pickle



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

'''
model = get_model()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


plot(model, to_file='tmp/model.png', show_shapes=True)

model.summary()
w_path = '../../../tmp_dir/keras_models/model'
#model.save_weights(w_path)
#model.load_weights(w_path)


json_string = model.to_json()
print(json_string)

pkl_file = open('model_json.pkl', 'w')
pickle.dump(json_string, pkl_file)
pkl_file.close()


yaml_string = model.to_yaml()
print(yaml_string)

pkl_file = open('model_yaml.pkl', 'w')
pickle.dump(yaml_string, pkl_file)
pkl_file.close()
'''

pkl_file = open('model_json.pkl', 'r')
json_string1 = pickle.load(pkl_file)
pkl_file.close()
print('load json')
print(json_string1)
model_load_json = model_from_json(json_string1)
r = model_load_json.summary()
print(r)


pkl_file = open('model_yaml.pkl', 'r')
yaml_string1 = pickle.load(pkl_file)
pkl_file.close()
print('load yaml')
print(yaml_string1)
model_load_yaml = model_from_json(json_string1)
r = model_load_yaml.summary()
print(r)
