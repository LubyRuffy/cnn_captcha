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
#from keras.utils.visualize_util import plot


from keras.preprocessing.image import ImageDataGenerator


#import tensorflow as tf


import os,sys
import cv2
import string
import random
import pickle
import h5py
import urllib



tmp_prex = '../../../cnn_captcha/tmp/'
dataset_name = '4_qq'
nb_epoch = 50
batch_size = 64
kernel_size = (3, 3)
pool_size = (2, 2)
IMG_W = 64
IMG_H = 64
IMG_C = 3
#IMG_W = 64
#IMG_H = 16
#IMG_C = 1
X_dim = IMG_W*IMG_H*IMG_C
Y_dim_each_classes = 26
Y_dim_nums = 4
Y_dim = Y_dim_each_classes * Y_dim_nums

EACH_HDF5_CNT = 32
SAVE_HDF5_CNT = batch_size*EACH_HDF5_CNT


def save_dataset_hdf5():
    lst = []
    lst_prex = os.getcwd() + '/../../tmp/' + dataset_name + '/'
    lst_path = '../../tmp/' + dataset_name + '/4_qq_res-all-OK.txt'

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
            #img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img00 = img0
            img = cv2.resize(img00, (IMG_W, IMG_H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            #img1 = img1.transpose(  )
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
        #if sam_idx > 0 and sam_idx % SAVE_HDF5_CNT == 0 or sam_idx/SAVE_HDF5_CNT*SAVE_HDF5_CNT+SAVE_HDF5_CNT > len(lst):
        if sam_idx > 0 and sam_idx % SAVE_HDF5_CNT == 0:
            X_data = X_data.reshape( (-1, IMG_W, IMG_H, IMG_C) )
            Y_data = Y_data

            #save_data as hdf5
            h5_file_name = tmp_prex + dataset_name + '-' + str(sam_idx/SAVE_HDF5_CNT) + '.h5'
            h5_file = h5py.File(h5_file_name, 'w')
            print(h5_file_name)
            h5_file.create_dataset('X_data', data=X_data)
            h5_file.create_dataset('Y_data', data=Y_data)
            h5_file.close()

            X_data = np.zeros( (0, X_dim), dtype=np.float32 )
            Y_data = np.zeros( (0, Y_dim), dtype=np.float32 )

            if  sam_idx/SAVE_HDF5_CNT*SAVE_HDF5_CNT+SAVE_HDF5_CNT > len(lst):
                break


# loading data
save_dataset_hdf5()
h5_file = h5py.File(tmp_prex + dataset_name + '.h5', 'r')
print(h5_file.keys())
