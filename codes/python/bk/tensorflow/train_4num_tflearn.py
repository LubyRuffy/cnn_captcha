import sys,os
import cv2
import numpy as np
import tensorflow as tf
import string
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data

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



#from .config import _EPSILON, _FLOATX
#from tflearn.utils import get_from_module
#from tflearn.config import _EPSILON, _FLOATX


lst = []
for line in open('../../../tmp_dir/code1.txt', 'r'):
    lst.append(line.strip())

sample_cnt = len(lst)

W = 64
#W = 224
H = 16
#H = 56
C = 1
X_dim = W*H*C
Y_dim = 40

X0 = np.zeros( (sample_cnt, X_dim) )
Y0 = np.zeros( (sample_cnt, Y_dim) )


for sam_idx in range(len(lst)):
    line = lst[sam_idx]
    v = string.split(line, ' ')
    v_len = len(v)
    img0 = cv2.imread(v[0])
    if img0 != None:
        img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img00, (W, H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        img1 = img.reshape( (-1,) )
        img1 = img1/255.0
        X0[sam_idx] = img1
        tmp = np.zeros( (40) )
        tmp[0 +int(v[1])] = 1.0
        #tmp[10+int(v[2])] = 1.0
        #tmp[20+int(v[3])] = 1.0
        #tmp[30+int(v[4])] = 1.0
        Y0[sam_idx, :] = tmp[0: Y_dim]
        #Y0[sam_idx, :] = tmp[0: 10]
        '''
        if sam_idx == 123:
            print int(v[1])
            print int(v[2])
            print int(v[3])
            print int(v[4])
            print tmp
            xx = raw_input('pause:')
        '''

# train mnist for test
'''W = 28
H = 28
C = 1
mnist_data = mnist_input_data.read_data_sets("MNIST_data/", one_hot=True)
print type(mnist_data)
(X_data, Y_data) = (mnist_data.train.images, mnist_data.train.labels)

print X_data.shape
print Y_data.shape
X0 = X_data
Y0 = Y_data
'''

print X0.shape
print Y0.shape

X = X0.reshape( (-1, W, H, C) )
Y = Y0

print X.shape
print Y.shape


train_cnt = int(sample_cnt*0.75)
trainX = X[0:train_cnt, :]
trainY = Y[0:train_cnt]
testX  = X[train_cnt:, :]
testY  = Y[train_cnt:]

#xx = raw_input('pause:')


'''
print "check"
testid = 123
line = lst[testid]
v = string.split(line, ' ')
img = cv2.imread(v[0])
cv2.imshow("img", img)
print v
print X[0]
print Y[0]
cv2.waitKey(0)
xx = raw_input('pause:')
'''
print("Load Data finished!")


def get_net():
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

    network = input_data(shape=[None, W, H, C], data_preprocessing=img_prep, data_augmentation=img_aug, name='input')

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
        print 'myloss'
        print y_pred.get_shape()
        print y_true.get_shape()

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


network = get_net()



def custom_loss(y_pred, y_true):
    print 'myloss'
    print y_pred.get_shape()
    print y_true.get_shape()

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


def custom_metric(y_pred, y_true, x):
    rt1 = tf.equal(tf.argmax(y_pred[:, 0:10], 1), tf.argmax(y_true[:, 0:10], 1))
    rt2 = tf.equal(tf.argmax(y_pred[:, 10:20], 1), tf.argmax(y_true[:, 10:20], 1))
    rt3 = tf.equal(tf.argmax(y_pred[:, 20:30], 1), tf.argmax(y_true[:, 20:30], 1))
    rt4 = tf.equal(tf.argmax(y_pred[:, 30:40], 1), tf.argmax(y_true[:, 30:40], 1))
    rtsum = tf.add( tf.add(rt1, rt2), tf.add(rt3, rt4) )
    acc = tf.reduce_mean(tf.cast(rtsum, tf.float32))
    return acc


print network.get_shape()

#network = regression(network, optimizer='SGD', learning_rate=0.01,
#                     loss=custom_loss, name='target')

network = regression(network, optimizer='SGD', learning_rate=0.01,
                     loss=custom_loss, name='target')


def model_train():
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='../../../tmp_dir/model_ckpt/model',
                            max_checkpoints=10)
    model.fit({'input': trainX}, {'target': trainY}, n_epoch=40000,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')




def model_test4():
    mymodel = tflearn.DNN(network)

    pic_root = '../../../tmp_dir/test_code/'
    for i in os.listdir(pic_root):
        if os.path.isfile(os.path.join(pic_root,i)):

            filename = os.path.join(pic_root, i)
            print filename

            img0 = cv2.imread(filename)
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img00, (W, H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img1 = img.reshape( (-1,) )
            img1 = img1/255.0
            preX = img1.reshape( (-1, W, H, C) )
            cv2.imshow('t', img0)

            res = ''
            for j in range(4):
                model_path = '../../../tmp_dir/model' + str(j+1) + '_ckpt/model-500'
                mymodel.load(model_path)
                preY = mymodel.predict({'input': preX})
                preYidx = np.argmax(preY[0])
                res = res + str(preYidx)

            print res
            cv2.waitKey(0)



model_train()
#model_test4()


