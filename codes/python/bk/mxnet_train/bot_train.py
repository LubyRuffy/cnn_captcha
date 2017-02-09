import cv2
import mxnet as mx
import os
import sys
import numpy as np
import logging

def get_symbol():
    data=mx.symbol.Variable("data")
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')
    conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2 , act_type='relu')
    conv5_3 = mx.symbol.Convolution(name='conv5_3', data=relu5_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, attr={'lr_mult':'0.01'})
    relu5_3 = mx.symbol.Activation(name='relu5_3', data=conv5_3 , act_type='relu')
    pool5 = mx.symbol.Pooling(name='pool5', data=relu5_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
    flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5)
    fc6 = mx.symbol.FullyConnected(name='fc6', data=flatten_0 , num_hidden=4096, no_bias=False)
    relu6 = mx.symbol.Activation(name='relu6', data=fc6 , act_type='relu')
    drop6 = mx.symbol.Dropout(name='drop6', data=relu6 , p=0.500000)
    fc7 = mx.symbol.FullyConnected(name='fc7', data=drop6 , num_hidden=4096, no_bias=False)
    relu7 = mx.symbol.Activation(name='relu7', data=fc7 , act_type='relu')
    drop7 = mx.symbol.Dropout(name='drop7', data=relu7 , p=0.500000)
    fc8 = mx.symbol.FullyConnected(name='myfc8', data=drop7 , num_hidden=12, no_bias=False)
    prob = mx.symbol.SoftmaxOutput(name='softmax', data=fc8 )
    return prob


def get_iterator(batch_size=128):
    data_shape=(3, 224, 224)
    data_iter="/home/invensun/workspace/deeplearning/mxnet/mxnet/tools/"
    train=mx.io.ImageRecordIter(
        path_imgrec  = data_iter+"bot_anamils_train.rec",
        # path_imglist = "./bot_train.lst",
        mean_img     = data_iter+"mean.bin",
        data_shape   = data_shape,
        batch_size   = batch_size,
        rand_crop    = True,
        rand_mirror  = True,
        preprocess_threads=2,
        # num_parts    = kv.num_workers,
        # part_index   = kv.rank
    )

    val=mx.io.ImageRecordIter(
        path_imgrec  = data_iter+"bot_anamils_val.rec",
        # path_imglist = "./bot_val.lst",
        mean_img     = data_iter+"mean.bin",
        data_shape   = data_shape,
        batch_size   = batch_size,
        rand_crop    = False,
        rand_mirror  = False,
        preprocess_threads=2,
        # num_parts    = kv.num_workers,
        # part_index   = kv.rank
    )
    return (train, val)

if __name__ == "__main__":
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    batch_size=32

    dev=mx.gpu(0)
    #old_model = mx.model.FeedForward.load("vgg16", 1)
    model= mx.model.FeedForward(ctx=dev,
                                symbol=get_symbol(),
                                num_epoch=200,
                                learning_rate=0.01,
                                wd=0.0001,
                                #arg_params=old_model.arg_params,
                                #aux_params=old_model.aux_params,
                                #allow_extra_params=True
                                )
    data_train, data_test=get_iterator(batch_size)
    model.fit(X=data_train,
              eval_data=data_test,
              # work_load_list=[0.25, 0.25, 0.25, 0.25],
              kvstore='local_allreduce_device',
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),
              epoch_end_callback=mx.callback.do_checkpoint("models/bot"))
