# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name

import os
import sys
#sys.path.insert(0, "/home/invensun/workspace/deeplearning/mxnet/mxnet/python/")

import cv2
import mxnet as mx
import numpy as np
import time
import random
import string
import Image

#from mxnet_predict import Predictor


from io import BytesIO
from captcha.image import ImageCaptcha

class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    num = random.randint(0, 9999)
    buf = str(num)
    while len(buf) < 4:
        buf = "0" + buf
    return buf

def get_label(buf):
    return np.array([int(x) for x in buf])


def str2num(strlabel):
    lst = []
    for e in strlabel:
        num = -1

        if e >= '0' and e <= '9':
            num = ord(e) - ord('0')

        if e >= 'a' and e <= 'z':
            num = 10 + ord(e) - ord('a')

        if e >= 'A' and e <= 'Z':
            #num = 10 + 26 + ord(e) - ord('A')
            num = 10 + ord(e) - ord('A')

        lst.append(num)
    return lst


class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        #self.captcha = ImageCaptcha(fonts=['./data/OpenSans-Regular.ttf'])
        self.captcha = ImageCaptcha(fonts=['./fonts/english/Helvetica Bold.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]

    def __iter__(self):
        lst = []
        for e in open('rec_code.txt', 'r'):
            lst.append(e.strip())

        l = len(lst)
        print l

        random.seed(time.time())

        for k in range(self.count / self.batch_size):
            data = []
            label = []
            random.shuffle(lst)

            for i in range(self.batch_size):
                '''num = gen_rand()
                img = self.captcha.generate(num)
                img = np.fromstring(img.getvalue(), dtype='uint8')
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.width, self.height))
                cv2.imwrite("./tmp" + str(i % 10) + ".png", img)
                img = np.multiply(img, 1/255.0)
                img = img.transpose(2, 0, 1)'''

                idx = k * self.batch_size + i
                idx = idx % l
                #print 'idx=',idx
                #print 'line=',lst[idx]

                v = string.split(lst[idx], '\t')
                #print v[0]
                #print v[1]

                img_path = v[0]
                if v[0][-4:] == '.gif':
                    imorg = Image.open(v[0])
                    img_path = v[0][:-4] + '.png'
                    imorg.save(img_path, 'png')

                #print img_path
                if not os.path.exists(img_path):
                    continue

                img0 = cv2.imread(img_path)
                img = cv2.resize(img0, (self.width, self.height))

                #cv2.imshow('img', img)

                img = np.multiply(img, 1/255.0)
                img = img.transpose(2, 0, 1)

                data.append(img)


                #print type(get_label(num))
                #print get_label(num)
                #label.append(get_label(num))

                lst_nums = str2num(v[1])
                np_label = np.array(lst_nums)
                label.append(np_label)

                #print 'img len',len(img)
                #print 'img shape',img.shape
                #print 'label len',len(np_label)



            #print label
            #print 'data shape',data.shape
            #print 'label shape',label.shape
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']

            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_ocrnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=16)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=16)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=16)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu3)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 128)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 36)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 36)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 36)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 36)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 4):
        ok = True
        for j in range(4):
            k = i * 4 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total



def train():
    network = get_ocrnet()
    devs = [mx.gpu(0)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 15,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)

    data_train = OCRIter(10000, 50, 4, 40, 100)
    data_test = OCRIter(500, 50, 4, 40, 100)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


    model_prefix = './models/mymodel'
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(32, 50), epoch_end_callback=checkpoint)

    prefix = "./models/model"
    iteration = 10
    model.save(prefix, iteration)



def load_model():
    prefix = "./models/model"
    network = get_ocrnet()
    devs = [mx.gpu(0)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 15,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)


    iteration = 10
    model.load(prefix, iteration)
    data_shape = (1, 40, 100)

    val = mx.io.ImageRecordIter(
    path_imgrec = './cap_val.rec',
    #mean_img    = '/xxx/xxx/' + "xxx.bin",
    path_imglist= './cap_val.lst',
    rand_crop   = False,
    rand_mirror = False,
    data_shape  = data_shape,
    batch_size  = 1,
    label_width = 4)

    print 'val type', type(val)
    [prob, data1, label1] = model.predict(val, return_data=True)



train()
#load_model()
