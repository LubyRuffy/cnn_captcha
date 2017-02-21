import cv2
import numpy as np
import tensorflow as tf



def tf_test1():
    t1 = tf.Variable([1, 2], name="var1")
    r1 = t1.get_shape()
    print r1


    t2 = tf.Variable([1, 2, 3, 4, 5], name="var2")
    #t2 = tf.constant([1, 2], name="var2")
    r2 = tf.shape(t2)
    print r2
    print r2[0]
    #print r2[1] out bound

    t3 = tf.Variable([1], name="var3")
    r3 = tf.shape(t3)
    print r3[0]



def tf_test2():
    MA = np.arange(24).reshape( (2, 3,4) )
    t1 = tf.Variable(MA , name="var1")
    r1 = t1.get_shape()
    print r1

    # method 1
    shape = t1.get_shape().as_list()
    dim = np.prod(shape[1:])
    t2 = tf.reshape(t1, [-1, dim])
    r2 = t2.get_shape()
    print r2

    # method 2
    t3 = tf.reshape(t1, [-1])
    r3 = t3.get_shape()
    print r3

    # method 3
    t4 = tf.squeeze(t1)
    r4 = t4.get_shape()
    print r4



if __name__ == "__main__":
    print "main"
    #tf_test1()
    tf_test2()

