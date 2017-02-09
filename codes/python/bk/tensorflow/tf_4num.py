import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, act_func = None):
    W = tf.Variable( tf.random_normal( [in_size, out_size] )  )
    bias = tf.Variable( tf.zeros([1, out_size]) ) + 0.05

    Wx_b = tf.matmul( inputs, W) + bias

    if act_func == None:
        outputs = Wx_b
    else:
        outputs = act_func(Wx_b)

    return outputs



def tf_train_test():
    # pre data
    x_data = np.linspace(-1, 1, 10000, dtype=np.float32)[:, np.newaxis]
    noise  = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y1_data = np.square(x_data) + 0.33 + noise
    y2_data = np.sin(x_data*np.pi) + noise
    #y_data = np.array([y1_data, y2_data]).transpose()
    y_data = np.hstack( (y1_data, y2_data) )
    #y_data = np.row_stack( (y1_data.T, y2_data.T) ).T

    # network
    train_x = tf.placeholder(tf.float32, [None, 1])
    train_y = tf.placeholder(tf.float32, [None, 2])

    l1 = add_layer(train_x, 1, 8, act_func=tf.nn.relu)
    #print l1.get_shape()

    #shape = l1.get_shape().as_list()
    #dim = np.prod(shape[1:])
    #flat = tf.reshape(l1, [-1, dim])
    flat = l1
    #print flat.get_shape()


    fc1 = add_layer(flat, 8, 6, act_func=tf.nn.relu)
    fc21 = add_layer(fc1, 6, 1, act_func=tf.nn.relu)
    fc22 = add_layer(fc1, 6, 1, act_func=tf.nn.relu)

    fc2 = tf.concat(1, [fc21, fc22])


    fc3 = add_layer(fc2, 2, 2, act_func=tf.nn.relu)
    prediction = add_layer(fc3, 2, 2)
    #prediction = fc2


    loss = tf.reduce_mean( tf.reduce_sum( tf.square(train_y - prediction), reduction_indices=1 ) )


    optimizer = tf.train.GradientDescentOptimizer(0.1)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.08, momentum=0.7)
    train_step = optimizer.minimize(loss)


    # init
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    #writer = tf.summary.SummaryWriter('logs', sess.graph)
    #writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)


    print 'y shape'
    print y_data.shape
    for i in range(20000):
        sess.run(train_step, feed_dict={train_x:x_data, train_y:y_data})

        if i % 500 == 0:
            #print "i=",i
            pred_y = sess.run(prediction, feed_dict={train_x:x_data, train_y:y_data})
            print sess.run(loss, feed_dict={train_x:x_data, train_y:y_data})
            pass


    print "trianing end"
    #os not function
    #os.system("clear")
    #os.system("pause")
    pause = raw_input("pause")


if __name__ == "__main__":
    print "main"
    tf_train_test()
