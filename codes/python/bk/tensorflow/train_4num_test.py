import sys,os
import cv2
import numpy as np
import tensorflow as tf
import string
from tensorflow.examples.tutorials.mnist import input_data

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def showSize():
  print "x size:",batch[0].shape
  #print "x_value",batch[0]
  print "y_ size:",batch[1].shape
  #print "y_example",batch[1][0]
  #print "y_example",batch[1][9]
  #print "y_example",batch[1][10]
  print "w_conv1 size:",W_conv1.eval().shape
  print "b_conv1 size:",b_conv1.eval().shape
  print "x_image size:",x_image.eval(feed_dict={x:batch[0]}).shape
  print "h_conv1 size:",h_conv1.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_pool1 size:",h_pool1.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "w_conv2 size:",W_conv2.eval().shape
  print "b_conv2 size:",b_conv2.eval().shape
  print "h_conv2 size:",h_conv2.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_pool2 size:",h_pool2.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "w_fc1 size:",W_fc1.eval().shape
  print "b_fc1 size:",b_fc1.eval().shape
  print "h_pool2_flat size:",h_pool2_flat.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_fc1 size:",h_fc1.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_fc1_drop size:",h_fc1_drop.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "w_fc2 size:",W_fc2.eval().shape
  print "b_fc2 size:",b_fc2.eval().shape
  print "y_conv size:",y_conv.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print("Download Done!")

lst = []
for line in open('code1.txt', 'r'):
    lst.append(line.strip())

sample_cnt = len(lst)

W = 60
H = 16
C = 3
x_dim = W*H*C
y_dim = 4

x_data = np.zeros( (sample_cnt, W*H*C) )
#y_data = np.zeros( (sample_cnt, 4) )
y_data = np.zeros( (sample_cnt, 10) )


sam_idx = 0
for e in lst:
    v = string.split(e, ' ')
    v_len = len(v)
    img0 = cv2.imread(v[0])
    if img0 != None:
        img = cv2.resize(img0, (W, H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        img1 = img.reshape( (-1,) )
        img1 = img1/255.0
        x_data[sam_idx] = img1
        tmp = np.zeros( (10) )
        tmp[int(v[4])] = 1.0
        y_data[sam_idx] = tmp
        #y_data[sam_idx][0] = int(v[1])
        #y_data[sam_idx][1] = int(v[2])
        #y_data[sam_idx][2] = int(v[3])
        #y_data[sam_idx][3] = int(v[4])

mnist = (x_data, y_data)
#xx = raw_input('pause:')
print("Load Data finished!")



# paras
W_conv1 = weight_varible([5, 5, 3, 8])
b_conv1 = bias_variable([8])

# conv layer-1
x = tf.placeholder(tf.float32, [None, x_dim])
x_image = tf.reshape(x, [-1, W, H, C])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([5, 5, 8, 12])
b_conv2 = bias_variable([12])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([15 * 4 * 16, 64])
b_fc1 = bias_variable([64])

h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 4 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#W_fc2 =

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([64, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])




# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(0.3).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

print "shape1111"
print tf.shape(y_conv)
print tf.shape(y_)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#sess = tf.InteractiveSession()
sess = tf.Session()
#sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()


print x_data.shape
print y_data.shape

train_cnt = int(sample_cnt*0.7)
train_x_data = x_data[0:train_cnt, :]
train_y_data = y_data[0:train_cnt]
test_x_data = x_data[train_cnt:, :]
test_y_data = y_data[train_cnt:]


save_path="./model1/model.ckpt"
saver.restore(sess,save_path)

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={x: test_x_data, y_: test_y_data, keep_prob: 1.0})))



test_idx = 4800
list_item = lst[test_idx]
print list_item
v = string.split(list_item, ' ')
test_path = v[0]
print test_path
img0 = cv2.imread(test_path)
if img0 != None:
    img = cv2.resize(img0, (W, H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img1 = img.reshape( (-1,) )
    img1 = img1/255.0
    tmp = np.zeros( (10) )
    tmp[int(v[4])] = 1.0
    print img1.shape
    print tmp.shape
    test_x_item = img1[np.newaxis, :]
    test_y_item = tmp[np.newaxis, :]
    print test_x_item.shape
    print test_y_item.shape
    print("test accuracy %g"%(accuracy.eval(feed_dict={x: test_x_item, y_: test_y_item, keep_prob: 1.0})))
    print y_conv.eval(feed_dict={x: test_x_item, y_: test_y_item, keep_prob: 1.0})
    print sess.run(y_conv, feed_dict={x: test_x_item, y_: test_y_item, keep_prob: 1.0})
    cv2.imshow("img0", img0)
    cv2.imshow("img",  img)
    cv2.waitKey(0)

