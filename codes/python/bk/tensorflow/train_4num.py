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


def showSize0():
  print "x size:",batch[0].shape
  #print "x_value",batch[0]
  print "y_ size:",batch[1].shape
  #print "y_example",batch[1][0]
  #print "y_example",batch[1][9]
  #print "y_example",batcllh[1][10]
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


for sam_idx in range(len(lst)):
    line = lst[sam_idx]
    v = string.split(line, ' ')
    v_len = len(v)
    img0 = cv2.imread(v[0])
    if img0 != None:
        img = cv2.resize(img0, (W, H), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        img1 = img.reshape( (-1,) )
        img1 = img1/255.0
        x_data[sam_idx] = img1
        tmp = np.zeros( (10) )

        '''
        print int(v[1])
        print int(v[2])
        print int(v[3])
        print int(v[4])
        xx = raw_input('pause:')
        print tmp.shape
        '''

        tmp[int(v[1])] = 1.0
        y_data[sam_idx, :] = tmp
        #print y_data[0]
        #xx = raw_input('pause:')
        #y_data[sam_idx][0] = int(v[1])
        #y_data[sam_idx][1] = int(v[2])
        #y_data[sam_idx][2] = int(v[3])
        #y_data[sam_idx][3] = int(v[4])


print x_data.shape
print y_data.shape

xx = raw_input('pause:')
mnist = (x_data, y_data)
'''
testid = 123
line = lst[testid]
v = string.split(line, ' ')
img = cv2.imread(v[0])
cv2.imshow("img", img)
cv2.waitKey(0)
print v
print x_data[0]
print y_data[0]
xx = raw_input('pause:')
'''
print("Load Data finished!")


# paras
W_conv1 = weight_varible([3, 3, 3, 8])
b_conv1 = bias_variable([8])

# conv layer-1
x = tf.placeholder(tf.float32, [None, x_dim])
x_image = tf.reshape(x, [-1, W, H, C])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([3, 3, 8, 8])
b_conv2 = bias_variable([8])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# conv layer-3
W_conv3 = weight_varible([3, 3, 8, 8])
b_conv3 = bias_variable([8])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# full connection
W_fc1 = weight_varible([15 * 4 * 8, 64])
b_fc1 = bias_variable([64])

h_pool3_flat = tf.reshape(h_pool3, [-1, 15 * 4 * 8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([64, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])


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
  #print "h_pool2 size:",h_pool2.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "w_conv3 size:",W_conv3.eval().shape
  print "b_conv3 size:",b_conv3.eval().shape
  print "h_conv3 size:",h_conv3.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_pool3 size:",h_pool3.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape

  print "w_fc1 size:",W_fc1.eval().shape
  print "b_fc1 size:",b_fc1.eval().shape
  print "h_pool3_flat size:",h_pool3_flat.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_fc1 size:",h_fc1.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "h_fc1_drop size:",h_fc1_drop.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape
  print "w_fc2 size:",W_fc2.eval().shape
  print "b_fc2 size:",b_fc2.eval().shape
  print "y_conv size:",y_conv.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}).shape




# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(0.3).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cross_entropy)



correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
#sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()


'''
batch1 = mnist.train.next_batch(1)
print type(batch1)
print type(batch1[0])

tt_x = batch1[0]
tt_y = batch1[1]

print tt_x.shape
print tt_x
print tt_y.shape
print tt_y

x = raw_input('pause')
'''


train_cnt = int(sample_cnt*0.7)
train_x_data = x_data[0:train_cnt, :]
train_y_data = y_data[0:train_cnt]
test_x_data = x_data[train_cnt:, :]
test_y_data = y_data[train_cnt:]



print "start train"

for i in range(200):
    #batch = mnist.train.next_batch(50)
    '''
    idx0 = (i*50) % sample_cnt
    idx1 = idx0 + 50
    if idx1 >= sample_cnt:
        idx1=sample_cnt
    print idx0,idx1
    batch = ( x_data[idx0:idx1, :], y_data[idx0:idx1] )
    '''

    batch = (train_x_data, train_y_data)
    showSize()
    x = raw_input('pause')

    if i % 100 == 0 or i < 100:
        #train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_accuacy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
        #showSize()

    #train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})


save_path="./model4/model.ckpt"
saver.save(sess,save_path)

# accuacy on test
#print("test accuracy %g"%(accuracy.eval(feed_dict={x: test_x_data, y_: test_y_data, keep_prob: 1.0})))
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: test_x_data, y_: test_y_data, keep_prob: 1.0}))

print "truth y"
print test_y_data[0:10]
pre_y = sess.run(y_conv, feed_dict={x: test_x_data, y_: test_y_data, keep_prob: 1.0})
print pre_y[0:10]
