'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import pandas

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


pos_filelist  = glob.glob('./Data/Pos_Imgs/*.jpg')
print(len(pos_filelist))
X_pos = np.array([np.array(Image.open(fname)) for fname in pos_filelist])
n_pos = X_pos.shape[0]
Y_pos = []
for i in range(n_pos):
  Y_pos.append(1)
  
Y_pos = np.array(Y_pos)

print("N_pos", n_pos)
  
neg_filelist  = glob.glob('./Data/Neg_Imgs/*.jpg')
print(len(neg_filelist))
X_neg = np.array([np.array(Image.open(fname)) for fname in neg_filelist])
n_neg = X_neg.shape[0]
Y_neg = []
for i in range(n_neg):
  Y_neg.append(0)
  
Y_neg = np.array(Y_neg)


print("N_neg = ", n_neg)
print ("x_NEG SHAPE:", X_neg.shape)
  
X_train_pos = X_pos[int(0.7*n_pos):]
X_test_pos = X_pos[-int(0.3*n_pos):]
Y_train_pos = Y_pos[int(0.7*n_pos):]
Y_test_pos = Y_pos[-int(0.3*n_pos):]
  
X_train_neg = X_neg[int(0.7*n_neg):]
X_test_neg = X_neg[-int(0.3*n_neg):]
Y_train_neg = Y_neg[int(0.7*n_neg):]
Y_test_neg = Y_neg[-int(0.3*n_neg):]

print ("x_TRAIN-POS SHAPE:", X_train_pos.shape)
print ("x_TRAIN-NEG SHAPE:", X_train_neg.shape)
  
train_data = np.concatenate((X_train_pos,X_train_neg),axis=0)
#train_data = X_trin_pos.dstack(X_train_neg)
train_labels_ = np.concatenate((Y_train_pos,Y_train_neg),axis=0)
#train_labels = Y_train_pos.dstack

#Converting class predictions to logits
train_labels=[]
for i in range(len(train_labels_)):
    if train_labels_[i] == 1:
        train_labels.append([1,0])
    else :
        train_labels.append([0,1])  

eval_data = np.concatenate((X_test_pos,X_test_neg),axis=0)
eval_labels_ = np.concatenate((Y_test_pos,Y_test_neg),axis=0) #np.stack doesnt work with appending along an axis

#Converting Class predictions to logits
eval_labels=[]
for i in range(len(eval_labels_)):
    if eval_labels_[i] == 1:
        eval_labels.append([1,0])
    else :
        eval_labels.append([0,1])
  
print ('Train_data shape:',train_data.shape)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 40000 # MNIST data input (img shape: 28*28)->my images are 200x200
n_classes = 2 # MNIST total classes (0-9 digits)->I have only 2 classes
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 200, 200, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 200, 200, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*64*100, 1024])),#[7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

current_batch_x_idx = 0
current_batch_y_idx = 0

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = train_data[current_batch_x_idx:current_batch_x_idx+batch_size]
        current_batch_x_idx = current_batch_x_idx+batch_size
        batch_y = train_labels[current_batch_y_idx:current_batch_y_idx+batch_size]
        current_batch_y_idx = current_batch_y_idx + batch_size
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
