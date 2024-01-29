""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import datetime
import utils

# Define paramaters for the model
learning_rate = 0.1
batch_size = 10000
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir('data')
    os.mkdir(mnist_folder)
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = None
#############################
########## TO DO ############
#############################
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)


# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w, b = None, None
#############################
########## TO DO ############
#############################
initializer = tf.initializers.glorot_uniform()

w = tf.Variable(initializer(shape=[784, 10]), dtype=tf.float32, name='weight_1')
# w = tf.Variable(name='weight_1', shape=(784, 10), initializer=tf.keras.initializers.glorot_normal())
b = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32, name='bias_1')
# b = tf.Variable(name='bias_1', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def compute_logits(img):
    return tf.matmul(img, w) + b
    # pass
#############################
########## TO DO ############
#############################



# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy'),
        name='loss')
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = None
#############################
########## TO DO ############
#############################
# print(dir(tf.keras.optimizers))
optimizer = tf.keras.optimizers.Nadam(learning_rate) #.minimize(loss)
# optimizer = tf.contrib.opt.NadamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy
def compute_accuracy(dataset):
    n_total = 0
    n_total_correct_prediction = 0
    for _, (imgs, labels) in enumerate(dataset):
        predicts = tf.nn.softmax(compute_logits(imgs))
        n_correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1)), tf.float32))
        n_total_correct_prediction += n_correct_prediction
        n_total += len(imgs)
    return n_total_correct_prediction / n_total

# TensorBoard writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_train = 'logs/logreg/' + current_time + '/train'
log_dir_test = 'logs/logreg/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(log_dir_train)
test_summary_writer = tf.summary.create_file_writer(log_dir_test)


# Step 8: train the model
start_time = time.time()
# train the model n_epochs times
for i in range(n_epochs):
    total_loss = 0
    n_batches = 0
    # draw samples from training data
    for j, (imgs, labels) in enumerate(train_data):
        with tf.GradientTape() as g:
            logits = compute_logits(imgs)
            loss = compute_loss(logits, labels)
        gradients = g.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradients, [w, b]))
        total_loss += loss
        n_batches += 1

    # compute loss and accuracy for training data and testing data
    # after each epoch
    average_loss = total_loss / n_batches
    train_accuracy = compute_accuracy(train_data)
    test_accuracy = compute_accuracy(test_data)

    with train_summary_writer.as_default():
        tf.summary.scalar('Loss',  average_loss, step=i)
        tf.summary.scalar('Accuracy', train_accuracy, step=i)
    with test_summary_writer.as_default():
        tf.summary.scalar('Accuracy', test_accuracy, step=i)

    template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Accuracy: {}'
    print(template.format(i+1, average_loss, train_accuracy*100, test_accuracy*100))
end_time = time.time()

print('Total Time {} ms.'.format(end_time- start_time))
