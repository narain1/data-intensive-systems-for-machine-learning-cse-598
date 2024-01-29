import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import datetime
import utils
import tensorflow as tf
from tensorflow.keras import layers, models


# Define paramaters for the model
learning_rate = 0.1
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir('data')
    os.mkdir(mnist_folder)
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=False)

def unsqueeze_channel_last(img, label):
    return tf.reshape(img, (28, 28, 1)), label

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.map(lambda x, label: unsqueeze_channel_last(x, label))
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = None
#############################
########## TO DO ############
#############################
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.map(lambda x, label: unsqueeze_channel_last(x, label))
test_data = test_data.batch(batch_size)

epochs = 25

# Define the model
model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

# Second convolutional layer
model.add(layers.Conv2D(128, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

# Flatten layer for fully connected layers
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(10, activation='softmax'))

# Compile the model (add your own compilation parameters)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
# model.summary()

# Training the model
model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=test_data)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
