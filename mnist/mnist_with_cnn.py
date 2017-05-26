# -*- coding: utf-8 -*-
'''
A Convolutional Neural Network implementation example using Tensorflow Library.
The example uses the mnist data in kaggle
https://www.kaggle.com/c/digit-recognizer
Author:sfailsthy
'''

# Libraries
import tensorflow as tf
import numpy as np
import csv

# setting parameters
learning_rate=1e-4
training_iterations=20000
dropout=0.5
batch_size = 50
validation_size = 2000

# Import all the training and testing data
test_data = np.genfromtxt('data/test.csv', skip_header=1, delimiter=',')
train_raw = np.genfromtxt('data/train.csv', skip_header=1, delimiter=',')
train_label = train_raw[:,0]
train_label_soft = np.zeros((len(train_label), 10))

for i in range( len(train_label) ):
    train_label_soft[i, int(train_label[i])] = 1

train_data = train_raw[:,1:train_raw.shape[1]]/255.0
train_label=train_label_soft

validation_images=train_data[0:validation_size]
validation_labels=train_label[0:validation_size]
train_images=train_data[validation_size:]
train_labels=train_label[validation_size:]


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Set up first layer
'''
The first layer is a convolution, followed by max pooling.
The convolution computes 32 features for each 5x5 patch.
Its weight tensor has a shape of [5, 5, 1, 32].
The first two dimensions are the patch size,
the next is the number of input channels (1 means that images are grayscale),
and the last is the number of output channels.
There is also a bias vector with a component for each output channel.
To apply the layer, we reshape the input data to a 4d tensor,
with the first dimension corresponding to the number of images,
second and third - to image width and height,
and the final dimension - to the number of colour channels.
After the convolution, pooling reduces the size of the output from 28x28 to 14x14.
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Set up the second layer
'''
The second layer has 64 features for each 5x5 patch.
Its weight tensor has a shape of [5, 5, 32, 64].
The first two dimensions are the patch size,
the next is the number of input channels (32 channels correspond to 32 featured that we got from previous convolutional layer),
and the last is the number of output channels.
There is also a bias vector with a component for each output channel.
Because the image is down-sampled by pooling
to 14x14 size second convolutional layer
picks up more general characteristics of the images.
Filters cover more space of the picture.
Therefore, it is adjusted for more generic features while the first layer finds smaller details.
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Densely Connected Layer
'''
Now that the image size is reduced to 7x7,
we add a fully-connected layer with 1024 neurones
to allow processing on the entire image
(each of the neurons of the fully connected layer is connected to all the activations/outpus of the previous layer)
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
'''
To prevent overfitting, we apply dropout before the readout layer.
Dropout removes some nodes from the network at each training stage.
Each of the nodes is either kept in the network with probability keep_prob or dropped with probability 1 - keep_prob.
After the training stage is over the nodes are returned to the NN with their original weights.
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.placeholder(tf.float32, [None, 10])
# => (40000, 10)

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow Session
# Initializing the variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iterations):
        # get new batch
        batch_xs,batch_ys=next_batch(batch_size)
        # train on batch
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:dropout})

        if i%100==0 or (i+1)==training_iterations:
            train_accuracy=accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
            validation_accuracy=accuracy.eval(feed_dict={x:validation_images[:batch_size],y_:validation_labels[:batch_size],keep_prob:1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))


    # Batch the test data and write to csv
    with open('data/submission.csv','w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('ImageId', 'Label'))
        test_data=test_data/255.0
        for start in range(0, test_data.shape[0], batch_size):
            # Ensure we stay within boundaries
            end = np.minimum(start + batch_size, test_data.shape[0])
            prediction = tf.argmax(y_conv,1)
            labels = prediction.eval(feed_dict={x: test_data[start:end,:], keep_prob: 1.0})
            for i in range(batch_size):
                writer.writerow((start + i + 1, labels[i]))

sess.close()
