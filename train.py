

import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
import matplotlib.pyplot as plt
import cifar10
import argparse

from cifar10 import img_size, num_channels, num_classes

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lr", required=True, type=float)
parser.add_argument("-b", "--batch_size", required=True, type=int)
parser.add_argument("-i", "--init", required=True, type=int)
parser.add_argument("-s", "--save_dir", required=True)
args=parser.parse_args()


#data_path= downloaded dataset location
cifar10.data_path = "D:/"

class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

if(args.init==1):
 w=tf.contrib.layers.xavier_initializer()
else:
 w=None
#None implies He initialization


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        # Plot image.
        ax.imshow(images[i, :, :, :], interpolation=interpolation)
        cls_true_name = class_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)
    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
    # Create the convolutional neural network using Pretty Tensor.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=3, depth=64, stride=(1, 1), edges="SAME", name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=3, depth=128, stride=(1, 1), edges="SAME", name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=3, depth=256, stride=(1, 1), edges="SAME", name='layer_conv3', batch_normalize=True).\
            conv2d(kernel=3, depth=256, stride=(1, 1), edges="SAME", name='layer_conv4', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1', weights=w).\
            fully_connected(size=1024, name='layer_fc2', weights=w).\
            softmax_classifier(num_classes=num_classes, labels=y_true)
    return y_pred, loss

def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        images = x

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)
    return y_pred, loss

def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable




def get_layer_output(layer_name):
    tensor_name = "network/" + layer_name + "/Relu:0"
    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor


def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images, size=args.batch_size, replace=False)
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    for i in range(num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        # Run the optimizer using this batch of training data.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    # Get the images from the test-set that have been
    # incorrectly classified.   
    images = images_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    # Plot the first 9 images.
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + args.batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)
    return correct, cls_pred


def  predict_cls_test():
    return predict_cls(images = images_test,labels = labels_test, cls_true = cls_test)


def  classification_accuracy(correct):
     return correct.mean(), correct.sum()

def print_test_accuracy(show_example_errors=False):
    correct, cls_pred = predict_cls_test()
     # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    # Number of images being classified.
    num_images = len(correct)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
     # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)



def  plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))
    num_filters = w.shape[3]
    # Number of grids to plot.
    num_grids = math.ceil(math.sqrt(num_filters))
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            img = w[:, :, input_channel, i]
            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max, interpolation='nearest', cmap='seismic')
         # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



def plot_layer_output(layer_output, image):
    # Create a feed-dict which holds the single input image.
    feed_dict = {x: [image]}
    values = session.run(layer_output, feed_dict=feed_dict)
    values_min = np.min(values)
    values_max = np.max(values)
    num_images = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_images))
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)
    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        if i<num_images:
            img = values[0, :, :, i]
            ax.imshow(img, vmin=values_min, vmax=values_max, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()





def  get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]


def plot_image(image):
    fig, axes = plt.subplots(1, 2)
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]
    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')
    plt.show()


#for training
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
_, loss = create_network(training=True)
#adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss, global_step=global_step)

#for testing
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

saver = tf.train.Saver()
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_path=os.path.join(args.save_dir,'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")
    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=args.save_dir)
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

optimize(num_iterations=100000)

print_test_accuracy(show_example_errors=True)


weights_conv1 = get_weights_variable(layer_name='layer_conv1')
plot_conv_weights(weights=weights_conv1, input_channel=1)


#test for an image
img, cls = get_test_image(16)
plot_image(img)
#output after second hidden layer
output_conv1 = get_layer_output(layer_name='layer_conv1')
plot_layer_output(output_conv1, image=img)
#output after 4th hidden layer
output_conv4 = get_layer_output(layer_name='layer_conv4')
plot_layer_output(output_conv4, image=img)
label_pred, cls_pred = session.run([y_pred, y_pred_cls],feed_dict={x: [img]})
np.set_printoptions(precision=3, suppress=True)
# Print the predicted label.
print(label_pred[0])

session.close()





