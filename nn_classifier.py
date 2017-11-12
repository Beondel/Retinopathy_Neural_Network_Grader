from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from data_wrangler import Data_Wrangler

tf.logging.set_verbosity(tf.logging.INFO)

#convolutional neural network model
def cnn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features["x"], [-1, 960, 1440, 1])

    # first convolutional layer
    conv1 = tf.layers.conv2d( inputs=input_layer,
                              filters=32,
                              kernel_size=[5, 5],
                              padding="same",
                              activation=tf.nn.relu )

    # first pooling layer
    pool1 = tf.layers.max_pooling2d( inputs=conv1,
                                     pool_size=[2, 2],
                                     strides=2 )

    # second convolutional layer
    conv2 = tf.layers.conv2d( inputs=pool1,
                              filters=64,
                              kernel_size=[5, 5],
                              padding="same",
                              activation=tf.nn.relu )

    # second pooling layer
    pool2 = tf.layers.max_pooling2d( inputs=conv2,
                                     pool_size=[2, 2],
                                     strides=2 )

    # flatten the feature map to prepare for dense layer
    pool2_flat = tf.reshape(pool2, [-1, 240 * 360 * 64])

    # dense layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # use dropout to aid convergence
    dropout = tf.layers.dropout( inputs=dense,
                                 rate=0.25,
                                 training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer (units --> amount of labels)
    logits = tf.layers.dense(inputs=dropout, units=5)

    # prediction dictionary for output:
    # classes are the target grades 0 through 4
    # probabilities are the certainty of the network for each class
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # prediction mode - return a prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec( mode=mode,
                                           predictions=predictions )

    # one-hot encoding of input tensor and definition of cost function
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels,
                                            logits=logits )

    # train mode - use gradient descent to train network
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize( loss=loss,
                                       global_step=tf.train.get_global_step() )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval mode (if not train of predict)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy( labels=labels,
                                         predictions=predictions["classes"] )
    }
    return tf.estimator.EstimatorSpec( mode=mode,
                                       loss=loss,
                                       eval_metric_ops=eval_metric_ops )

def main(unused_argv):

    # prep data
    data = Data_Wrangler()
    train_data = data.get_training_features()
    train_labels = data.get_training_labels()
    eval_data = data.get_testing_features()
    eval_labels = data.get_testing_labels()

    # construct and save the estimator for training/evaluation
    retinopathy_classifier = tf.estimator.Estimator( model_fn=cnn,
                                                     model_dir="./model" )

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log,
                                               every_n_iter=25 )

    train_input_fn = tf.estimator.inputs.numpy_input_fn( x={"x": train_data},
                                                         y=train_labels,
                                                         batch_size=10,
                                                         num_epochs=None,
                                                         shuffle=True )

    retinopathy_classifier.train( input_fn=train_input_fn,
                                  steps=20000,
                                  hooks=[logging_hook] )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn( x={"x": eval_data},
                                                        y=eval_labels,
                                                        num_epochs=1,
                                                        shuffle=False )

    print(mnist_classifier.evaluate(input_fn=eval_input_fn))




if __name__ == "__main__":
    tf.app.run()
