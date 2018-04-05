'''
Created on Nov 30, 2017

@author: Sara
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

def conv2d(input, output_shape, k=4, s=2, name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input, output_shape, [k, k], stride=s)

def flatten(input, name):
    with tf.variable_scope(name):
        return slim.flatten(input, scope=name)


def fc(input, output_shape, act_fn=tf.nn.relu, name='fc'):
    with tf.variable_scope(name):
        return slim.fully_connected(input, output_shape, activation_fn=act_fn)

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

def conv_layer(scope, input ):
    with tf.variable_scope(scope):
                conv = conv2d(input, 7, 1, 32)
                relu = tf.nn.relu(conv)
                pool = max_pool(relu, 3, 2)            
                print('conv layer: ' + str(pool.get_shape()))