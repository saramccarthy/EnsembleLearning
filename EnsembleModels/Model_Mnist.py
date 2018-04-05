'''
Created on Nov 30, 2017

@author: Sara
'''
import tensorflow as tf
from layer_utils import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import LoadData as load
from autograd import grad, elementwise_grad

import numpy as np
import os
import math
import cv2

import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

class Model(object):
    def __init__(self,  model_num):
        self.model_num=model_num
        self.num_epoch = 5
        self.batch_size = 128
        self.log_step = 50
        self.learning_rate=5e-4
        self.global_step=tf.Variable(0, trainable=False)
        self.learning_step = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           500, 0.96)
        
        self._build_model()
        
        

    def _model(self):
        x_reshaped = tf.reshape(self.X,[-1,28,28,1])
        x_tiled = tf.tile(x_reshaped, [1,1,1,3])
        sconv_1 = slim.conv2d(x_tiled,5,[5,5])
        spool_1 = slim.max_pool2d(sconv_1,[2,2])
        sconv_2 = slim.conv2d(spool_1,5,[5,5])
        spool_2 = slim.max_pool2d(sconv_2,[2,2])
        sconv_3 = slim.conv2d(spool_2,20,[5,5])
        s_dropout3 = slim.dropout(sconv_3, self.keep_prob)
        output = slim.fully_connected(slim.flatten(s_dropout3), 10, activation_fn=tf.nn.softmax)
        
        self.vars = [ sconv_1, sconv_2, sconv_3, output]
        return output

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 784],name="x-in")
        self.Y = tf.placeholder(tf.float32, [None, 10],name="y-in")
        self.keep_prob = tf.placeholder("float", name="keep-prob")
        self.A = tf.placeholder(tf.bool, [None, 784], name="A")
        self.l2_grads=0.0001

    def _build_optimizer(self):
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss_op)    
       
        
    def _loss(self, labels, logits, grads, A):
        def l2_norm(params):
            flattened = flatten(params, name="grads")
            return np.dot(flattened, flattened)
        # Softmax cross entropy loss 'self.loss_op'
        print np.asarray(grads).shape, A.shape
        mask = tf.boolean_mask(grads,A)
        rightreasons = self.l2_grads * tf.reduce_sum(np.multiply(mask, mask))
        print mask.shape, rightreasons.shape
        self.loss_op = -tf.reduce_sum(labels * tf.log(logits)) + rightreasons    
        return self.loss_op
    
    def input_gradients(self, logits, y=None, scale='log'):
        # log probabilities or probabilities
        if scale is 'log':
            p = lambda x: logits
        else:
            p = lambda x: np.exp(logits)
    
      # max, sum, or individual y
        if y is None: y = 'sum' if scale is 'log' else 'max'
        if y is 'sum':
            p2 = p
        elif y is 'max':
            p2 = lambda x: np.max(p(x), axis=1)
        else:
            p2 = lambda x: p(x)[:, y]
        return elementwise_grad(p2)
    def _build_model(self):
        self._input_ops()
        with tf.variable_scope("model%d" %self.model_num):
    
            # Define input variables
            
    
            # Build a model and get logits
            self.logits = self._model()
    
            self.input_grads = self.calc_input_grads(self.X, self.logits) #tf.gradients(predict, self.X )#
    
            # Compute loss
            loss = self._loss(self.Y, self.logits, self.input_grads, self.A)
            
            # Build optimizer
            self._build_optimizer()
    
            # Compute accuracy
            correct_prediction = tf.equal(tf.argmax(self.logits,1 ), tf.argmax(self.Y, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        #self.input_grads = tf.gradients(loss, self.X )#
    def calc_input_grads(self, x, y, scale='log'):
        return tf.gradients(tf.reduce_sum(tf.log(y)),x )[0]
        # log probabilities or probabilities
        from autograd import grad, elementwise_grad

        p = lambda x: self._model( x)
        p2 = p
      
        return elementwise_grad(p2)
    def train(self, sess, mnist, A=None):
        batchSize = 50
        dropout_p = 0.5
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        losses = []
        accuracies = []
        
        
        for i in range(1000):
    
            batch = mnist.train.next_batch(batchSize, shuffle=False)
            if A is None: Ai = np.zeros_like(batch[0]).astype(bool)
            else: Ai = A[batchSize*i:batchSize*(i+1)] 
            
            [ t , loss] = sess.run([self.train_op, self.loss_op], feed_dict={self.X:batch[0], self.Y:batch[1], self.keep_prob:dropout_p, self.A:Ai})
            if i % 100 == 0 and i != 0:
                trainAccuracy = sess.run(self.accuracy_op, feed_dict={self.X:batch[0], self.Y:batch[1], self.keep_prob:1.0, self.A:Ai})
                print("loss ", loss)
                print("step %d, training accuracy %g"%(i, trainAccuracy))
    def evaluate(self, sess, mnist):
        img = mnist.test.images
        labels = mnist.test.labels
        A = np.zeros_like(img).astype(bool)
        feed_dict = {self.X: img, self.Y : labels,self.keep_prob:1.0, self.A:A}
           
        accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
        return accuracy
   
    def evaluate_img(self, sess, img, labels):
       
        A = np.zeros_like(img).astype(bool)
        feed_dict = {self.X: img, self.Y : labels,self.keep_prob:1.0, self.A:A}
           
        accuracy, predictions = sess.run([self.accuracy_op, self.logits], feed_dict=feed_dict)
        return accuracy, predictions
    def get_grads(self, sess, mnist, n_grads):
        batch = mnist.train.next_batch(n_grads)

        grads = sess.run(self.input_grads, feed_dict={self.X:batch[0], self.Y:batch[1], self.keep_prob:1.0})
        return grads, batch[0]
    
    def get_A_Matrix(self, grads, threshold):
        return np.array([np.abs(g) > threshold*np.abs(g).max() for g in grads])

def visualise(image, grads, A):
    # Reverse the BGR channel to RGB
    print np.asarray(A).shape
    # Initialzie CAM weights
    imageToShow = np.expand_dims(np.reshape(image,[28,28]), axis=-1)
    imageToShow = np.tile(imageToShow, (1,1,3))
    
    CAM = np.expand_dims(np.reshape(grads,[28,28]), axis=-1)
    CAM= np.tile(CAM, (1,1,3))
    A = A
    AN = [ 0 for i in range(len(A))]
    for i in range(len(A)):
        if A[i]:
            AN[i]=1
            
    AN= np.expand_dims(np.reshape(AN,[28,28]), axis=-1)
    AN= np.tile(AN, (1,1,3))
    
    # Passing through ReLU
    CAM = np.maximum(CAM, 0)
    # scale CAM to [0,1]
    CAM /= np.max(CAM)
    # Render the CAM heatmap
    heatmap = cv2.applyColorMap(np.uint8(CAM*255.0), cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(np.uint8(AN*255.0), cv2.COLORMAP_JET)

    # Draw the results figures
    fig = plt.figure(figsize=(10,10))   
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    ax1.imshow(imageToShow)
    ax1.set_title('Input Image')
    ax2.imshow(heatmap)
    ax2.set_title('Grad-CAM')
    ax3.imshow(heatmap2)
    ax3.set_title('Annotation Matrix')

    # Show the resulting image
    plt.show()
    
import pickle

def main2():
    i=0
    A = None
    for i in range(10):    
        # Clear old computation graphs
        tf.reset_default_graph()
        
        sess = tf.Session()
        mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
        
        model = Model(i)
    
        saver = tf.train.Saver()
        #try:
         #   restored = saver.restore(sess,"./models/mnist/model%d.ckpt"%i)
         #   print("Model Loaded")
    
        
        #except:
        model.train(sess,mnist,A)
            
        model_path = saver.save(sess, "./models/mnist/model%d.ckpt"%i)
        print("Model saved in %s" % model_path)
        
            
        grads, x = model.get_grads(sess, mnist, 2000)
        print ("Test Accuracy: ", model.evaluate(sess, mnist))
        A1 = model.get_A_Matrix(grads, 0.3)
        A2 = model.get_A_Matrix(grads, 0.4)
        A3 = model.get_A_Matrix(grads, 0.5)
        A4 = model.get_A_Matrix(grads, 0.6)
        A5 = model.get_A_Matrix(grads, 0.7)
        A6 = model.get_A_Matrix(grads, 0.8)
        print np.sum(A1), np.sum(A4)
    
        visualise(x[0], grads[0], A1[0])
        visualise(x[0], grads[0], A2[0])
        visualise(x[0], grads[0], A3[0])
        visualise(x[0], grads[0], A4[0])
        visualise(x[0], grads[0], A5[0])
        visualise(x[0], grads[0], A6[0])
    
        sess.close()
def main():
    i=0
    A = None
    for i in range(10):    
        # Clear old computation graphs
        tf.reset_default_graph()
        
        sess = tf.Session()
        mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
        
        model = Model(i)
    
        saver = tf.train.Saver()
        #try:
         #   restored = saver.restore(sess,"./models/mnist/model%d.ckpt"%i)
         #   print("Model Loaded")
    
        
        #except:
        model.train(sess,mnist,A)
        
        directory = "./models/mnist/35model%d.ckpt"%i
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = saver.save(sess, "./models/mnist/35model%d.ckpt"%i)
        print("Model saved in %s" % model_path)
        
            
        grads, x = model.get_grads(sess, mnist, 2000)
        print ("Test Accuracy: ", model.evaluate(sess, mnist))
        An = model.get_A_Matrix(grads, 0.27)
        if A is not None:
            A = np.add(A,An)
        else:
            A = An
        #pickle.dump( A, open( "A_model%d.p"%i, "wb" ) )
        
        print np.asarray(grads).shape, np.asarray(A).shape
    
        #visualise(x[0], grads[0], An[0])
        #visualise(x[10], grads[10], An[10])
        #visualise(x[100], grads[100], An[100])
    
        sess.close()
#main()