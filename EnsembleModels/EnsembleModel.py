'''
Created on Dec 27, 2017

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
import pandas as pd
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from Model_Mnist import Model

num_models = 10
models = []
Y = None
logits = None
acc_op = None

class EnsembleModel(object):
    def __init__(self, sess, num_models=10):
        self.models = []
        self.build_model(sess)
    def build_model(self, sess):
        X = tf.placeholder(tf.float32, [None, 784],name="x-in")
        self.Y = tf.placeholder(tf.float32, [None, 10],name="y-in")
    
        for i in range(num_models):
            path = "/Users/Sara/Documents/EnsembleLearn/models/mnist"#/model%d"%i
            model = Model(i)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model%d"%i)
            saver = tf.train.Saver(var_list=vars)
            restored = saver.restore(sess,path+"/35model%d.ckpt"%(i))
            models.append(model)
        
        self.logits, self.model_logits = self.def_logits()
        self.acc_op = self.def_accuracy_op() 
           
        return models    
    def def_logits(self):
        
        #for i in range(num_models):
            #logits = models[i].logits
            #print logits.shape
            #logits_ind = tf.argmax(logits, axis=1)
            #print logits_ind
            #logits_m =tf.Variable(logits)
            #logits[:].assign(0)
            #logits_m=logits_m[logits_ind].assign(1)
            #logits[logits_ind].assign(1)
            #print logits
        p = tf.stack([models[i].logits for i in range(num_models)])
        
        #pmax = tf.argmax(p,axis=1)
        #logits = np.zeros_like(p)
        #logits[np.arange(p.shape), p.argmax(axis=1)] = 1
        
        psum = p[0][:][:]
        for i in range(num_models-1):
            psum += p[i+1][:][:]
        #logits = np.zeros_like(psum)
        #logits[np.arange(len(psum)), psum.argmax(axis=1)] = 1
        
        pnorm = tf.divide(tf.transpose(psum), tf.norm(psum, axis=1))
        logits = np.zeros_like(p[0])
        
        

        #p = np.argmax(counts, axis=2)
        
        logits = tf.transpose(pnorm)
        return logits, p 
    
    def def_accuracy_op(self):
        print self.logits, self.Y
        correct_prediction = tf.equal(tf.argmax(self.logits,1 ), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy_op
    def compare_accuracy(self, sess, img, adv, labels):
        feed_dict = self.get_feed_dict(img, labels)
        acc, logits, m = sess.run([self.acc_op, self.logits, self.model_logits], feed_dict=feed_dict)
        
        feed_dict = self.get_feed_dict(adv, labels)
        acc2, logits2, m2 = sess.run([self.acc_op, self.logits, self.model_logits], feed_dict=feed_dict)
        
        
        d = {"True":np.argmax(labels,1), "Ensemble_true": np.argmax(logits, 1), "Ensemble_adv" : np.argmax(logits2,1)}
        for i in range(len(models)):
            S = "model%d"%i
            Sadv = "adv_model%d"%i
            d.update( {S : np.argmax(m[i],1), Sadv : np.argmax(m2[i],1)})
        df = pd.DataFrame(data=d)
        df.to_csv("./accuracy.csv")
    def get_accuracy(self, sess, img, labels):
        
        feed_dict = self.get_feed_dict(img, labels)
        acc, logits = sess.run([self.acc_op, self.logits], feed_dict=feed_dict)
        
        return acc, logits
    
    def get_feed_dict(self, img, labels=None):
        feed_dict = {}
        for model in models:
            feed_dict.update({model.X : np.reshape(img,[-1,784]), model.keep_prob : 1.0})
        if labels is not None:
            feed_dict.update({self.Y:labels})
        return feed_dict
    
def predict_model(sess, model_num, inputs):
    m = models[model_num]
    A = np.zeros_like(inputs).astype(bool)
    feed_dict = {m.X: inputs,m.keep_prob:1.0, m.A:A}
    prediction = sess.run(m.logits, feed_dict=feed_dict)
    return prediction

def get_predictions(sess, inputs):
    predictions = []
    for i in range(num_models):
        predictions.append(predict_model(sess, i, inputs))
    p = np.asarray(predictions)
    psum = p[0][:][:]
    for i in range(num_models-1):
        psum += p[i+1][:][:]
    pnorm = psum/np.linalg.norm(psum)
    #vals, counts = np.unique(predictions, return_counts=True, axis=2)
    #print vals, counts
    #p = np.argmax(counts, axis=2)
    return pnorm



def main():
    sess = tf.Session()
            
        
    
    models = []
    for i in range(num_models):
        
        #saver = tf.train.import_meta_graph('./models/mnist/model%d/model%d.ckpt.meta'%(i,i), import_scope="model%d"%i)
        model = Model(i)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model%d"%i)
        saver = tf.train.Saver(var_list=vars)
        restored = saver.restore(sess,"./models/mnist/model%d/model%d.ckpt"%(i,i))
        models.append(model)
    print tf.global_variables()
    
    mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
    
    img = mnist.test.images
    labels = mnist.test.labels
    
    logits = get_predictions(img[:2])
  
