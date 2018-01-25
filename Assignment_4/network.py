# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:30:20 2018

@author: Hakan
"""


# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# In[2]:
class ConvNet():
    
    def __init__(self,cub_siz,pob_siz,hist_len,logits_units,num_filt1=32,kernel_size1=5,num_filt2=64,kernel_size2=5,
                 pool_size=2,dense_units=1024,dropout_rate=0.4,learning_rate=0.001):
        self.dropout_rate = dropout_rate
        self.state_siz = cub_siz*pob_siz
        self.hist_len = hist_len
        helper = int((self.state_siz-pool_size)/2)+1
        helper = int((helper-pool_size)/2)+1
        
        self.x = tf.placeholder(tf.float32, shape=(None, (cub_siz*pob_siz)**2*hist_len))
        self.u = tf.placeholder(tf.float32, shape=(None, logits_units))
        self.ustar = tf.placeholder(tf.float32, shape=(None, logits_units))
        self.xn = tf.placeholder(tf.float32, shape=(None, (cub_siz*pob_siz)**2*hist_len))
        self.r = tf.placeholder(tf.float32, shape=(None, 1))
        self.term = tf.placeholder(tf.float32, shape=(None, 1))
        self.keep_prob = tf.placeholder(tf.float32)
        
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([kernel_size1,kernel_size1,hist_len,num_filt1])),
            'wc2': tf.Variable(tf.random_normal([kernel_size2,kernel_size2,num_filt1,num_filt2])),
            'wd1': tf.Variable(tf.random_normal([helper**2*num_filt2, dense_units])),
            'out': tf.Variable(tf.random_normal([dense_units, logits_units]))
        }
        
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([num_filt1])),
            'bc2': tf.Variable(tf.random_normal([num_filt2])),
            'bd1': tf.Variable(tf.random_normal([dense_units])),
            'out': tf.Variable(tf.random_normal([logits_units]))
        }
        
        self.Q = self.conv_net(self.x,self.weights,self.biases,self.keep_prob)
        self.Qn = self.conv_net(self.xn,self.weights,self.biases,self.keep_prob)
        
        self.loss_op = self.Q_loss(self.Q, self.u, self.Qn, self.ustar, self.r, self.term)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        
        self.init = tf.global_variables_initializer()

        
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def maxpool2d(self,x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
    
    def conv_net(self,x, weights, biases, dropout):
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, self.state_siz, self.state_siz, self.hist_len])
    
        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
    
        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)
    
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
    
    def Q_loss(self, Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
        """
        All inputs should be tensorflow variables!
        We use the following notation:
           N : minibatch size
           A : number of actions
        Required inputs:
           Q_s: a NxA matrix containing the Q values for each action in the sampled states.
                This should be the output of your neural network.
                We assume that the network implements a function from the state and outputs the 
                Q value for each action, each output thus is Q(s,a) for one action 
                (this is easier to implement than adding the action as an additional input to your network)
           action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                          (e.g. each row contains only one 1)
           Q_s_next: a NxA matrix containing the Q values for the next states.
           best_action_next: a NxA matrix with the best current action for the next state
           reward: a Nx1 matrix containing the reward for the transition
           terminal: a Nx1 matrix indicating whether the next state was a terminal state
           discount: the discount factor
        """
        # calculate: reward + discount * Q(s', a*),
        # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
        target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
        # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
        #       use it as the target for Q_s
        target_q = tf.stop_gradient(target_q)
        # calculate: Q(s, a) where a is simply the action taken to get from s to s'
        selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
        loss = tf.reduce_sum(tf.square(target_q - selected_q))    
        return loss
    
#Q_Network = ConvNet(5,5,4,5,32,5,64,5,2,512,0.4,0.001)