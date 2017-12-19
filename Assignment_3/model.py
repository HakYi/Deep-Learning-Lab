
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


# In[2]:


class ConvNet(tfe.Network):
    
    def __init__(self,cub_siz,pub_siz,hist_len,logits_units,num_filt1=32,kernel_size1=5,num_filt2=64,kernel_size2=5,
                 pool_size=2,dense_units=1024,dropout_rate=0.4):
        super(ConvNet, self).__init__()
        self._input_shape = [-1, cub_siz*pub_siz, cub_siz*pub_siz, hist_len]
        self.conv1 = self.track_layer(tf.layers.Conv2D(filters=num_filt1,kernel_size=kernel_size1,padding="same",activation=tf.nn.relu))
        self.maxpool1 = self.track_layer(tf.layers.MaxPooling2D(pool_size=pool_size, strides=2))
        self.conv2 = self.track_layer(tf.layers.Conv2D(filters=num_filt2,kernel_size=kernel_size2,padding="same",activation=tf.nn.relu))
        self.maxpool2 = self.track_layer(tf.layers.MaxPooling2D(pool_size=pool_size, strides=2))
        self.dense1 = self.track_layer(tf.layers.Dense(units=dense_units, activation=tf.nn.relu))
        self.dropoutlayer = self.track_layer(tf.layers.Dropout(rate=dropout_rate))
        self.logits = self.track_layer(tf.layers.Dense(units=logits_units))
        
        ds = tf.data.Dataset.from_tensor_slices((np.zeros((1,2500),dtype=np.float32),np.zeros((1,5),np.int32)))
        for (image,label) in tfe.Iterator(ds):
            self.predict(image,training=False)
    
    def predict(self, inputs, training):
        """Actually runs the model."""
        input_layer = tf.reshape(inputs, self._input_shape)
        result = self.conv1(input_layer)
        result = self.maxpool1(result)
        result = self.conv2(result)
        result = self.maxpool2(result)
        result = tf.layers.flatten(result)
        if training:
            result = self.dropoutlayer(result)
        result = self.logits(result)
        return result
    
    def loss(self, predictions, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    
    def compute_accuracy(self, predictions, labels):
        return tf.reduce_sum(tf.cast(tf.equal(
              tf.argmax(predictions, axis=1,
                        output_type=tf.int64),
              tf.argmax(labels, axis=1,
                        output_type=tf.int64)),
                dtype=tf.float32)) / float(predictions.shape[0].value)
    
    def train_one_epoch(self, optimizer, dataset, n_minibatches, log_interval):
        """Trains model on `dataset` using `optimizer`."""
        tf.train.get_or_create_global_step()

        def model_loss(images, labels):
            prediction = self.predict(images, training=True)
            loss_value = self.loss(prediction, labels)
            tf.contrib.summary.scalar('loss', loss_value)
            tf.contrib.summary.scalar('accuracy',
                                      self.compute_accuracy(prediction, labels))
            return loss_value

        for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                batch_model_loss = functools.partial(model_loss, images, labels)
                optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
            if log_interval and batch % log_interval == 0:
                print('Batch #%d\tLoss: %.6f' % (batch, batch_model_loss()))
            if batch == n_minibatches-1:
                return
    
    def test(self, dataset):
        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tfe.metrics.Mean('loss')
        accuracy = tfe.metrics.Accuracy('accuracy')
        
        for (images, labels) in tfe.Iterator(dataset):
            predictions = self.predict(images, training=False)
            avg_loss(self.loss(predictions, labels))
            accuracy(tf.argmax(labels, axis=1, output_type=tf.int64),tf.argmax(predictions, axis=1, output_type=tf.int64))
            break
        print('Validation set: Average loss: %.4f, Accuracy: %.2f%%\n' % (avg_loss.result(), 100 * accuracy.result()))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', avg_loss.result())
            tf.contrib.summary.scalar('accuracy', accuracy.result())
    
    def train(self,X_train,y_train,X_valid,y_valid,batch_size=50,n_minibatches=500,num_epochs=10,learning_rate=1e-3,
              dataset_size=None,log_interval=None,no_gpu=True,checkpoint_dir=None):
        
        if no_gpu or tfe.num_gpus() <= 0:
            device = '/cpu:0'
        else:
            device = '/gpu:0'
        print('Using device %s.' % (device))
        
        # create appropriate datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid,y_valid))
        if dataset_size == None:
            dataset_size = X_train.shape[0]
        train_ds = train_ds.shuffle(dataset_size).batch(batch_size)
        valid_ds = valid_ds.batch(X_valid.shape[0])
        
        # create optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        
        # store output during training
        out_dir = '\\tmp\\tensorflow\\NeuralPlanner\\output'
        
        train_dir = os.path.join(out_dir,'train')
        test_dir = os.path.join(out_dir,'eval')
        tf.gfile.MakeDirs(out_dir)
        summary_writer = tf.contrib.summary.create_file_writer(train_dir,flush_millis=10000)
        test_summary_writer = tf.contrib.summary.create_file_writer(test_dir,flush_millis=10000,name='test')
        dir_copy = checkpoint_dir
        checkpoint_dir = os.path.join(dir_copy,'train')
        checkpoint_dir_save = os.path.join(dir_copy,'save')
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint_prefix_save = os.path.join(checkpoint_dir_save, 'ckpt')
        
        # store data attributed to learning curve 
        training_accuracy = np.zeros(num_epochs)
        training_loss = np.zeros(num_epochs)
        valid_accuracy = np.zeros(num_epochs)
        valid_loss = np.zeros(num_epochs)
        
        with tf.device(device):
            for epoch in range(1,num_epochs+1):
                # return should be learning curve data
                curr_pred_train = self.predict(X_train,training=False)
                curr_pred_valid = self.predict(X_valid,training=False)
                training_accuracy[epoch-1] = self.compute_accuracy(curr_pred_train,y_train)
                training_loss[epoch-1] = self.loss(curr_pred_train,y_train)
                valid_accuracy[epoch-1] = self.compute_accuracy(curr_pred_valid,y_valid)
                valid_loss[epoch-1] = self.loss(curr_pred_valid,y_valid)
                
                with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_dir)):
                    global_step = tf.train.get_or_create_global_step()
                    start = time.time()
                    with summary_writer.as_default():
                        self.train_one_epoch(optimizer, train_ds, n_minibatches, log_interval)
                    end = time.time()
                    print('\nTrain time for epoch #%d (global step %d): %f' % (epoch, global_step.numpy(), end - start))
                with test_summary_writer.as_default():
                    self.test(valid_ds)
                all_variables = (self.variables + optimizer.variables() + [global_step])
                tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
                tfe.Saver(self.variables).save(checkpoint_prefix_save, global_step=global_step)
        return training_accuracy,training_loss,valid_accuracy,valid_loss

