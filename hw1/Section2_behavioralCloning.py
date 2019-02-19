#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:50:42 2019

@author: liangliang
"""
'''The following code is adapted 
according to https://github.com/Observerspy/CS294/blob/master/hw1/BehavioralCloning.py
and only for self learning
'''

import tensorflow as tf
import os
import numpy as np
import gym 
import tqdm # progress
import time
import math
import logz

class Config(object):
    '''
    Nomally class Config initialize all the parameters
    needed in training time
    '''
    n_features = 11
    n_classes = 3
    dropout = 0.5
    hidden_size_1 = 128
    hidden_size_2 = 256
    hidden_size_3 = 64
    batch_size = 256
    learning_rate = 0.0005
    itera = 20
    training_itera = 20
    envname = 'Hopper-v2'
    max_steps = 1000
    
class NN:
    def add_placeholders(self):
        '''
        tf.placeholder()
        tf.placeholder(
                dtype,
                shape=None,
                name=None
                )
        '''
        self.inputs_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, Config.n_features], name="input")
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, Config.n_classes], name="class")
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout")
        self.is_training = tf.placeholder(tf.bool)
        
    def create_feed_dict(self, input_batch, label_batch = None, dropout = 1, is_training = False):
        '''
        feed data into placehoder using feed_dict
        feed the placehoder above
        Need to know whether label_batch is None, since when training we have label, but we don't need 
        label when predicting
        '''
        if label_batch is None:
            feed_dict = {self.inputs_placeholder : input_batch,
                         self.dropout_placeholder : dropout,
                         self.is_training : is_training}
        else:
            feed_dict = {self.inputs_placeholder : input_batch,
                         self.labels_placeholder : label_batch,
                         self.dropout_placeholder : dropout,
                         self.is_training : is_training}    
        return feed_dict
        
    def add_prediction_op(self):
        '''
        tf.contrib.layers.fully_connected(
        inputs,
        num_outputs,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
        )
        '''
        # Initialize the step as 0 for learning rate decay
        self.global_step = tf.Variable(0)
        # define the 3-layer fully connected NN
        with tf.name_scope('layer_1'):
            hidden_layer1 = tf.contrib.layers.fully_connected(inputs = self.inputs_placeholder,
                                                              num_outputs = Config.hidden_size_1,
                                                              activation_fn=tf.nn.relu)
        with tf.name_scope('layer_2'):
            hidden_layer2 = tf.contrib.layers.fully_connected(inputs = hidden_layer1,
                                                              num_outputs = Config.hidden_size_2,
                                                              activation_fn=tf.nn.relu)
        with tf.name_scope('layer_3'):
            hidden_layer3 = tf.contrib.layers.fully_connected(inputs = hidden_layer2,
                                                              num_outputs = Config.hidden_size_3,
                                                              activation_fn=tf.nn.relu)
        with tf.name_scope('output'):
            pred = tf.contrib.layers.fully_connected(inputs = hidden_layer3,
                                                              num_outputs = Config.n_classes,
                                                              activation_fn=None)
        return pred 
    
    def add_loss_op(self, pred):
        '''
        Gradient Descent Part:
            optimized the loss function 
        tf.losses.mean_squared_error(
                labels,
                predictions)
        '''
        msn = tf.losses.mean_squared_error(labels = self.labels_placeholder, predictions = pred) # the loss function
        tf.summary.scalar('msn', msn)
        return msn
    
    def add_training_op(self, loss):
        '''
        https://byjiang.com/2017/11/26/TensorFlow_BN_Layer/
        detail explaination of extra_update_ops
        https://www.youtube.com/watch?v=BZh1ltr5Rkg
        '''
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            learning_rate = tf.train.exponential_decay(Config.learning_rate, self.global_step, 1000, 0.8, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
        
        return train_op
    
    def train_on_batch(self, sess, inputs_batch, labels_batch, merged, train_writer, i):
        '''
        training part 
        '''
        feed = self.create_feed_dict(inputs_batch, labels_batch, dropout = self.config.dropout, is_training = True)
        rs, _, loss = sess.run([merged, self.train_op, self.loss], feed_dict=feed)
        train_writer.add_summary(rs, i)
        return loss
    
    #def fit(self, sess, train_x, train_y):
    #    loss = self.train_on_batch(sess, train_x, train_y)
    
    def __init__(self, config):
        self.config = config
        self.build()
        
    
    def build(self):
        with tf.name_scope('inputs'):
            self.add_placeholders()
        with tf.name_scope('predict'):
            self.pred = self.add_prediction_op()
        with tf.name_scope('loss'):
            self.loss = self.add_loss_op(self.pred)
        with tf.name_scope('train'):
            self.train_op = self.add_training_op(self.loss)
        
        
    def get_pred(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch, dropout=1, is_training=False)
        p = sess.run(self.pred, feed_dict=feed)
        return p
        
def load(path):
    '''
    ???
    '''
    all = np.load(path)
    X = all["arr_0"]
    y = all["arr_1"]
    y1 = y.reshape(y.shape[0], y.shape[2])
    return X, y1

def main():
    '''
    start session and call functions to run
    '''
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(PROJECT_ROOT, "data/"+Config.envname+".train.npz")
    train_log_path = os.path.join(PROJECT_ROOT, "log/train/")
    logz.configure_output_dir(os.path.join(PROJECT_ROOT, "log/"+Config.envname+"_BC_"+time.strftime("%d-%m-%Y_%H-%M-%S")))
    
    
    X_train, y_train = load(train_path)
    
    print("train size :", X_train.shape, y_train.shape)
    print("=============start training====================")
    
    with tf.Graph().as_default():
        config = Config()
        nn = NN(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep = 10, keep_checkpoint_every_n_hours=0.5)
        # shuffule should be outside of session
        shuffle_batch_x, shuffle_batch_y = tf.train.shuffle_batch(
            [X_train, y_train], batch_size=Config.batch_size, capacity=10000,
            min_after_dequeue=5000, enqueue_many=True)
        
        with tf.Session() as session:
            # Merges all summaries collected in the default graph.
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_log_path, session.graph)
            session.run(init)
            #This class implements a simple mechanism to coordinate the termination of a set of threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)
            
            for j in tqdm.tqdm(range(Config.itera)):
                i = 0
                try:
                    for i in range(int(math.ceil(Config.training_itera * X_train.shape[0] / Config.batch_size))):
                        batch_x, batch_y = session.run([shuffle_batch_x, shuffle_batch_y])
                        loss = nn.train_on_batch(session, batch_x, batch_y, merged, train_writer, i)
                        i += 1
                        if i % 1000 == 0:
                            print("step:", i, "loss:", loss)
                            saver.save(session, os.path.join(PROJECT_ROOT, "model/model_ckpt"), global_step=i)
                except tf.errors.OutOfRangeError:
                    print("")
                finally:
                    coord.request_stop()
                coord.join(threads)
                
                env = gym.make(Config.envname)
                rollouts = 20
                returns = []
                
                
                for _ in range(rollouts):
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = nn.get_pred(session, obs[None, :])
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                        # if args.render:
                        #     env.render()
                        if steps >= Config.max_steps:
                            break
                    returns.append(totalr)

                # print('results for ', Config.envname)
                # print('returns', returns)
                # print('mean return', np.mean(returns))
                # print('std of return', np.std(returns))
                logz.log_tabular('Iteration', j)
                logz.log_tabular('AverageReturn', np.mean(returns))
                logz.log_tabular('StdReturn', np.std(returns))
                logz.dump_tabular()

                
if __name__ == '__main__':
    main()
    
    