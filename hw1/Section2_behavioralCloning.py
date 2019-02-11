#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:50:42 2019

@author: liangliang
"""

import tensorflow as tf
import os
import numpy as np
import gym 
import tqdm # progress
import time
import math

class Config(object):
    n_features = 11
    n_classes = 3
    droupout = 0.5
    hidden_size_1 = 128
    hidden_size_2 = 256
    hidden_size_3 = 64
    batch_size = 256
    learning_rate = 0.0005
    itera = 20
    training_itera = 20
    envname = 'Hopper-v1'
    max_steps = 1000
    
class NN(obejct):
    def add_placeholders():
    
    def create_feed_dict():
    
    def add_prediction_op():
    
    def add_loss_op():
        
    def add_training_op():
        
    def train_on_batch():
        
    def __init__():
    
    def build(self):
        
    def get_pred():

def load():

def main():

if __name__ == '__main':
    main()
    
    