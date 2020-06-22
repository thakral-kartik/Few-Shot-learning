# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:24:14 2020

@author: Kartik
"""

#number of classes
n = 5

#number of images per class
k = 1

#number of channels in the input image of the dataset
in_channel = 1

#learning rate
learning_rate = 1e-3

#Number of tasks in a mini-batch of tasks (default: 16).
batch_size = 16

#Dimension of the embedding/latent space (default: 64).
embedding_size = 64

#Number of channels for each convolutional layer (default: 64).
hidden_size = 64

#Number of batches the prototypical network is trained over (default: 100).
num_batches = 100

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")