# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:56:04 2020

@author: Kartik
"""

import config

from keras.datasets import mnist, cifar100
from torchmeta.datasets.helpers import omniglot, cifar_fs, doublemnist, CIFARFS
from torchmeta.datasets import pascal5i, Pascal5i
from torchmeta.utils.data import BatchMetaDataLoader


def load_meta_trainset():
    '''
    dataset = Omniglot("data",
                   # Number of ways
                   num_classes_per_task=5,
                   # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                   target_transform=Categorical(num_classes=5),
                   # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   download=True)
    '''
    
    trainset = omniglot("data", ways=config.n, shots=config.k, test_shots=15, shuffle=False, meta_train=True, download=True)
    trainloader = BatchMetaDataLoader(trainset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    #trainset = Pascal5i("data", num_classes_per_task=config.n, meta_train=True, download=True)
    #trainloader = BatchMetaDataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
	#trainset = CIFARFS("data", ways=config.n, shots=config.k, test_shots=15, shuffle=False, meta_train=True, download=True)
    #trainloader = BatchMetaDataLoader(trainset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
	
    return trainset, trainloader

def load_meta_testset():
    testset = omniglot("data", ways=config.n, shots=config.k, test_shots=15, shuffle=False, meta_test=True, download=True)
    testloader = BatchMetaDataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    #testset = Pascal5i("data", num_classes_per_task=config.n, meta_test=True, download=True)
    #testloader = BatchMetaDataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
	
	#testset = CIFARFS("data", ways=config.n, shots=config.k, test_shots=15, shuffle=False, meta_test=True, download=True)
    #testloader = BatchMetaDataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return testset, testloader
