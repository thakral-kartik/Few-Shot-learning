# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:21:12 2020

@author: Kartik
"""

import os
import torch
import numpy as np

from tqdm import tqdm
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

#self imports
import model
import config
import dataloader
from model import get_accuracy

def load_vgg_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
    return model

def check_dir():
    if not os.path.isdir('saved_models'):
        print("\nCreating folder to save the trained model")
        os.mkdir('saved_models')
    else:
        print("\nDirectory to save model already exists")
    return True

def train(device, dataset, dataloader, model):
    print("in train")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    images_per_batch = {}
    batch_count, images_per_batch['train'], images_per_batch['test'] = 0, [], []
    with tqdm(dataloader, total=config.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets,
                dataset.num_classes_per_task)
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            loss.backward()
            optimizer.step()
            
            #Just keeping the count here
            batch_count += 1
            images_per_batch['train'].append(train_inputs.shape[1])
            images_per_batch['test'].append(test_inputs.shape[1])

            with torch.no_grad():
                accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

            if batch_idx >= config.num_batches:
                break
    
    print("Number of batches in the dataloader: ", batch_count)
    
    # Save model
    if check_dir() is not None:
        filename = os.path.join('saved_models', 'protonet_cifar_fs_{0}shot_{1}way.pt'.format(config.k, config.n))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
            print("Model saved")
            
    return batch_count, images_per_batch

#main
print("Device: ", config.device)

#x_train, y_train, x_test, y_test = dataloader.load_dataset()
#print("Dataset loaded..")

#trainloader, testloader = dataloader.return_loader(x_train, y_train, x_test, y_test)
#print("Trainloader and Test loader created")

#dataset, dataload = dataloader.load_metadataset()
trainset, trainloader = dataloader.load_meta_trainset()
print("Dataset loaded")

#model = load_model()
#print("Model loaded..")

'''
if torch.cuda.is_available():
    #trainloader = trainloader.to('cuda')
    #testloader = testloader.to('cuda')
    dataload = dataload.to(device)
    model.to('cuda')
'''  
print("before train")

model = model.PrototypicalNetwork(config.in_channel, config.embedding_size, config.hidden_size)
#model = model.load_alexnet()
batch_count, images_per_batch = train(config.device, trainset, trainloader, model)