# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:44:20 2020

@author: Kartik
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

import model
import config
import dataloader
from model import get_accuracy

def load_model():
    m = model.PrototypicalNetwork(config.in_channel, config.embedding_size, config.hidden_size)
    m.load_state_dict(torch.load(os.path.join('saved_models', 'protonet_omniglot_5shot_5way.pt')))
    
    return m

def test(device, testset, testloader, model):
    model.to(device)
    model.eval()
    acc = []
    with tqdm(testloader, total=config.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)
            
            train_embeddings = model(train_inputs)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets,
                testset.num_classes_per_task)
            
            with torch.no_grad():
                accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
                acc.append(accuracy)
            
            if batch_idx >= config.num_batches:
                break    
    return acc

#main
trainset, trainloader = dataloader.load_meta_trainset()
testset, testloader = dataloader.load_meta_testset()
print("dataset loaded")

m = load_model()
print("Model loaded")

accuracy = test(config.device, testset, testloader, m)
print("Final accuracy: ", accuracy[-1])