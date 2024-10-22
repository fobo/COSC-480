# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
#import pandas as pd

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batchsize = 100
training_data = datasets.FashionMNIST(root="../fashion_mnist", train = True, transform=transforms.toTensor())
test_data = datasets.FashionMNIST(root="../fashion_mnist", train = False, transform=transforms.toTensor())
 
train_dataloader = DataLoader(training_data, batch_size=batchsize)
test_dataloader = DataLoader(test_data, batch_size=batchsize)

sequence_len = 1
input_len = 1
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 5
learning_rate = 0.01

class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_class, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
    def forward(self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out,_ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:,-1,:])
        return out
    
    
model = LSTM(input_len, hidden_size, num_classes, num_layers)


loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(num_epochs, model, train_dataloader):
    total_steps = len(train_dataloader)
    
    for epoch in range(num_epochs):
        for batch, (image,labels) in enumerate(train_dataloader):
            images = images.reshape(-1, sequence_len, input_len)
            
            output = model(images)
            loss = loss_func(output,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch+1)%100 == 0:
                print(f"Epoch: {epoch+1}; Batch {batch+1} / {total_steps}; Loss: {loss.item():>4d}")
                
train()

