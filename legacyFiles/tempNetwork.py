# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
#from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
#import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081,))
    ])

#
#       Training and Test datasets linked here, currently commeted out are the MNIST digit data sets
#
#train_dataset = MNIST(root='./data',train=True,download = True, transform = transform)
#test_dataset = MNIST(root='./data',train=False,download = True, transform = transform)
try:
            with open("PlayerList.txt", "r") as file1:
                full_player_list = file.read()
            player_strings = full_player_list.split("(?<=[a-zA-Z]{3}\\d{2})")
            self.baseball_players = [baseBallPlayer.BaseBallPlayer(info) for info in player_strings]

            with open("PitcherList.txt", "r") as file2:
                full_player_list = file.read()
            pitcher_strings = full_player_list.split("(?<=[a-zA-Z]{3}\\d{2})")
            self.pitchers = [pitcher.Pitcher(info) for info in pitcher_strings]
    
pitchers_train_dataset = MNIST(root='PlayerList.txt',train=True,download = True, transform = transform)
pitchers_test_dataset = MNIST(root='PlayerListTest.txt',train=False,download = True, transform = transform)

pitchers_train_dataset = MNIST(root='PitcherList.txt',train=True,download = True, transform = transform)
pitchers_test_dataset = MNIST(root='PitcherListTrain.txt',train=False,download = True, transform = transform)

batch_size = 64

## error coming from old MNIST datasets being commented out
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle = False)


class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x = x.view(-1,28*28)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#We will need to decide how big our input, hidden, and outputs are going to be
#input_size = 28*28
#hidden_size = 128
#num_classes = 10
model = MLP(input_size, hidden_size, num_classes)
print(model)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        #forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)
        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())
    
accuracy = 100 * correct / total
print(f'Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%')
print('Final Evaluation onTest Set:')
print(f'Accuracy: {accuracy_score(all_labels,all_predictions) * 100:.2f}%')
        
