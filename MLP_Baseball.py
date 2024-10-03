import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# **********************************
# import fetchPlayerData    
# import fetchTestingData
# **********************************
# Create the two above classes which will return our data in a nice way
# for both training and testing

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914), (0.247))  
])

# fetching training dataset from library
train_dataset = [[1,1,1234,1234,123,412,34,1234,1235,4567,54], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

# fetching testing dataset from library
test_dataset = [[1,2, 1234, 126,8567 ,5678 ,5783, 245, 2435], [1, 2, 3, 4, 5, 6, 7, 8, 9]]

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)     
        self.relu1 = nn.ReLU()                            
        self.fc2 = nn.Linear(hidden_size, num_classes)    
        #add more layers maybe

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x
    
input_size = 11*11
hidden_size = 9*9
num_classes = 10

model = SimpleMLP(input_size, hidden_size, num_classes)
print(model)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs in train_loader:
        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs)
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
    for inputs in test_loader:
        outputs = model(inputs)
        _,predicted = torch.max(outputs.data,1)
        total += inputs.size(0)
        correct += (predicted == inputs).sum().item()
        all_predictions.extend(predicted.tolist())
        all_labels.extend(inputs.tolist())
    
accuracy = 100 * correct / total
print(f'Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%')
print('Final Evaluation onTest Set:')
print(f'Accuracy: {accuracy_score(all_labels,all_predictions) * 100:.2f}%')
        
