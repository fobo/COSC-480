import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('out.csv', encoding='ISO-8859-1')
 #Fix once CSV data is done

 
# Standardization
mean_value = data['fantasy_points'].mean()
std_value = data['fantasy_points'].std()

data['normalized_fantasy_points'] = (data['fantasy_points'] - mean_value) / std_value

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[['batting_avg', 'home_runs', 'RBIs']])



class FantasyMLP(nn.Module):
    def __init__(self, input_size):
        super(FantasyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)         # Second hidden layer
        self.fc3 = nn.Linear(32, 1)          # Output layer (e.g., predict fantasy points)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

X = torch.tensor(normalized_data, dtype=torch.float32)
y = torch.tensor(data['normalized_fantasy_points'].values, dtype=torch.float32).view(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = FantasyMLP(input_size=X_train.shape[1])
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()  # Zero the gradient
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')


new_data = torch.tensor([[0.300, 25, 75]], dtype=torch.float32).reshape(1, -1)
prediction = model(new_data)
print(f'Predicted Fantasy Points: {prediction.item():.2f}')
print(f"Mean Value: {mean_value}")
print(f"Stand Deviation:  {std_value}")

predicted_normalized = 27.14
mean_value = 259.41
std_value = 198.63

predicted_fantasy_points = (predicted_normalized * std_value) + mean_value
print(predicted_fantasy_points)
