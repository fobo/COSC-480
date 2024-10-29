import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


# Load data
data = pd.read_csv('downloads/data.csv', encoding='ISO-8859-1')

# Initialize scalers for each feature to normalize data
scalers = {col: MinMaxScaler() for col in ["batting_avg", "home_runs", "RBIs", "fantasy_points"]}

# Normalize each column (except player_name and season) independently
for col, scaler in scalers.items():
    data[[col]] = scaler.fit_transform(data[[col]])

# Sort and group data by player
data = data.sort_values(by=["player_name", "season"])

# Create sequences of prior seasons data per player
sequences = []
for player, player_data in data.groupby("player_name"):
    sequence = player_data[["batting_avg", "home_runs", "RBIs", "fantasy_points"]].values
    for i in range(len(sequence) - 1):
        # Append player name with each sequence
        sequences.append((sequence[:i+1], sequence[i+1, -1], player))


# Convert sequences to tensors
class FantasyDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label, player_name = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), player_name


dataset = FantasyDataset(sequences)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)





class FantasyLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(FantasyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # Pack the sequences
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(x_packed)  # hn is the hidden state at the last time step
        out = self.fc(hn[-1])  # Take the last layer's hidden state
        return out.squeeze()

# Adjust collate function to return lengths
def collate_fn(batch):
    sequences, labels, player_names = zip(*batch)  # Separate sequences, labels, and player names

    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = [seq.clone().detach() for seq in sequences]
    sequences = sorted(sequences, key=len, reverse=True)
    
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    
    return sequences_padded, labels, lengths, player_names  # Include player names



# Assuming `dataset` is your original Dataset object
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Define train and test DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize DataLoader with the custom collate_fn
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
model = FantasyLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def mean_absolute_error(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def mean_squared_error(preds, labels):
    return torch.mean((preds - labels) ** 2)

num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_mae = 0.0
    epoch_mse = 0.0
    model.train()  # Set model to training mode
    
    for sequences, labels, lengths, player_names in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences, lengths)
        
        # Compute loss and metrics
        loss = criterion(outputs, labels)
        mae = mean_absolute_error(outputs, labels)
        mse = mean_squared_error(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics for the epoch
        epoch_loss += loss.item()
        epoch_mae += mae.item()
        epoch_mse += mse.item()

    # Average metrics over batches
    avg_loss = epoch_loss / len(dataloader)
    avg_mae = epoch_mae / len(dataloader)
    avg_mse = epoch_mse / len(dataloader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")

import random

def test_model(model, test_loader):
    model.eval()
    cases_to_display = 10

    all_test_results = []

    with torch.no_grad():
        for sequences, labels, lengths, player_names in test_loader:
            outputs = model(sequences, lengths)
            outputs = scalers["fantasy_points"].inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
            labels = scalers["fantasy_points"].inverse_transform(labels.cpu().numpy().reshape(-1, 1)).flatten()

            for i in range(len(outputs)):
                predicted = outputs[i]
                actual = labels[i]
                player_name = player_names[i]  # Access correct player name
                all_test_results.append((player_name, predicted, actual))

    unique_players = list(set([result[0] for result in all_test_results]))
    selected_players = random.sample(unique_players, min(10, len(unique_players)))

    print(f"\nSelected Test Cases (Player, Predicted, Actual):")
    print("Player               | Predicted Fantasy Points  | Actual Fantasy Points")
    print("----------------------------------------------------------------------")
    
    for player in selected_players:
        player_results = [result for result in all_test_results if result[0] == player]
        if player_results:
            predicted = player_results[0][1]
            actual = player_results[0][2]
            print(f"{player:<20} | {predicted:<24.2f} | {actual:.2f}")

# Run the test model function
test_model(model, test_loader)



def predict_next_season(model, player_sequence):
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.tensor(player_sequence, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, input_size]
        lengths = torch.tensor([sequence_tensor.size(1)])  # Lengths tensor
        
        # Make prediction
        prediction = model(sequence_tensor, lengths)
    return prediction.item()

# Example usage of predict_next_season
player_sequence = [[0.281, 13, 60, 377]]  # Replace with actual normalized historical data sequence for testing
predicted_fantasy_points = predict_next_season(model, player_sequence)
predicted_fantasy_points = scalers["fantasy_points"].inverse_transform([[predicted_fantasy_points]]) #turn the fantasy points back into real point values
print(f"Predicted Fantasy Points for Next Season: {predicted_fantasy_points[0][0]:.2f}")
