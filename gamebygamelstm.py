import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

url = "https://raw.githubusercontent.com/fobo/COSC-480/refs/heads/main/baseballdata.csv"

# Load data directly from GitHub raw URL
data = pd.read_csv(url)

# Normalize features
scalers = {col: MinMaxScaler() for col in data.columns[2:]}
for col, scaler in scalers.items():
    data[col] = scaler.fit_transform(data[[col]])

# Split data using the new function
def split_train_test_by_player(data, test_size=0.2, random_state=42):
    unique_players = data["Player"].unique()
    train_players, test_players = train_test_split(unique_players, test_size=test_size, random_state=random_state)
    train_data = data[data["Player"].isin(train_players)]
    test_data = data[data["Player"].isin(test_players)]
    return train_data, test_data

train_data, test_data = split_train_test_by_player(data)

# Create sequences
def create_sequences(data, sequence_length=5):
    sequences, targets = [], []
    player_seq_map = {}
    for player, group in data.groupby('Player'):
        group = group.sort_values('Date')
        features = group.iloc[:, 2:-1].values
        target = group['fantasy_points'].values
        player_sequences = []
        for i in range(len(features) - sequence_length):
            seq = features[i:i+sequence_length]
            tgt = target[i+sequence_length]
            sequences.append(seq)
            targets.append(tgt)
            player_sequences.append((seq, tgt))
        player_seq_map[player] = player_sequences
    print(f"Total sequences: {len(sequences)}, Players with sequences: {len(player_seq_map)}")
    return sequences, targets, player_seq_map

train_sequences, train_targets, train_player_map = create_sequences(train_data)
test_sequences, test_targets, test_player_map = create_sequences(test_data)

class FantasyDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.float32) for seq in sequences], 
        batch_first=True
    )
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded_sequences, targets

train_dataset = FantasyDataset(train_sequences, train_targets)
test_dataset = FantasyDataset(test_sequences, test_targets)

class FantasyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(FantasyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Hyperparameters
input_size = len(data.columns) - 3  # Exclude Player, Date, and fantasy_points
hidden_size = 64  # Increased hidden size
output_size = 1
num_layers = 3  # Increased number of layers
lr = 0.001
epochs = 10  # Increased number of epochs

model = FantasyLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if epoch % 1 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                predictions = model(sequences)
                loss = criterion(predictions.squeeze(), targets)
                val_loss += loss.item()

        print(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(test_loader):.4f}")


# Function to calculate moving average
def moving_average(values, window_size):
    return np.convolve(values, np.ones(window_size), 'valid') / window_size

# Function to select three random players from the test set and plot predictions vs. actual
def plot_predictions(model, test_loader, scalers, test_data, save_path="results", num_players=3, window_size=10):
    model.eval()
    player_predictions = {player: {'actual': [], 'predicted': []} for player in test_data['Player'].unique()}

    idx = 0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            outputs = model(sequences)
            outputs = scalers['fantasy_points'].inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
            targets = scalers['fantasy_points'].inverse_transform(targets.cpu().numpy().reshape(-1, 1)).flatten()

            for i in range(len(outputs)):
                player_name = test_data['Player'].iloc[idx + i]
                player_predictions[player_name]['actual'].append(targets[i])
                player_predictions[player_name]['predicted'].append(outputs[i])
            idx += len(outputs)

    # Filter out players with no games
    filtered_players = {player: preds for player, preds in player_predictions.items() if preds['actual']}

    if not filtered_players:
        print("No players with available data for testing.")
        return

    selected_players = random.sample(list(filtered_players.keys()), min(num_players, len(filtered_players)))

    # Print actual and predicted values for selected players
    for player in selected_players:
        actual = filtered_players[player]['actual']
        predicted = filtered_players[player]['predicted']
        print(f"\nPlayer: {player}, Number of Games: {len(actual)}")
        print(f"{'Game':>5} {'Actual':>10} {'Predicted':>15}")
        print("-" * 30)
        for i, (a, p) in enumerate(zip(actual, predicted)):
            print(f"{i+1:>5} {a:>10.2f} {p:>15.2f}")

    # Plot predictions for selected players with moving average
    plt.figure(figsize=(12, 6))
    for player in selected_players:
        actual = filtered_players[player]['actual']
        predicted = filtered_players[player]['predicted']
        smoothed_actual = moving_average(actual, window_size)
        smoothed_predicted = moving_average(predicted, window_size)
        plt.plot(smoothed_actual, label=f"{player} - Actual (Smoothed)", linestyle='-', marker='o')
        plt.plot(smoothed_predicted, label=f"{player} - Predicted (Smoothed)", linestyle='--', marker='x')

    plt.xlabel("Game Index (Smoothed)")
    plt.ylabel("Fantasy Points")
    plt.title("Fantasy Points Prediction for Selected Players (Smoothed)")
    plt.legend()

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "fantasy_points_prediction.png")
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")

# Plot predictions for three random players with moving average
plot_predictions(model, test_loader, scalers, test_data)
