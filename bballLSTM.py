import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data, change this to your directory
data = pd.read_csv('downloads/full_data.csv', encoding='ISO-8859-1')

# Define a starting year
starting_year = 2015  # You can change this year as needed

# Filter data to include only seasons from the specified starting year
filtered_data = data[data['Season'] >= starting_year]

def filter_players_by_season_count(data, min_seasons):
    # Group by player name and count the unique seasons
    player_season_counts = data.groupby("Name")["Season"].nunique()

    # Filter players who have at least the specified number of seasons
    eligible_players = player_season_counts[player_season_counts >= min_seasons].index

    # Filter the original data to include only eligible players
    filtered_data = data[data["Name"].isin(eligible_players)]
    
    # Display summary
    print(f"Filtering players with at least {min_seasons} seasons.")
    print(f"Total eligible players: {len(eligible_players)}")
    print(f"Filtered dataset records: {len(filtered_data)}")

    return filtered_data


# Example usage
min_seasons = 4  # Set minimum season count
filtered_data_seasons = filter_players_by_season_count(filtered_data, min_seasons)


# Initialize scalers for each feature to normalize data
scalers = {col: MinMaxScaler() for col in ["AVG", "HR", "1B", "2B", "3B", "HR.1", "R", "RBI", 
                                           "BB", "HBP", "SB", "CS", "SV", "W", "IP", "ERA", "fantasy_points"]}

# Normalize each column independently
for col, scaler in scalers.items():
    filtered_data_seasons[[col]] = scaler.fit_transform(filtered_data_seasons[[col]])

# Sort and create sequences as before

filtered_data_seasons = filtered_data_seasons.sort_values(by=["Name", "Season"])


# Updated sequences creation
sequences = []
for player, player_data in filtered_data_seasons.groupby("Name"):
    if player_data["Season"].nunique() >= min_seasons:  # Ensure minimum seasons condition
        sequence = player_data[["AVG", "HR", "1B", "2B", "3B", "HR.1", "R", "RBI", "BB", "HBP", 
                                "SB", "CS", "SV", "W", "IP", "ERA", "fantasy_points"]].values
        for i in range(len(sequence) - 1):
            sequences.append((sequence[:i+1], sequence[i+1, -1], player))

# Test check for sequences#################
player_season_counts_in_sequences = {}
for _, _, player in sequences:
    player_season_counts_in_sequences[player] = player_season_counts_in_sequences.get(player, 0) + 1

print("Players in sequences with fewer than min_seasons:")
for player, season_count in player_season_counts_in_sequences.items():
    if season_count < min_seasons:
        print(f"{player}: {season_count} seasons")


###############end testing################################




def count_seasons_per_player(data):
    # Count total records
    total_records = len(data)
    
    # Group data by 'Name' and count unique seasons for each player
    player_season_counts = data.groupby("Name")["Season"].nunique()
    
    # Count players for each season count
    season_counts_summary = player_season_counts.value_counts().sort_index()
    
    # Display total records and total players
    print(f"Total Records: {total_records}")
    print(f"Total Players: {len(player_season_counts)}")
    print("\nPlayer           | Seasons")
    print("--------------------------")
    #for player, season_count in player_season_counts.items():
    #    print(f"{player:<15} | {season_count}")
    
    # Display summary of how many players have each number of seasons
    print("\nSeasons | Number of Players")
    print("---------------------------")
    for seasons, count in season_counts_summary.items():
        print(f"{seasons:<7} | {count}")

    return season_counts_summary

# Example usage
#season_counts = count_seasons_per_player(filtered_data)

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
    def __init__(self, input_size=17, hidden_size=64, num_layers=5):  # input_size updated
        super(FantasyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(x_packed)
        out = self.fc(hn[-1])
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

num_epochs = 10
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
import matplotlib.pyplot as plt

# Step 1: Select and display players with season counts, excluding those with only 1 season
def test_model_with_season_counts(model, test_loader, cases_to_display=10):
    model.eval()
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

    # Count seasons for each player
    player_season_counts = {}
    for result in all_test_results:
        player_name = result[0]
        if player_name in player_season_counts:
            player_season_counts[player_name] += 1
        else:
            player_season_counts[player_name] = 1

    # Filter out players with only 1 season
    eligible_players = [player for player, count in player_season_counts.items() if count > 3]
    
    if not eligible_players:
        print("No players with more than one season available.")
        return []

    selected_players = random.sample(eligible_players, min(cases_to_display, len(eligible_players)))

    print(f"\nSelected Test Cases (Player, Predicted, Actual, Total Seasons):")
    print("Player               | Predicted Fantasy Points  | Actual Fantasy Points | Total Seasons")
    print("--------------------------------------------------------------------------------------")
    
    for player in selected_players:
        player_results = [result for result in all_test_results if result[0] == player]
        if player_results:
            predicted = player_results[0][1]
            actual = player_results[0][2]
            total_seasons = len(player_results)
            print(f"{player:<20} | {predicted:<24.2f} | {actual:<22.2f} | {total_seasons}")

    return selected_players

# Step 2: Plot predictions for the selected players
def test_model_for_selected_players(model, test_loader, selected_players):
    model.eval()
    player_predictions = {player: {'actual': [], 'predicted': []} for player in selected_players}
    
    with torch.no_grad():
        for sequences, labels, lengths, player_names in test_loader:
            outputs = model(sequences, lengths)
            predictions = scalers["fantasy_points"].inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
            actuals = scalers["fantasy_points"].inverse_transform(labels.cpu().numpy().reshape(-1, 1)).flatten()
            
            for i, player_name in enumerate(player_names):
                if player_name in selected_players:
                    player_predictions[player_name]['actual'].append(actuals[i])
                    player_predictions[player_name]['predicted'].append(predictions[i])

    # Plot predictions for each selected player
    plt.figure(figsize=(12, 6))
    for player, results in player_predictions.items():
        plt.plot(results['actual'], label=f"{player} - Actual", linestyle='-', marker='o')
        plt.plot(results['predicted'], label=f"{player} - Predicted", linestyle='--', marker='x')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Fantasy Points")
    plt.title("Fantasy Points Prediction for Selected Players")
    plt.legend()
    plt.show()

# Run the selection and plotting
selected_players = test_model_with_season_counts(model, test_loader)
if selected_players:  # Only run if there are players to plot
    test_model_for_selected_players(model, test_loader, selected_players)
