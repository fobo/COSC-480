import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg






def plot_metrics(r2, mae, rmse, weight_norm):
    # Create a figure for the plots
    fig, ax = plt.subplots(figsize=(5, 4))

    # Data for plotting
    metrics = ['R² Score', 'Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Weight Norm']
    values = [r2, mae, rmse, weight_norm]

    # Create a bar plot
    ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Values')
    ax.set_title('Model Evaluation Metrics')

    return fig

def display_gui(r2, mae, rmse, weight_norm):
    # Create the main window
    root = tk.Tk()
    root.title("Model Evaluation Metrics")

    # Create a frame for the Matplotlib figure
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Plot the metrics
    fig = plot_metrics(r2, mae, rmse, weight_norm)

    # Create a canvas to display the figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Start the GUI event loop
    root.mainloop()


# Load and process data
data = pd.read_csv('out.csv', encoding='ISO-8859-1')

# Check for any NaN or invalid values
print(data.isna().sum())
data = data.dropna()  # Drop NaNs
data = data.replace([float('inf'), float('-inf')], float('nan')).dropna()

# Standardization of fantasy points
mean_value = data['fantasy_points'].mean()
std_value = data['fantasy_points'].std()
data['normalized_fantasy_points'] = (data['fantasy_points'] - mean_value) / std_value

# Standardize the inputs (batting_avg, home_runs, RBIs)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[['batting_avg', 'home_runs', 'RBIs']])

# Define a small MLP model for each statistic
class SingleStatMLP(nn.Module):
    def __init__(self, input_size):
        super(SingleStatMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# Define the final MLP that takes the output of three MLPs
class CombinedMLP(nn.Module):
    def __init__(self, input_size):
        super(CombinedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create three MLPs for the three statistics
mlp_batting_avg = SingleStatMLP(1)
mlp_home_runs = SingleStatMLP(1)
mlp_RBIs = SingleStatMLP(1)

# Create the combined MLP for final prediction
combined_mlp = CombinedMLP(96)  # 32 outputs from each of the 3 SingleStatMLP

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

mlp_batting_avg.apply(init_weights)
mlp_home_runs.apply(init_weights)
mlp_RBIs.apply(init_weights)
combined_mlp.apply(init_weights)

# Convert data to tensors
X_batting_avg = torch.tensor(normalized_data[:, 0].reshape(-1, 1), dtype=torch.float32)
X_home_runs = torch.tensor(normalized_data[:, 1].reshape(-1, 1), dtype=torch.float32)
X_RBIs = torch.tensor(normalized_data[:, 2].reshape(-1, 1), dtype=torch.float32)
y = torch.tensor(data['normalized_fantasy_points'].values, dtype=torch.float32).view(-1, 1)

# Split data into training and test sets
X_train_ba, X_test_ba, X_train_hr, X_test_hr, X_train_rbi, X_test_rbi, y_train, y_test = train_test_split(
    X_batting_avg, X_home_runs, X_RBIs, y, test_size=0.2, random_state=42
)

# Define optimizer and loss function
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = optim.Adam(list(mlp_batting_avg.parameters()) + 
                       list(mlp_home_runs.parameters()) + 
                       list(mlp_RBIs.parameters()) + 
                       list(combined_mlp.parameters()), lr=0.0001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    mlp_batting_avg.train()
    mlp_home_runs.train()
    mlp_RBIs.train()
    combined_mlp.train()

    optimizer.zero_grad()  # Zero the gradient

    # Forward pass through each MLP
    output_ba = mlp_batting_avg(X_train_ba)
    output_hr = mlp_home_runs(X_train_hr)
    output_rbi = mlp_RBIs(X_train_rbi)

    # Concatenate the outputs
    combined_output = torch.cat((output_ba, output_hr, output_rbi), dim=1)

    # Final prediction using the combined MLP
    final_output = combined_mlp(combined_output)

    # Compute loss
    loss = criterion(final_output, y_train)

    # Backpropagation
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(combined_mlp.parameters(), max_norm=1.0)
    
    optimizer.step()  # Update the weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model on test data
mlp_batting_avg.eval()
mlp_home_runs.eval()
mlp_RBIs.eval()
combined_mlp.eval()


with torch.no_grad():
    # Get predictions on the test set
    output_ba_test = mlp_batting_avg(X_test_ba)
    output_hr_test = mlp_home_runs(X_test_hr)
    output_rbi_test = mlp_RBIs(X_test_rbi)
    
    combined_output_test = torch.cat((output_ba_test, output_hr_test, output_rbi_test), dim=1)
    test_outputs = combined_mlp(combined_output_test)
    
    # Calculate test loss
    test_loss = criterion(test_outputs, y_test)
    
    # Denormalize the predicted and true fantasy points
    predicted_denormalized = test_outputs * std_value + mean_value
    true_denormalized = y_test * std_value + mean_value
    
    # Convert to numpy for metrics calculation
    predicted_np = predicted_denormalized.numpy().flatten()
    true_np = true_denormalized.numpy().flatten()
    
    # Additional stats
    r2 = r2_score(true_np, predicted_np)
    mae = mean_absolute_error(true_np, predicted_np)
    rmse = np.sqrt(test_loss.item()) * std_value
    
    print("Evaluation Results:")
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    print(f"Mean Fantasy Points: {mean_value:.2f}")
    print(f"Standard Deviation: {std_value:.2f}")
    print("\nPredicted vs True Fantasy Points (First 10):")
    
    for i in range(10):  # Limit to first 10 for readability
        print(f"Predicted: {predicted_denormalized[i].item():.2f}, True: {true_denormalized[i].item():.2f}")


    # Print the additional statistics
    print("\nAdditional Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Print weight norms
    weight_norms = sum([p.norm().item() for p in combined_mlp.parameters()])
    print(f"Combined MLP Weight Norm: {weight_norms:.4f}")


    # Now call the display_gui function with these values
    display_gui(r2, mae, rmse, weight_norms)



